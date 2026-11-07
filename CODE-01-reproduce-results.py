"""Reproduce every value reported in the letter. Writes
RESULTS-01-manuscript-value-check.csv and exits 1 on any mismatch."""
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Published inputs
PREV_BLACK, PREV_WHITE = 102.17, 33.54   # diagnosed prevalence per 100,000, ref 1
REGISTRY_BLACK, REGISTRY_WHITE = 232, 707  # CARS Table 3 (ref 8): African American/Black and Caucasian
REGISTRY_RACE_REPORTED, REGISTRY_ASIAN, REGISTRY_OTHER = 967, 18, 10  # same table, ATTR column
CLAIMS_BLACK_PCT, CLAIMS_WHITE_PCT = 28.8, 54.1  # diagnosed claims cohort, ref 1
SEED, N_BOOT, RADIUS_MI = 20260929, 20000, 50

trials = pd.read_csv("DATA-01-pooled-trials.csv")
landmark = pd.read_csv("DATA-02-landmark-trials.csv")
sites = pd.read_csv("DATA-03-us-sites.csv")
county = pd.read_csv("DATA-04-county-population.csv", dtype={"fips": str})
nhanes = pd.read_csv("DATA-05-nhanes-creatinine.csv")
audit = pd.read_csv("DATA-06-reporting-audit.csv")


CHECKS = []


def _norm(x):
    return re.sub(r"[,\s]", "", str(x))


def show(label, value, reported, ok=None, rule="equal at printed precision"):
    ok = _norm(value) == _norm(reported) if ok is None else bool(ok)
    CHECKS.append(dict(quantity=label, reported_in_manuscript=reported, recomputed_from_deposit=value,
                       rule=rule, status="PASS" if ok else "FAIL"))
    print(f"{'' if ok else 'FAIL '}{label:<58} {value:>10}   (letter: {reported})")


def check(label, ok, detail=""):
    CHECKS.append(dict(quantity=label, reported_in_manuscript="", recomputed_from_deposit=detail,
                       rule="input validation", status="PASS" if ok else "FAIL"))
    if not ok:
        print(f"FAIL {label}: {detail}")


# 0. Input validation
EXPECTED = {"ATTR-ACT": ("NCT01994889", 0), "APOLLO-B": ("NCT03997383", 1), "ATTRibute-CM": ("NCT03860935", 0),
            "CARDIO-TTRansform": ("NCT04136171", 6), "HELIOS-B": ("NCT04153149", 1)}  # trial: (NCT, randomized not in race population)
check("Pooled trials are the 5 expected trials", dict(zip(trials.trial, trials.nct_id)) == {k: v[0] for k, v in EXPECTED.items()},
      ", ".join(trials.trial))
check("Race rows sum to each race population", (trials.black + trials.white + trials.other_or_not_reported == trials.race_population).all())
check("Randomized minus race population is 0, 1, 0, 6, 1",
      all(r.randomized - r.race_population == EXPECTED[r.trial][1] for r in trials.itertuples()))
check("Denominators positive and at least numerators",
      (trials.variant_denominator >= trials.variant_attr).all() and (trials.us_denominator.dropna() >= trials.us_enrolled.dropna()).all())
check("DATA-02 has 23 unique trials", landmark.nct_id.is_unique and len(landmark) == 23)
for col, allowed in [("tier", {"A", "B1", "B2"}), ("reports_race", {"yes", "no"}), ("ongoing", {"yes", "no"}),
                     ("design", {"randomized", "open-label extension", "nonrandomized open-label"})]:
    check(f"DATA-02 {col} complete and in allowed values", landmark[col].notna().all() and set(landmark[col]) <= allowed)
check("DATA-02 pooled trials match DATA-01", set(landmark.nct_id[landmark.analysis == "pooled"]) == set(trials.nct_id))
check("DATA-02 pooled trials all report race", (landmark.reports_race[landmark.analysis == "pooled"] == "yes").all())
check("DATA-06 covers exactly the 5 pooled trials", set(audit.trial) == set(EXPECTED))
check("DATA-06 values present", audit.value.notna().all())
per_trial = ["race efficacy subgroup reported", "representativeness table"]
per_table = ["African ancestry or p.V142I cited in race row", "describes cohort as representative",
             "Black proportion compared with a disease-burden benchmark"]
for t in EXPECTED:
    a = audit[audit.trial == t]
    has_table = (a.value[a.item == "representativeness table"] == "yes").any()
    for item in per_trial + (per_table if has_table else []):
        rows = a[a.item == item]
        check(f"DATA-06 {t}: one '{item}' row with yes/no", len(rows) == 1 and rows.value.isin(["yes", "no"]).all())
    check(f"DATA-06 {t}: subgroup rows equal DATA-01 count",
          (a.item == "efficacy subgroup dimension").sum() == int(trials.set_index("trial").efficacy_subgroup_dimensions[t]))
    check(f"DATA-06 {t}: table flag agrees with DATA-01",
          has_table == (trials.set_index("trial").ancestry_or_v142i_in_race_row[t] != "no table"))


# 1. Trial selection
show("Landmark ATTR-CM trials", len(landmark), "23")
show("Landmark trials reporting race", (landmark.reports_race == "yes").sum(), "9")
show("Trials without race data that are ongoing", ((landmark.reports_race == "no") & (landmark.ongoing == "yes")).sum(), "12")
show("Pooled trials", (landmark.analysis == "pooled").sum(), "5")

# 2. Pooled Black proportion, cluster bootstrap and leave-one-out
b, w = trials.black.to_numpy(), trials.white.to_numpy()
pooled = 100 * b.sum() / (b.sum() + w.sum())
idx = np.random.default_rng(SEED).integers(0, len(trials), size=(N_BOOT, len(trials)))
boot = 100 * b[idx].sum(1) / (b[idx].sum(1) + w[idx].sum(1))
lo, hi = np.percentile(boot, [2.5, 97.5])
loo = [100 * (b.sum() - b[i]) / (b.sum() + w.sum() - b[i] - w[i]) for i in range(len(trials))]
show("Black of Black or White participants", f"{b.sum()} of {b.sum() + w.sum()}", "282 of 3,295")
show("Pooled Black proportion, % (95% CI)", f"{pooled:.2f} ({lo:.2f}-{hi:.2f})", "8.56 (6.64-11.61)")
show("Leave-one-trial-out range, %", f"{min(loo):.1f}-{max(loo):.1f}", "7.6-9.3")
trials["pct"] = 100 * trials.black / (trials.black + trials.white)

# 3. Benchmarks
def share(bl, wh):
    return 100 * bl / (bl + wh)

def band_sum(df, race, bands):
    return df[[f"{race}_{x}" for x in bands]].to_numpy().sum()

age = {"55": ["55_64", "65_74", "75_84", "85p"], "65": ["65_74", "75_84", "85p"], "75": ["75_84", "85p"]}
nat = {k: share(band_sum(county, "black", v), band_sum(county, "white", v)) for k, v in age.items()}


def miles(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


catch, pool_b, pool_w = {}, 0, 0
for nct, s in sites.groupby("nct_id"):
    d = miles(s.lat.to_numpy()[:, None], s.lon.to_numpy()[:, None], county.lat.to_numpy()[None], county.lon.to_numpy()[None])
    inside = county[(d <= RADIUS_MI).any(0)]
    cb, cw = band_sum(inside, "black", age["65"]), band_sum(inside, "white", age["65"])
    catch[nct] = share(cb, cw)
    pool_b, pool_w = pool_b + cb, pool_w + cw  # counties counted once per trial
catchment = share(pool_b, pool_w)

r = PREV_BLACK / PREV_WHITE
weight = lambda p: 100 * r * (p / 100) / ((1 - p / 100) + r * (p / 100))
registry_white = REGISTRY_WHITE
assert REGISTRY_BLACK + REGISTRY_WHITE + REGISTRY_ASIAN + REGISTRY_OTHER == REGISTRY_RACE_REPORTED
benchmarks = {
    "US adults >=55 y": nat["55"], "US adults >=65 y": nat["65"], "US adults >=75 y": nat["75"],
    "Population >=65 y within 50 miles of US sites": catchment,
    "Specialty ATTR registry": share(REGISTRY_BLACK, registry_white),
    "Disease-weighted US adults >=65 y": weight(nat["65"]),
    "Diagnosed claims cohort": share(CLAIMS_BLACK_PCT, CLAIMS_WHITE_PCT),
    "Disease-weighted population within 50 miles": weight(catchment),
}
disease = list(benchmarks)[4:]
print("\nBenchmark (Black % | participation ratio, 95% CI)")
for k, v in benchmarks.items():
    print(f"  {k:<46} {v:6.2f}% | {pooled / v:.2f} ({lo / v:.2f}-{hi / v:.2f})")
show("Ratio range, general population", f"{min(pooled / benchmarks[k] for k in list(benchmarks)[:3]):.2f}-{max(pooled / benchmarks[k] for k in list(benchmarks)[:3]):.2f}", "0.70-0.91")
show("Ratio range, disease-related", f"{min(pooled / benchmarks[k] for k in disease):.2f}-{max(pooled / benchmarks[k] for k in disease):.2f}", "0.24-0.35")
show("Registry Black of Black or White patients", f"{REGISTRY_BLACK} of {REGISTRY_BLACK + registry_white}", "232 of 939")

# 4. Trial catchments
trials["catchment"] = trials.nct_id.map(catch)
trials["ratio"] = trials.pct / trials.catchment
show("Catchment Black proportion range, %", f"{trials.catchment.min():.1f}-{trials.catchment.max():.1f}", "14.4-16.5")
show("National Black proportion, adults >=65 y, %", f"{nat['65']:.1f}", "10.7")
show("Enrollment-to-catchment ratio range", f"{trials.ratio.min():.2f}-{trials.ratio.max():.2f}", "0.34-0.97")
print("Figure B (catchment % | enrollment % | ratio)")
for x in trials.sort_values("ratio").itertuples():
    print(f"  {x.trial:<46} {x.catchment:7.1f} | {x.pct:7.1f} | {x.ratio:5.2f}")
show("Figure B enrollment-to-catchment ratios", ", ".join(f"{x:.2f}" for x in trials.sort_values("ratio").ratio), "0.34, 0.48, 0.56, 0.63, 0.97")
show("Spread: catchment vs enrollment, percentage points",
     f"{trials.catchment.max() - trials.catchment.min():.1f} vs {trials.pct.max() - trials.pct.min():.1f}", "2.1 vs 9.9")

# 5. Ratio decomposition
for k, rep_ in [("Population >=65 y within 50 miles of US sites", "x1.42, x0.56"),
                ("Specialty ATTR registry", "x2.30, x0.35"), ("Diagnosed claims cohort", "x3.24, x0.25")]:
    show(f"Reference vs nation, enrollment vs reference: {k[:20]}", f"x{benchmarks[k] / nat['65']:.2f}, x{pooled / benchmarks[k]:.2f}", rep_)

# 6. Entry criteria
def ckd_epi_2021(scr, age_y, female):
    k, a = np.where(female, 0.7, 0.9), np.where(female, -0.241, -0.302)
    return 142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.200 * 0.9938 ** age_y * np.where(female, 1.012, 1.0)

nhanes["egfr"] = ckd_epi_2021(nhanes.creatinine_mg_dl, nhanes.age, nhanes.female == 1)
show("NHANES adults >=65 y, Black / White (n)", f"{(nhanes.race == 'Black').sum()} / {(nhanes.race == 'White').sum()}", "447 / 1,012")
for floor in (15, 25, 30):
    keep = {g: 1 - np.average(x.egfr < floor, weights=x.weight_mec) for g, x in nhanes.groupby("race")}
    closure = []
    for k in disease:
        p = benchmarks[k] / 100
        after = 100 * p * keep["Black"] / (p * keep["Black"] + (1 - p) * keep["White"])
        closure.append(100 * (benchmarks[k] - after) / (benchmarks[k] - pooled))
    show(f"eGFR <{floor}: disease-benchmark gap closed, %", f"{min(closure):.1f}-{max(closure):.1f}",
         {15: "1.0-1.4", 25: "2.2-3.0", 30: "4.7-6.3"}[floor])
    if floor == 30:
        show("eGFR <30: excluded Black / White, %", f"{100 * (1 - keep['Black']):.1f} / {100 * (1 - keep['White']):.1f}", "6.3 / 1.0")
keep30 = {g: 1 - np.average(x.egfr < 30, weights=x.weight_mec) for g, x in nhanes.groupby("race")}


def design_ci(d, y, domain):
    """Survey-weighted proportion, Taylor-linearized logit 95% CI."""
    w = d.weight_mec * domain
    p = (w * y).sum() / w.sum()
    z = pd.DataFrame({"h": d.strata, "i": d.psu, "z": w * (y - p) / w.sum()}).groupby(["h", "i"]).z.sum().reset_index()
    var = sum(len(g) / (len(g) - 1) * ((g.z - g.z.mean()) ** 2).sum() for _, g in z.groupby("h") if len(g) > 1)
    df = len(z) - z.h.nunique()
    half = stats.t.ppf(0.975, df) * np.sqrt(var) / (p * (1 - p))
    logit = np.log(p / (1 - p))
    return 100 * p, 100 / (1 + np.exp(-(logit - half))), 100 / (1 + np.exp(-(logit + half)))


ci = {g: design_ci(nhanes, (nhanes.egfr < 30).astype(float), (nhanes.race == g).astype(float)) for g in ("Black", "White")}
show("eGFR <30 excluded, %, design-based 95% CI, Black", "{:.1f} ({:.1f}-{:.1f})".format(*ci["Black"]), "6.3 (3.8-10.1)")
show("eGFR <30 excluded, %, design-based 95% CI, White", "{:.1f} ({:.1f}-{:.1f})".format(*ci["White"]), "1.0 (0.5-1.8)")
show("NHANES participants with eGFR <30, Black / White (n)",
     f"{((nhanes.race == 'Black') & (nhanes.egfr < 30)).sum()} / {((nhanes.race == 'White') & (nhanes.egfr < 30)).sum()}", "29 / 13")


def closure_at(ages):
    e = ckd_epi_2021(nhanes.creatinine_mg_dl, ages, nhanes.female == 1)
    kp = {g: 1 - np.average(e[nhanes.race == g] < 30, weights=nhanes.weight_mec[nhanes.race == g]) for g in ("Black", "White")}
    out = []
    for k in disease:
        p = benchmarks[k] / 100
        after = 100 * p * kp["Black"] / (p * kp["Black"] + (1 - p) * kp["White"])
        out.append(100 * (benchmarks[k] - after) / (benchmarks[k] - pooled))
    return min(out), max(out)


base_c = closure_at(nhanes.age)
for sub in (85, 90):
    lo_c, hi_c = closure_at(nhanes.age.where(nhanes.age < 80, sub))
    show(f"Age top-code 80 set to {sub}: gap closed at eGFR <30, %",
         f"{lo_c:.2f}-{hi_c:.2f} (max change {max(abs(lo_c - base_c[0]), abs(hi_c - base_c[1])):.2f})", "change <0.2",
         ok=max(abs(lo_c - base_c[0]), abs(hi_c - base_c[1])) < 0.2, rule="maximum change below 0.2 percentage points")
n_bw = b.sum() + w.sum()
short_d = []
print("\nFigure D: expected Black participants among", n_bw, "(at benchmark | after eGFR <30 exclusion | shortfall vs", b.sum(), "enrolled)")
for k in ["US adults >=65 y", "Population >=65 y within 50 miles of US sites", "Specialty ATTR registry",
          "Disease-weighted US adults >=65 y", "Diagnosed claims cohort", "Disease-weighted population within 50 miles"]:
    p_ = benchmarks[k] / 100
    after = p_ * keep30["Black"] / (p_ * keep30["Black"] + (1 - p_) * keep30["White"])
    print(f"  {k:<46} {n_bw * p_:7.0f} | {n_bw * after:7.0f} | {n_bw * after - b.sum():5.0f}")
    short_d.append(f"{n_bw * after - b.sum():.0f}")
show("Figure D remaining shortfalls", ", ".join(short_d), "55, 198, 499, 566, 822, 844")
show("Aged >=85 y among adults >=65 y, Black / White, %",
     f"{100 * county.black_85p.sum() / band_sum(county, 'black', age['65']):.1f} / {100 * county.white_85p.sum() / band_sum(county, 'white', age['65']):.1f}",
     "9.5 / 11.9")

# 7. Trial reporting
tables = trials[trials.ancestry_or_v142i_in_race_row != "no table"]
show("Trials citing African ancestry or p.V142I / with table", f"{(tables.ancestry_or_v142i_in_race_row == 'yes').sum()} / {len(tables)}", "3 / 4")
bm = audit[audit.item == "Black proportion compared with a disease-burden benchmark"]
show("Tables comparing Black enrollment with a disease-burden benchmark", f"{(bm.value == 'yes').sum()} of {len(bm)}", "0 of 4")
rep = trials[trials.claims_representative == 1]
show("Black % as reported by trials claiming representativeness",
     ", ".join(f"{100 * x.black / x.race_population:.1f}" for x in rep.itertuples()), "4.7, 8",
     ok=[round(100 * x.black / x.race_population, d) for x, d in zip(rep.itertuples(), (1, 0))] == [4.7, 8],
     rule="4.7 to one decimal, 8 to whole percent as the trials report")
show("Same trials, Black of Black or White, %", ", ".join(f"{x:.1f}" for x in rep.pct), "5.1, 8.0")
show("Efficacy subgroup dimensions (none by race)", trials.efficacy_subgroup_dimensions.sum(), "28")
dims = audit[audit.item == "efficacy subgroup dimension"].groupby("trial").size()
race_rows = audit[audit.item == "race efficacy subgroup reported"]
show("Trials reporting a race efficacy subgroup", f"{(race_rows.value == 'yes').sum()} of {race_rows.trial.nunique()}", "0 of 5")
show("Variant ATTR-CM share range, %",
     f"{(100 * trials.variant_attr / trials.variant_denominator).min():.1f}-{(100 * trials.variant_attr / trials.variant_denominator).max():.1f}", "9.7-24.0")
us = trials.dropna(subset=["us_enrolled"])
show("US enrollment, % (ATTR-ACT, APOLLO-B, ATTRibute-CM)", ", ".join(f"{100 * x.us_enrolled / x.us_denominator:.0f}" for x in us.itertuples()), "63, 27, 19")

# 8. Summary
out = pd.DataFrame(CHECKS)
out.to_csv("RESULTS-01-manuscript-value-check.csv", index=False)
n_fail = int((out.status == "FAIL").sum())
print(f"\n{len(out) - n_fail} of {len(out)} checks passed; written to RESULTS-01-manuscript-value-check.csv")
sys.exit(1 if n_fail else 0)
