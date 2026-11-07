"""Rebuild DATA-02 to DATA-05 from public sources into ./rebuilt/; stops unless each
equals the deposited file. Requires CENSUS_API_KEY. --requery prints current registry
counts for the original search."""
import io, json, os, sys, urllib.parse, urllib.request, zipfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rebuilt")
CTG = "https://clinicaltrials.gov/api/v2/studies"
QUERY_TERMS = ["transthyretin amyloidosis", "amyloid cardiomyopathy", "ATTR amyloidosis",
               "familial amyloid polyneuropathy", "cardiac amyloidosis", "amyloidosis"]
POOLED = ["NCT01994889", "NCT03860935", "NCT03997383", "NCT04136171", "NCT04153149"]
BANDS = {"55_64": (13, 28), "65_74": (14, 29), "75_84": (15, 30), "85p": (16, 31)}


def get(url, timeout=300):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def requery():
    ids = set()
    for term in QUERY_TERMS:
        token = None
        while True:
            q = {"query.cond": term, "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
                 "fields": "NCTId", "pageSize": 1000}
            if token:
                q["pageToken"] = token
            j = json.loads(get(CTG + "?" + urllib.parse.urlencode(q)))
            ids |= {s["protocolSection"]["identificationModule"]["nctId"] for s in j["studies"]}
            token = j.get("nextPageToken")
            if not token:
                break
        print(f"  {term}: cumulative {len(ids)}")


def landmark_trials():
    """Landmark rule applied to DATA-00. acronym, design, reports_race and analysis are
    copied from DATA-02 (hand-coded); only the pooled trials are checked."""
    s = pd.read_csv(os.path.join(HERE, "DATA-00-screened-trials.csv"))
    base = ((s.ttr_population == "yes") & (s.intent == "therapeutic")
            & s.phenotype.isin(["CM", "mixed"]) & (s.disease_modifying == "yes"))
    rand = s.allocation.eq("RANDOMIZED")
    nonrand = s.allocation.eq("NON_RANDOMIZED") | (s.allocation.isna() & s.arm_groups.eq(1))
    tier = pd.Series(None, index=s.index, dtype=object)
    tier[base & s.phase.isin(["PHASE3", "PHASE4"]) & nonrand & (s.enrollment >= 100)] = "B2"
    tier[base & s.phase.eq("PHASE2") & rand & (s.enrollment >= 50)] = "B1"
    tier[base & s.phase.isin(["PHASE3", "PHASE2/PHASE3"]) & rand & (s.enrollment >= 100)] = "A"
    s["tier"] = tier
    if not (s.tier.fillna("") == s.landmark_tier.fillna("")).all():
        raise SystemExit("landmark rule does not reproduce the coded tiers in DATA-00")
    lm = s[s.tier.notna()].copy()
    lm["ongoing"] = (~lm.status.isin(["COMPLETED", "TERMINATED", "WITHDRAWN"])).map({True: "yes", False: "no"})
    dep = pd.read_csv(os.path.join(HERE, "DATA-02-landmark-trials.csv"))
    cols = ["nct_id", "tier", "phase", "status", "completion_date", "results_posted", "ongoing"]
    new = lm[cols].merge(dep[["nct_id", "acronym", "design", "reports_race", "analysis"]], on="nct_id", how="left")
    pooled = set(pd.read_csv(os.path.join(HERE, "DATA-01-pooled-trials.csv")).nct_id)
    if set(new.nct_id[new.analysis == "pooled"]) != pooled:
        raise SystemExit("pooled trials in DATA-02 differ from DATA-01")
    return new[dep.columns.tolist()]


def sites():
    rows = []
    for nct in POOLED:
        j = json.loads(get(f"{CTG}/{nct}?fields=ContactsLocationsModule"))
        for loc in j["protocolSection"].get("contactsLocationsModule", {}).get("locations", []):
            gp = loc.get("geoPoint")
            if loc.get("country") == "United States" and gp:
                rows.append(dict(nct_id=nct, city=loc.get("city"), state=loc.get("state"),
                                 lat=gp["lat"], lon=gp["lon"]))
    return pd.DataFrame(rows).drop_duplicates().sort_values(["nct_id", "state", "city"])


def counties():
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise SystemExit("Set CENSUS_API_KEY (free key: https://api.census.gov/data/key_signup.html)")
    out = None
    for race, table in [("black", "B01001B"), ("white", "B01001A")]:
        cells = [f"{table}_{i:03d}E" for pair in BANDS.values() for i in pair]
        url = (f"https://api.census.gov/data/2023/acs/acs5?get=NAME,{','.join(cells)}"
               f"&for=county:*&in=state:*&key={key}")
        raw = get(url)
        try:
            j = json.loads(raw)
        except ValueError:
            raise SystemExit("Census API did not return JSON; check CENSUS_API_KEY: " + raw[:200].decode(errors="replace"))
        d = pd.DataFrame(j[1:], columns=j[0])
        d["fips"] = d.state + d.county
        for b, (m, f) in BANDS.items():
            d[f"{race}_{b}"] = d[f"{table}_{m:03d}E"].astype(int) + d[f"{table}_{f:03d}E"].astype(int)
        d = d[["fips", "NAME"] + [f"{race}_{b}" for b in BANDS]].rename(columns={"NAME": "name"})
        out = d if out is None else out.merge(d.drop(columns="name"), on="fips")
    z = zipfile.ZipFile(io.BytesIO(get("https://www2.census.gov/geo/docs/maps-data/data/"
                                       "gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip")))
    g = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    g = g.rename(columns={"GEOID": "fips", "INTPTLAT": "lat", "INTPTLONG": "lon"})
    out = g[["fips", "USPS", "lat", "lon"]].merge(out, on="fips")
    out = out[out.USPS != "PR"].drop(columns="USPS")            # 50 states and DC
    cols = ["fips", "name", "lat", "lon"] + [f"{r}_{b}" for r in ("black", "white") for b in BANDS]
    return out[cols].sort_values("fips")


def nhanes():
    base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    demo = pd.read_sas(io.BytesIO(get(base + "P_DEMO.xpt")), format="xport")
    bio = pd.read_sas(io.BytesIO(get(base + "P_BIOPRO.xpt")), format="xport")
    d = demo.merge(bio[["SEQN", "LBXSCR"]], on="SEQN")
    d = d[d.RIDRETH3.isin([3, 4]) & (d.RIDAGEYR >= 65) & d.LBXSCR.notna() & (d.WTMECPRP > 0)]
    return pd.DataFrame({
        "SEQN": d.SEQN.astype(int), "race": d.RIDRETH3.map({3: "White", 4: "Black"}),
        "age": d.RIDAGEYR.astype(int), "female": (d.RIAGENDR == 2).astype(int),
        "creatinine_mg_dl": d.LBXSCR, "weight_mec": d.WTMECPRP,
        "strata": d.SDMVSTRA.astype(int), "psu": d.SDMVPSU.astype(int)}).sort_values("SEQN")


def compare(name, new, keys):
    new.to_csv(os.path.join(OUT, name), index=False)
    old = pd.read_csv(os.path.join(HERE, name), dtype={"fips": str})
    new = pd.read_csv(os.path.join(OUT, name), dtype={"fips": str})
    problems = []
    if list(old.columns) != list(new.columns):
        problems.append(f"columns differ: {list(old.columns)} vs {list(new.columns)}")
    for d, label in ((old, "deposited"), (new, "rebuilt")):
        if d.duplicated(keys).any():
            problems.append(f"duplicate keys in {label} file")
    if not problems:
        a = old.sort_values(keys).reset_index(drop=True)
        b = new.sort_values(keys).reset_index(drop=True)
        if len(a) != len(b) or not (a[keys].astype(str).values == b[keys].astype(str).values).all():
            problems.append("key sets differ")
        else:
            for c in a.columns:
                num = [pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c]) for x in (a, b)]
                if all(num):
                    same = ((a[c] - b[c]).abs() <= 1e-6) | (a[c].isna() & b[c].isna())
                else:
                    same = (a[c].astype(str) == b[c].astype(str)) | (a[c].isna() & b[c].isna())
                if not same.all():
                    problems.append(f"{c}: {int((~same).sum())} differing rows")
    if problems:
        raise SystemExit(f"{name} does not match: " + "; ".join(problems))
    print(f"  {name}: {len(new)} rows, identical to deposited file")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--requery" in sys.argv:
        print("Current registry search")
        requery()
    compare("DATA-02-landmark-trials.csv", landmark_trials(), ["nct_id"])
    compare("DATA-03-us-sites.csv", sites(), ["nct_id", "city", "state"])
    compare("DATA-04-county-population.csv", counties(), ["fips"])
    compare("DATA-05-nhanes-creatinine.csv", nhanes(), ["SEQN"])
