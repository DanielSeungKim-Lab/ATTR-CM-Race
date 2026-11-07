Black representation in transthyretin amyloid cardiomyopathy trials
Data and code for the research letter. Tested with Python 3.11.15.

RUN
  pip install -r requirements.txt
  python CODE-01-reproduce-results.py      reproduces all reported values
  python CODE-00-build-inputs.py           optional; rebuilds DATA-02 to DATA-05 from
                                           public sources (needs CENSUS_API_KEY)

FILES
  DATA-00  918 screened registry records with coded fields
  DATA-01  Race, US enrollment and variant counts for the 5 pooled trials
  DATA-02  23 landmark trials
  DATA-03  US site coordinates of the pooled trials
  DATA-04  Black and White adults by county and age band
  DATA-05  NHANES participants aged 65 years or older with creatinine
  DATA-06  Subgroup and representativeness reporting audit

SOURCES
  ClinicalTrials.gov (retrieved 29 September 2026); ACS 2019-2023 5-year tables
  B01001A and B01001B; 2023 Census Gazetteer; NHANES 2017-March 2020 (P_DEMO,
  P_BIOPRO). Published constants in CODE-01: Mitchell et al., J Am Heart Assoc 2026;
  CARS, Stern et al., J Card Fail 2024, Table 3.
