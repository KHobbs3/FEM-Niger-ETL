"""
Shared configuration for the Niger FEM survey analysis pipeline.

The pipeline reads from the PII dataset and writes pre-aggregated CSVs
into the app's data/ folder. The app never touches the raw data.

Directory layout expected:
    <project_root>/
        pipeline/           ← this folder (run from here or project root)
        niger_app/
            data/           ← pre-aggregated CSVs written here
            src/
            app.py
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
station_path = "/Users/kt/Documents/work/AIM Charities/FEM/Family Planning/2. Everything Radio/Radio Reach/cloudrf/output/gpkg/niger/habib_niger_buildings/"

# Raw mapped dataset (PII — never committed to app repo)
DIR_MAPPED_DATA = os.environ.get(
    "FEM_MAPPED_DATA",
    "../table_analysis/data/2_cleaned/fem_survey_niger_mapped.csv",
)

# Where the app reads its pre-aggregated data from
APP_DATA_DIR = os.environ.get(
    "FEM_APP_DATA_DIR",
    "../niger_app/data",
)

# Statement label mapping (non-PII, already in app repo)
DIR_STATEMENT_LABELS = os.path.join(APP_DATA_DIR, "statement_labels.csv")

# ── Survey constants ──────────────────────────────────────────────────────────

WEIGHT_COL = "combined_weight_adjusted"

USER_GROUPS    = ["user", "past_user"]
NONUSER_GROUPS = ["nonuser", "future_user"]
ALL_USE_GROUPS = USER_GROUPS + NONUSER_GROUPS

SPLIT_COLS = ["use", "gender", "age_group"]

WTT_MAP = {
    "0-5 minutes":    2.5,
    "5-15 minutes":   10,
    "15-30 minutes":  22.5,
    "30-45 minutes":  37.5,
    "45-60 minutes":  52.5,
    "60-90 minutes":  75,
    "90-120 minutes": 105,
    "2-3 hours":      150,
    "3-4 hours":      210,
    "4+ hours":       270,
}

# Contraceptive method integer -> English label (from SurveyCTO choices sheet)
CONTRACEPTIVE_METHODS = {
    1:  "Sterilisation",      2:  "Implants",
    3:  "Oral contraceptive pills", 4: "IUD",
    5:  "Injectables",        6:  "Condoms",
    7:  "Vaginal ring",       8:  "Contraceptive patch",
    9:  "Vaginal barrier",    10: "Withdrawal",
    11: "Abstinence",         12: "Calendar / Rhythm",
    13: "Standard Days",      14: "Lactational Amenorrhea",
    15: "Emergency contraception",
    0:  "None", -22: "Other",
}

LIFE_GOALS = {
    1:"Enough food for children", 2:"Well-behaved children",
    3:"Children go to school",    4:"Better education for myself",
    5:"Good health for family",   6:"Better personal health",
    7:"Living a long life",       8:"Looking good and fresh",
    9:"More rest / less stress",  10:"Good / supportive spouse",
    11:"Having more children",    12:"Having fewer children",
    13:"Specific gender child",   14:"Peaceful family",
    15:"Getting a job",           16:"Stable income",
    17:"Better / respected work", 18:"Starting a business",
    19:"Financial prosperity",    20:"Building / owning a house",
    21:"Owning car or motorcycle",22:"More time for religion",
    23:"Visiting Mecca",          24:"Be responsible / wise",
    25:"Respect from peers",      26:"Helping others",
    27:"Living abroad",           28:"Progress for the country",
    29:"Reversing past mistakes", -22:"Other",
}

ROLE_MODELS = {
    1:"Prophet",               3:"Religious leader",
    4:"Women preachers",       5:"Traditional leader",
    7:"Fervent believer",      8:"Spouse",
    9:"Mother",               10:"Father",
    11:"Brother",             12:"Sister",
    13:"Friend",              14:"Children",
    15:"Political leader",    16:"Neighbour",
    17:"Colleague",           18:"The elders",
    19:"Military officers",   20:"Businessman",
    21:"Shopkeeper",          22:"School teacher",
    23:"Health worker",       24:"Superstar",
    25:"Successful people",   26:"My boss",
    27:"Rich people",         28:"Uncle",
    29:"Aunt",                30:"Grandmother",
    31:"Grandfather",         32:"A family relative",
    33:"Radio or TV character",34:"Nobody",
    -22:"Other",
}

LIKEABLE_TRAITS = {
    1:"Calm",               2:"Minds own business",
    3:"Sociable",           4:"Kind",
    5:"Not materialistic",  6:"No noise making",
    7:"Patient",            8:"Endurance",
    9:"Honest",            10:"Respectful",
    11:"Educated",         13:"Forgives easily",
    14:"Disciplined",      15:"Good believer",
    16:"Good behaviours",  17:"Good lineage",
    18:"Kind-hearted",     19:"Good personality",
    20:"Helps people",     21:"Respected",
    22:"Reliable",         23:"Not a cheat",
    24:"Listens to people",25:"Religious",
    26:"Neat",             27:"Cares about relatives",
    28:"Encourages good behaviour", 29:"Brave",
    30:"Embraces everyone",31:"Has educated children",
    32:"Is rich",          33:"Never complains",
    34:"Takes care of family", -22:"Other",
}

FORMING_BELIEFS = {
    1:"Listen to my body and heart",
    2:"Think about it alone",
    3:"Look to religious texts",
    4:"Seek knowledgeable opinions",
    5:"Think about past experiences",
    6:"See what is common around me",
    7:"Try different options",
    8:"Look at research",
    -22:"Other",
}

DECISION_CONFIDENT = {
    1:"Husband",           2:"Friends",
    3:"Radio",             4:"Sisters",
    5:"In-laws",           6:"Mother",
    7:"Father",            8:"Wife / Wives",
    9:"Brothers",         10:"Governmental authority",
    11:"Family relatives",12:"Neighbours",
    13:"Health workers",  14:"Religious leaders (alive)",
    15:"Leaders",         16:"Myself",
    17:"My children",     18:"Religious leaders (texts)",
    19:"Media",           20:"Co-wives",
    -22:"Other",
}

VARS_FOR_CLUSTERING = ["age", "gender", "occupation", "religion", "life_goals"]
