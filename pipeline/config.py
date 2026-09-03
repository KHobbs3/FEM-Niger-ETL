"""
Shared configuration for the Benin FEM survey analysis pipeline.

The pipeline reads from the PII dataset and writes pre-aggregated CSVs
into the app's data/ folder. The app never touches the raw data.

Directory layout expected:
    1_formative_research/            ← project root (paths below are anchored here)
        table_analysis/
            data/2_cleaned/          ← reads the weighted survey CSV from here
        pipeline_output/
            pipeline/                ← this folder
        benin_app/
            data/                    ← pre-aggregated CSVs written here
            src/
            app.py
"""

import os
from pathlib import Path

# Country name, used to build output filenames below (etl_radio.py etc.) --
# update this (and the paths below) when running this pipeline for a new country.
COUNTRY = "Benin"

# Anchor default paths to this file's own location rather than the process's
# current working directory -- otherwise these relative defaults only resolve
# correctly when run from one specific directory, and silently break with
# "File not found" if run from anywhere else (e.g. from inside pipeline/).
# On disk: 1_formative_research/{table_analysis, pipeline_output/pipeline/config.py, benin_app}
# -- i.e. table_analysis and benin_app are siblings of pipeline_output, not of
# pipeline/ itself, so this is 3 levels up from config.py, not 2.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Paths ─────────────────────────────────────────────────────────────────────
station_path = "/Users/kt/Documents/work/AIM Charities/FEM/Family Planning/2. Everything Radio/Radio Reach/cloudrf/output/gpkg/benin/fem_benin_buildings/"

# Raw mapped dataset (PII — never committed to app repo)
# NOTE: must match the filename table_analysis/src/config.py's DIR_WEIGHTED_DATA
# actually writes -- currently "02_fem_survey_benin_weighted.csv" (the "02_"
# prefix was added when the export path was moved off of DIR_CLEANED_DATA).
DIR_MAPPED_DATA = os.environ.get(
    "FEM_MAPPED_DATA",
    str(PROJECT_ROOT / "table_analysis" / "data" / "2_cleaned" / "02_fem_survey_benin_weighted.csv"),
)

# Where the app reads its pre-aggregated data from
APP_DATA_DIR = os.environ.get(
    "FEM_APP_DATA_DIR",
    str(PROJECT_ROOT / "benin_app" / "data"),
)

# Statement label mapping (non-PII, already in app repo)
DIR_STATEMENT_LABELS = os.path.join(APP_DATA_DIR, "statement_labels.csv")

# ── Survey constants ──────────────────────────────────────────────────────────

WEIGHT_COL = "combined_weight_adjusted"

USER_GROUPS    = ["user", "past_user"]
NONUSER_GROUPS = ["nonuser", "future_user"]
ALL_USE_GROUPS = USER_GROUPS + NONUSER_GROUPS

SPLIT_COLS = ["use", "gender", "age_group"]

# 2026-09-02: verified against methodology sources (no respondent data
# touched) that this covers every possible value, not just every value seen
# so far. The raw XLSForm field is a plain integer (minutes) -- there's no
# fixed choice list to enumerate -- but table_analysis/00_clean_data.ipynb
# bins it before this pipeline ever sees it:
#   pd.cut(data['willingness_to_travel'],
#          bins=[0, 5, 15, 30, 45, 60, 90, 120, 180, 240, 1500],
#          labels=['0-5 minutes', '5-15 minutes', ..., '4+ hours'],
#          right=True)
# Those 10 labels are exactly WTT_MAP's 10 keys below, verbatim and in the
# same order -- pd.cut can't produce a category outside this set, so there's
# no missing key to add here.
#
# KNOWN GAP -- not fixable in this file: pd.cut's default right=True makes
# each bin (lower, upper], so a raw answer of exactly 0 minutes falls
# outside every bin (0 is not > 0) and becomes NaN during cleaning, before
# WTT_MAP is ever consulted -- silently dropping that respondent's travel
# time rather than mapping it to "0-5 minutes". validate_pipeline.py's
# unmapped-value check can't catch this either, since it only compares
# already-binned non-null values against WTT_MAP -- a value that became NaN
# upstream never reaches that comparison. If 0-minute responses exist in the
# data, the fix is in 00_clean_data.ipynb (e.g. bins=[-1, 5, 15, ...], or
# right=False), not here.
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

# ── Closed-category label dictionaries ──────────────────────────────────────
# Canonical language is French, matching Benin's raw survey data (French/Fon
# bilingual labels; the French half is what _strip_hausa/_clean_bilingual etc.
# keep). Codes and French text pulled directly from the Benin choices sheet
# (table_analysis/data/meta/... via DIR_META_DATA), verified against the
# 'value'/'label' columns for each list_name -- not carried over from Niger's
# dicts, since the codes don't line up between the two countries' forms.
#
# Each *_EN counterpart is an English translation for a planned "English"
# toggle in benin_app. These were translated by Claude (AI-generated) from
# the French choices-sheet labels, not sourced from an official/certified
# translation -- show a visible disclaimer wherever EN is selected in the app.

CONTRACEPTIVE_METHODS = {
    -99: "Préfère ne pas répondre", -88: "Je ne sais pas", -22: "Autre",
    0:  "Aucune des réponses ci-dessus",
    1:  "Stérilisation",                                    2:  "Implants",
    3:  "Pilules contraceptives orales",                    4:  "Dispositifs intra-utérins DIU (aussi appelés « stérilets »)",
    5:  "Injectables",                                      6:  "Préservatifs",
    7:  "Anneau vaginal",                                   8:  "Patch contraceptif",
    9:  "Méthodes de barrière vaginale (diaphragme, cape cervicale, spermicides)", 10: "Retrait",
    11: "Abstinence",                                       12: "Méthode du calendrier (« méthode Ogino »)",
    13: "Méthode des jours fixes (collier de perles)",      14: "Méthode de l'allaitement (aménorrhée de lactation)",
    15: "Pilules contraceptives d'urgence",
}
CONTRACEPTIVE_METHODS_EN = {
    -99: "Prefer not to say", -88: "Don't know", -22: "Other",
    0:  "None of the above",
    1:  "Sterilisation",                                    2:  "Implants",
    3:  "Oral contraceptive pills",                         4:  "Intrauterine device (IUD / \"coil\")",
    5:  "Injectables",                                      6:  "Condoms",
    7:  "Vaginal ring",                                     8:  "Contraceptive patch",
    9:  "Vaginal barrier methods (diaphragm, cervical cap, spermicides)", 10: "Withdrawal",
    11: "Abstinence",                                       12: "Calendar method (\"Ogino method\")",
    13: "Standard Days Method (bead necklace)",             14: "Lactational amenorrhea method (breastfeeding)",
    15: "Emergency contraceptive pills",
}

LIFE_GOALS = {
    -99: "Préfère ne pas répondre", -88: "Je ne sais pas", -22: "Autre",
    1:  "Avoir toujours assez de nourriture pour mes enfants",  2:  "Des enfants bien élevés",
    3:  "Des enfants pouvant aller à l'école",                  4:  "Une meilleure éducation pour moi-même",
    5:  "Une bonne santé pour les membres de ma famille",       6:  "Une bonne/meilleure santé personnelle",
    7:  "Une longue vie",                                       8:  "Avoir bonne mine et être en forme",
    9:  "Davantage de temps pour me reposer ou moins de stress",10: "Un(e) conjoint(e) aimant(e), responsable et attentionné(e)",
    11: "Avoir davantage d'enfants",                            12: "Avoir moins d'enfants",
    13: "Désir d'avoir un enfant d'un genre spécifique (garçon ou fille)", 14: "Une famille paisible",
    15: "Trouver un emploi",                                    16: "Avoir un revenu stable",
    17: "Avoir un travail meilleur / agréable / respecté",      18: "Créer une entreprise",
    19: "Prospérité financière",                                20: "Construire / posséder une maison",
    21: "Posséder une voiture ou une moto",                     22: "Avoir plus de temps pour la pratique religieuse",
    23: "Visiter La Mecque ou d'autres lieux de pèlerinage religieux", 24: "Être une personne responsable / sage / patiente",
    25: "Être respecté(e) de ses pairs",                        26: "Aider les autres",
    27: "Vivre à l'étranger",                                   28: "Progrès pour le pays (meilleures routes, eau ou économie)",
    29: "Correction des erreurs passées",                       30: "Épouser une autre femme (en plus de celle(s) qu'il a déjà)",
}
LIFE_GOALS_EN = {
    -99: "Prefer not to say", -88: "Don't know", -22: "Other",
    1:  "Always having enough food for my children",           2:  "Well-behaved children",
    3:  "Children able to go to school",                       4:  "Better education for myself",
    5:  "Good health for my family members",                   6:  "Good/better personal health",
    7:  "A long life",                                         8:  "Looking good and being in good shape",
    9:  "More time to rest or less stress",                    10: "A loving, responsible, and caring spouse",
    11: "Having more children",                                12: "Having fewer children",
    13: "Wanting a child of a specific gender (boy or girl)",  14: "A peaceful family",
    15: "Finding a job",                                       16: "Having a stable income",
    17: "Having a better / enjoyable / respected job",         18: "Starting a business",
    19: "Financial prosperity",                                20: "Building / owning a house",
    21: "Owning a car or motorcycle",                          22: "Having more time for religious practice",
    23: "Visiting Mecca or other religious pilgrimage sites",  24: "Being a responsible / wise / patient person",
    25: "Being respected by peers",                            26: "Helping others",
    27: "Living abroad",                                       28: "Progress for the country (better roads, water, or economy)",
    29: "Correcting past mistakes",                             30: "Marrying another wife (in addition to the one(s) he already has)",
}

# Sourced from the "role_models_list" choices list (37 entries) -- NOT the
# choices list literally named "role_models" (12 entries), which is actually
# used for the redesigned `forming_beliefs` question -- see FORMING_BELIEFS.
ROLE_MODELS = {
    -99: "Je ne sais pas", -88: "Préfère ne pas répondre", -22: "Autre",
    1:  "Le prophète Mahomet",              2:  "Dieu",
    3:  "Le chef religieux",                4:  "Malama (prédicatrices)",
    5:  "Chef traditionnel",                6:  "Jésus-Christ",
    7:  "Croyant fervent",                  8:  "Conjoint",
    9:  "Mère",                             10: "Père",
    11: "Frère",                            12: "Sœur",
    13: "Ami",                              14: "Enfants",
    15: "Dirigeant politique",              16: "Voisin",
    17: "Collègue",                         18: "Les anciens, les personnes âgées",
    19: "Officiers militaires",             20: "Hommes d'affaires",
    21: "Commerçants",                      22: "Enseignants",
    23: "Professionnel de santé",           24: "Superstars",
    25: "Les personnes qui ont réussi",     26: "Mon patron",
    27: "Les gens riches",                  28: "Oncle",
    29: "Tante",                            30: "Grand-mère",
    31: "Grand-père",                       32: "Un membre de la famille",
    33: "Personnages de radio ou de télévision", 35: "Aucune personne",
}
ROLE_MODELS_EN = {
    -99: "Don't know", -88: "Prefer not to say", -22: "Other",
    1:  "The Prophet Muhammad",             2:  "God",
    3:  "Religious leader",                 4:  "Malama (female preachers)",
    5:  "Traditional leader",               6:  "Jesus Christ",
    7:  "Devout believer",                  8:  "Spouse",
    9:  "Mother",                           10: "Father",
    11: "Brother",                          12: "Sister",
    13: "Friend",                           14: "Children",
    15: "Political leader",                 16: "Neighbour",
    17: "Colleague",                        18: "Elders, older people",
    19: "Military officers",                20: "Businessmen",
    21: "Traders / shopkeepers",            22: "Teachers",
    23: "Health professional",              24: "Superstars",
    25: "Successful people",                26: "My boss",
    27: "Rich people",                      28: "Uncle",
    29: "Aunt",                             30: "Grandmother",
    31: "Grandfather",                      32: "A family member",
    33: "Radio or TV personalities",        35: "No one",
}

LIKEABLE_TRAITS = {
    -99: "Préfère ne pas répondre", -88: "Je ne sais pas", -22: "Autre",
    1:  "Calme",                              2:  "Ne se mêle pas des affaires des autres",
    3:  "Sociable",                           4:  "Gentillesse",
    5:  "N'aime pas les choses matérielles",  6:  "Ne fait pas de bruit",
    7:  "Patience",                           8:  "Endurance",
    9:  "Honnêteté",                          10: "Respect",
    11: "Éducation",                          12: "Sincère",
    13: "Pardonne facilement",                14: "Discipline",
    15: "Bon croyant",                        16: "Bon comportement",
    17: "Issu d'une bonne lignée et d'un bon statut social", 18: "Bon cœur",
    19: "Bonne personnalité",                 20: "Aidant les autres",
    21: "Respectable",                        22: "Fiabilité",
    23: "Honnêteté",                          24: "À l'écoute",
    25: "Religieux",                          26: "Soigné",
    27: "Se soucie de ses proches",           28: "Encourage les bons comportements",
    29: "Courageux",                          30: "Accueille tout le monde",
    31: "A des enfants bien éduqués",         32: "Riche",
    33: "Ne se plaint jamais",                34: "Prend bien soin de sa famille",
}
LIKEABLE_TRAITS_EN = {
    -99: "Prefer not to say", -88: "Don't know", -22: "Other",
    1:  "Calm",                               2:  "Doesn't meddle in others' business",
    3:  "Sociable",                           4:  "Kindness",
    5:  "Not materialistic",                  6:  "Doesn't make noise",
    7:  "Patient",                            8:  "Endurance",
    9:  "Honest",                             10: "Respectful",
    11: "Educated / well brought-up",         12: "Sincere",
    13: "Forgives easily",                    14: "Disciplined",
    15: "Good believer",                      16: "Good behaviour",
    17: "From a good lineage and good social standing", 18: "Kind-hearted",
    19: "Good personality",                   20: "Helps others",
    21: "Respectable",                        22: "Reliable",
    23: "Honest",                             24: "A good listener",
    25: "Religious",                          26: "Neat / well-kept",
    27: "Cares about loved ones",             28: "Encourages good behaviour",
    29: "Brave",                              30: "Welcomes everyone",
    31: "Has well brought-up children",       32: "Rich",
    33: "Never complains",                    34: "Takes good care of their family",
}

# "Forming beliefs" was redesigned for Benin's survey; the raw XLSForm points
# this question's choice list at the list literally named "role_models" (a
# confusing but confirmed-intentional naming leftover, not a data error) --
# 12 entries about how the respondent forms health decisions/beliefs.
FORMING_BELIEFS = {
    -88: "Je préfère ne pas répondre", -22: "Autre",
    1: "J'écoute mon corps et mon cœur",
    2: "Je réfléchis moi-même au sujet et j'essaie de le comprendre seul",
    3: "Je me réfère à ce qui est recommandé dans les textes religieux",
    4: "Je demande l'avis des personnes qui savent le mieux comment prendre cette décision",
    5: "Je réfléchis à des expériences similaires que j'ai vécues dans le passé",
    6: "Je regarde ce qui est commun chez les personnes qui m'entourent",
    7: "J'essaie différentes options et je choisis celle qui me convient le mieux",
    8: "Je regarde ce qui a été découvert dans le cadre de recherches",
    9: "Je cherche des informations dans les médias, sur Internet ou sur les réseaux sociaux",
    10: "Je ne sais pas",
}
FORMING_BELIEFS_EN = {
    -88: "Prefer not to say", -22: "Other",
    1: "I listen to my body and heart",
    2: "I think it through myself and try to understand it alone",
    3: "I refer to what is recommended in religious texts",
    4: "I ask the opinion of people who best know how to make this decision",
    5: "I think about similar experiences I've had in the past",
    6: "I look at what's common among the people around me",
    7: "I try different options and choose the one that suits me best",
    8: "I look at what research has found",
    9: "I look for information in the media, on the internet, or on social media",
    10: "Don't know",
}

DECISION_CONFIDENT = {
    -99: "Préfère ne pas répondre", -88: "Je ne sais pas", -22: "Autre",
    1:  "Mon mari",                2:  "Ma femme",
    3:  "Amis",                    4:  "Sœurs",
    5:  "Beaux-parents",           6:  "Mère",
    7:  "Père",                    8:  "Frères",
    9:  "Autorités gouvernementales", 10: "Membre de ma famille",
    11: "Voisins",                 12: "Personnel de santé",
    13: "Chefs religieux",         14: "Dirigeants",
    15: "Moi-même",                16: "Mes enfants",
    17: "La divinité (Dieu, Jésus, le Prophète…)", 18: "Radio",
    19: "Médias",
}
DECISION_CONFIDENT_EN = {
    -99: "Prefer not to say", -88: "Don't know", -22: "Other",
    1:  "My husband",              2:  "My wife",
    3:  "Friends",                 4:  "Sisters",
    5:  "In-laws",                 6:  "Mother",
    7:  "Father",                  8:  "Brothers",
    9:  "Government authorities",  10: "A family member",
    11: "Neighbours",              12: "Health workers",
    13: "Religious leaders",       14: "Leaders",
    15: "Myself",                  16: "My children",
    17: "The divine (God, Jesus, the Prophet...)", 18: "Radio",
    19: "Media",
}

# 2026-09-02: added top_driver/top_barrier (each respondent's single main
# reason for/against contraceptive use) and region, per the user's request.
# region is deliberately NOT included here -- it's used as a clustering
# *split* (see PROVINCE_TO_REGION/REGIONS below and etl_personas.py), so
# baking it in as a pooled feature would make "do clusters concentrate
# geographically?" true by construction rather than a real finding. It IS
# added as a feature for the standalone culture-clustering analysis's
# validation step -- see run_culture_clusters() in etl_personas.py -- just
# not as an input to that analysis's own clustering (same reasoning).
VARS_FOR_CLUSTERING = ["age", "gender", "occupation", "religion", "life_goals",
                       "top_driver", "top_barrier"]

# Clustering features for the standalone "do cultural traits geographically
# cluster?" analysis (run_culture_clusters() in etl_personas.py) -- religion,
# life goals, and main driver/barrier only. region is checked against the
# resulting clusters afterward, not used to form them.
CULTURE_CLUSTERING_VARS = ["religion", "life_goals", "top_driver", "top_barrier"]

# ── Region grouping ───────────────────────────────────────────────────────────
# Not a raw survey field -- derived from `province` per the mapping supplied by
# the user (2026-08-24). Mirrors the same grouping used in
# table_analysis/03_driver_barrier_table_w_counts.ipynb's REGION split.
PROVINCE_TO_REGION = {
    "Borgou": "North-East",
    "Donga": "North-West",
    "Littoral": "South-South", "Oueme": "South-South", "Ouémé": "South-South", "Atlantique": "South-South",
    "Couffo": "Mid-South", "Zou": "Mid-South", "Plateau": "Mid-South",
}
REGIONS = ["North-East", "North-West", "South-South", "Mid-South"]

# 2026-09-02: coarser 2-way grouping, requested on top of the 4-way one above
# (both are available -- this doesn't replace REGIONS). Confirmed with the
# user: "Centre-South" is the same thing as "Mid-South" above (an informal
# name for it found elsewhere in this codebase, e.g. the radio page's
# STATION_STATE descriptions), not a distinct third group -- so SOUTH here
# is just South-South + Mid-South, not 3 components.
REGION_TO_NORTH_SOUTH = {
    "North-East": "North", "North-West": "North",
    "South-South": "South", "Mid-South": "South",
}
NORTH_SOUTH_REGIONS = ["North", "South"]
