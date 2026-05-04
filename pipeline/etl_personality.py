"""
ETL: Personality & Influencers page
Produces pre-aggregated CSVs — no PII in outputs.

Output files:
  personality_life_goals.csv        — life goal proportions by split
  personality_role_models.csv       — role model proportions by split
  personality_likeable_traits.csv   — likeable trait proportions by split
  personality_forming_beliefs.csv   — belief formation proportions by split
  personality_decision_confident.csv — trusted person proportions by split
  personality_wellbeing.csv         — happiness/satisfaction Likert by split
  personality_goals_achievable.csv  — achievable through FP yes/no by split

Matching strategy
-----------------
All multi- and single-select columns contain raw bilingual text from SurveyCTO,
typically "Hausa phrase / French phrase" or "Hausa phrase / English phrase".
We use case-insensitive substring matching (str.contains, case=False, regex=False)
against the English display label, which appears as a substring after the "/" separator.

Exception — yes/no questions: the raw values use French ("Oui"/"Non") with no
English equivalent, so we supply explicit `search_patterns` overrides that map the
English display label to the French substring that appears in the raw data.
"""

import pandas as pd
import numpy as np
import os
import warnings

from pipeline.config import (
    WEIGHT_COL, SPLIT_COLS, APP_DATA_DIR,
    LIFE_GOALS, ROLE_MODELS, LIKEABLE_TRAITS,
    FORMING_BELIEFS, DECISION_CONFIDENT,
)
from pipeline.utils import (
    weighted_counts, weighted_multiselect_counts,
    split_weighted_counts, split_weighted_multiselect,
    save,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning)

LIKERT = {
    5: "Always - all the time",
    4: "Often – most days",
    3: "Sometimes – some days",
    2: "Rarely",
    1: "Almost never",
    0: "Never – not at all",
}

# Raw bilingual values for yes/no responses have French, not English.
# "Yes" → "Ee / Oui", "No" → "A'a / Non", etc.
YESNO = {1: "Yes", 0: "No", -1: "Don't know", -2: "Prefer not to say"}
YESNO_SEARCH = {
    "Yes":              "Oui",
    "No":               "Non",
    "Don't know":       "Ne sais pas",
    "Prefer not to say":"fère ne pas",  # matches "Préfère ne pas le dire"
}


def _multiselect_long(df, col, label_map, tag, split_cols, search_patterns=None):
    """Return long-format DataFrame of proportions for a select_multiple column.

    label_map: {code: english_display_label}
      - The english_display_label is used as the search substring (case-insensitive)
        AND as the output label, unless overridden by search_patterns.
    search_patterns: optional {english_display_label: raw_substring_to_search}
      - Use when the raw data doesn't contain the English label as a substring
        (e.g. yes/no questions where raw data is in French).
    """
    sp = search_patterns or {}

    def compute_props(data):
        out = {}
        for code, english_label in label_map.items():
            pattern = sp.get(english_label, english_label)
            mask = data[col].str.contains(pattern, na=False, case=False, regex=False)
            out[english_label] = mask.mean()
        return pd.Series(out)

    rows = []

    # Overall
    s = compute_props(df)
    tmp = s.reset_index()
    tmp.columns = ["label", "proportion"]
    tmp["split"] = "none"
    tmp["group"] = "all"
    rows.append(tmp)

    # By splits
    for split_col in split_cols:
        frame = df.groupby(split_col).apply(compute_props)
        melted = (
            frame.reset_index()
            .melt(id_vars=split_col, var_name="label", value_name="proportion")
            .rename(columns={split_col: "group"})
        )
        melted["split"] = split_col
        rows.append(melted)

    result = pd.concat(rows, ignore_index=True)
    result["question"] = tag
    return result


def _single_long(df, col, label_map, tag, split_cols,
                 exclude=(-88, -99, -22), search_patterns=None):
    """Return long-format DataFrame of proportions for a single-select column.

    label_map: {code: english_display_label}
    search_patterns: optional {english_display_label: raw_substring_to_search}
      - Same override mechanism as _multiselect_long.
    exclude: raw values to drop before computing proportions (numeric codes).
    """
    sp = search_patterns or {}

    def compute_props(data):
        # Exclude coded missing values (only effective when col stores numeric codes)
        valid = ~data[col].isin(exclude)
        data = data.loc[valid]

        out = {}
        for code, english_label in label_map.items():
            pattern = sp.get(english_label, english_label)
            mask = data[col].astype(str).str.contains(
                pattern, na=False, case=False, regex=False
            )
            out[english_label] = mask.mean()
        return pd.Series(out)

    rows = []

    # Overall
    s = compute_props(df)
    tmp = s.reset_index()
    tmp.columns = ["label", "proportion"]
    tmp["split"] = "none"
    tmp["group"] = "all"
    rows.append(tmp)

    # By splits
    for split_col in split_cols:
        frame = df.groupby(split_col).apply(compute_props)
        melted = (
            frame.reset_index()
            .melt(id_vars=split_col, var_name="label", value_name="proportion")
            .rename(columns={split_col: "group"})
        )
        melted["split"] = split_col
        rows.append(melted)

    result = pd.concat(rows, ignore_index=True)
    result["question"] = tag
    return result


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [personality] running...")

    # ── Life goals ────────────────────────────────────────────────────────────
    goal_parts = []
    if "life_goals" in df.columns:
        g = _multiselect_long(df, "life_goals", LIFE_GOALS, "top3", SPLIT_COLS)
        goal_parts.append(g)
    if "life_goals_main" in df.columns:
        g = _single_long(df, "life_goals_main", LIFE_GOALS, "main", SPLIT_COLS)
        goal_parts.append(g)
    if goal_parts:
        save(pd.concat(goal_parts, ignore_index=True),
             os.path.join(APP_DATA_DIR, "personality_life_goals.csv"), "life goals")

    # Goals achievable through FP — yes/no, raw data in French
    if "life_goals_achievable" in df.columns:
        g = _single_long(
            df, "life_goals_achievable", YESNO, "achievable", SPLIT_COLS,
            exclude=(-88,), search_patterns=YESNO_SEARCH,
        )
        save(g, os.path.join(APP_DATA_DIR, "personality_goals_achievable.csv"),
             "goals achievable")

    # ── Role models ───────────────────────────────────────────────────────────
    if "role_models" in df.columns:
        g = _multiselect_long(df, "role_models", ROLE_MODELS, "role_models", SPLIT_COLS)
        save(g, os.path.join(APP_DATA_DIR, "personality_role_models.csv"), "role models")

    # ── Likeable traits ───────────────────────────────────────────────────────
    if "likeable_traits" in df.columns:
        g = _multiselect_long(df, "likeable_traits", LIKEABLE_TRAITS, "traits", SPLIT_COLS)
        save(g, os.path.join(APP_DATA_DIR, "personality_likeable_traits.csv"),
             "likeable traits")

    # ── Health belief formation ───────────────────────────────────────────────
    if "forming_beliefs" in df.columns:
        g = _multiselect_long(df, "forming_beliefs", FORMING_BELIEFS,
                              "forming_beliefs", SPLIT_COLS)
        save(g, os.path.join(APP_DATA_DIR, "personality_forming_beliefs.csv"),
             "belief formation")

    # ── Decision confident (trusted person) ───────────────────────────────────
    trust_parts = []
    if "decision_confident" in df.columns:
        g = _single_long(df, "decision_confident", DECISION_CONFIDENT,
                         "single", SPLIT_COLS)
        trust_parts.append(g)
    if "decision_confident_3" in df.columns:
        g = _multiselect_long(df, "decision_confident_3", DECISION_CONFIDENT,
                              "top3", SPLIT_COLS)
        trust_parts.append(g)
    if trust_parts:
        save(pd.concat(trust_parts, ignore_index=True),
             os.path.join(APP_DATA_DIR, "personality_decision_confident.csv"),
             "trusted person")

    # ── Wellbeing (Likert) ────────────────────────────────────────────────────
    wellbeing_rows = []
    for col, tag in [("happiness", "happiness"), ("satisfaction", "satisfaction")]:
        if col not in df.columns:
            print(f"  [personality] WARNING: '{col}' column not found, skipping.")
            continue

        # Overall
        s = weighted_counts(df, col, LIKERT, exclude=())
        tmp = s.reset_index()
        tmp.columns = ["label", "proportion"]
        tmp["question"] = tag
        tmp["split"] = "none"
        tmp["group"] = "all"
        wellbeing_rows.append(tmp)

        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, col, split_col, LIKERT, exclude=())
            melted = (
                frame.reset_index()
                .melt(id_vars=frame.index.name or "index",
                      var_name="group",
                      value_name="proportion")
                .rename(columns={frame.index.name or "index": "label"})
            )
            melted["question"] = tag
            melted["split"] = split_col
            wellbeing_rows.append(melted[["label", "split", "group",
                                          "proportion", "question"]])

    if wellbeing_rows:
        save(pd.concat(wellbeing_rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "personality_wellbeing.csv"), "wellbeing")

    print("  [personality] done.")
