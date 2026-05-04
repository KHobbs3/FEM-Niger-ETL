"""
ETL: Respondent Profile page
Produces pre-aggregated CSVs — no PII in outputs.

Output files:
  respondents_profile.csv  — weighted counts and proportions for key demographics
"""

import pandas as pd
import os

from pipeline.config import APP_DATA_DIR
from pipeline.utils import save


# Columns to profile and their display labels
DEMO_COLS = {
    "use":             "FP use group",
    "gender":          "Gender",
    "age_group":       "Age group",
    "occupation":      "Occupation",
    "religion":        "Religion",
    "urban_rural": "Settlement type",
}

USE_LABELS = {
    "user":        "Current user",
    "past_user":   "Past user",
    "future_user": "Future user",
    "nonuser":     "Non-user",
}

GENDER_LABELS = {
    "Mace / Femme":   "Female",
    "Namiji / Homme": "Male",
}


def _clean_bilingual(series):
    """'Hausa / English' → 'English'  (no-op if no slash)."""
    return series.astype(str).str.split(" / ").str[-1].str.strip()


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [respondents] running...")

    rows = []

    # ── Total sample ──────────────────────────────────────────────────────────
    rows.append({
        "variable":   "_total",
        "category":   "Total",
        "count":      len(df),
        "proportion": 1.0,
    })

    # ── Per-variable breakdowns ───────────────────────────────────────────────
    for col, label in DEMO_COLS.items():
        if col not in df.columns:
            print(f"  [respondents] '{col}' not found, skipping")
            continue

        valid = df[[col]].dropna().copy()
        n_total = len(valid)
        if n_total == 0:
            print(f"  [respondents] no valid data for '{col}', skipping")
            continue

        # Clean category labels
        if col == "use":
            valid[col] = valid[col].map(USE_LABELS).fillna(valid[col])
        elif col == "gender":
            valid[col] = valid[col].map(GENDER_LABELS).fillna(valid[col])
        elif col in ("occupation", "religion"):
            valid[col] = _clean_bilingual(valid[col])

        for cat, grp in valid.groupby(col):
            n = len(grp)
            rows.append({
                "variable":         col,
                "category":         str(cat),
                "count":            n,
                "proportion":       n / n_total,
            })

    result = pd.DataFrame(rows)
    save(result, os.path.join(APP_DATA_DIR, "respondents_profile.csv"),
         "respondent profile")
    print("  [respondents] done.")
