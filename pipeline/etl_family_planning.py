"""
ETL: Family Planning page
Produces pre-aggregated CSVs — no PII in outputs.

Output files:
  fp_funnel.csv             — aware/ever/current rates by split group
  fp_timing.csv             — timing of next pregnancy distribution
  fp_methods.csv            — method proportions (known/ever/current) by split
  fp_reason_use.csv         — reason for use by split
  fp_intent.csv             — future intent + considered use by split
  fp_nonuse_reasons.csv     — free-text non-use reason counts (no names/IDs)
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import (
    WEIGHT_COL, SPLIT_COLS, APP_DATA_DIR,
    CONTRACEPTIVE_METHODS,
)
from pipeline.utils import (
    weighted_prop, weighted_counts, weighted_multiselect_counts,
    weighted_multiselect_counts_text,
    split_weighted_counts, split_weighted_multiselect,
    split_weighted_multiselect_text,
    safe_melt,
    save,
)

TIME_TO_PREGNANT = {
    1:"Within 6 months", 2:"6-12 months", 3:"1-2 years",
    4:"More than 2 years", 5:"Already pregnant", 6:"No more children",
}
REASON_USE = {1:"Space births", 2:"No more children", -22:"Other"}
YESNO = {1:"Oui", 0:"Non"}

HAUSA_ENG = {
    "Il n'est pas intéressé": "He's not interested",
    "Inason haifuwa yara": "I want to have children",
    "Rien": "Nothing",
    "Ban sansu ba": "I don't know them",
    "Saboda mijina baya guida": "Because my husband doesn't guide",
    "Babu takameman dalili": "No specific reason",
    "Inason haifuwa": "I want to have children",
    "Rashin ganin jinin haila yayin shayarwa": "Not getting my period while breastfeeding",
    "Ina ayiki da allurai yanzuma": "I'm on birth control right now",
    "Zanyi se nan gaba": "I'll do it later",
    "Sabida ina da ciki": "Because I am pregnant",
    "Sabida mijina baya guida": "Because my husband is infertile",
    "Saboda yarana sunada issassar tazara": "Because my children are far apart in age",
    "Bukatar samun yara": "Want to have more children",
    "Haifuwa nikeso": "I want to give birth",
    "Manque de connaissance": "Lack of knowledge",
    "Jikina Yana bani tazara": "My body spacing",
    "Bukatar karin samun yara": "Want to have more children",
    "Manque de connaissance sur le planning familial": "Lack of knowledge about family planning",
    "Tsoron matsalar zubar jini": "Fear of complications from bleeding"
}

def _strip_hausa(text: str) -> str:
    """Return only the English part of bilingual 'Hausa / English' labels.
    
    Handles both single labels and pipe-delimited multiple labels.
    Preserves NaN values.
    """
    # Preserve NaN/None
    if pd.isna(text):
        return np.nan
    
    text = str(text).strip()
    
    # Handle pipe-delimited multiple options
    if "|" in text:
        parts = text.split("|")
        english_parts = [_strip_hausa(part) for part in parts]
        # Filter out NaN results
        english_parts = [p for p in english_parts if pd.notna(p)]
        return "|".join(english_parts) if english_parts else np.nan
    
    # Handle single bilingual label (Hausa / English)
    if "/" in text:
        split = text.split("/", 1)
        if len(split) == 2:
            english = split[1].strip()
            return english if english else np.nan
    
    return text if text else np.nan

def _translate_hausa(s):
    return HAUSA_ENG[s]

def _funnel_row(df, split_col, group_val):
    """Compute funnel proportions for one subgroup."""
    sub = df if group_val == "all" else df[df[split_col] == group_val]
    w = sub[WEIGHT_COL].sum()
    if w == 0:
        return None

    def wp(mask):
        return sub.loc[mask, WEIGHT_COL].sum() / w

    aware = wp(sub["birth_spacing"].fillna("NA").str.contains('Oui')) if "birth_spacing" in sub.columns else np.nan
    ever  = wp(sub["ever_use"].fillna("NA").str.contains('Oui'))       if "ever_use"      in sub.columns else np.nan
    curr  = wp(sub["current_use"].fillna("NA").str.contains('Oui'))    if "current_use"   in sub.columns else np.nan

    return {
        "split": split_col, "group": group_val,
        "aware": aware, "ever_used": ever, "current_use": curr,
    }


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [family_planning] running...")

    # ── Funnel ────────────────────────────────────────────────────────────────
    funnel_rows = []
    for split_col in SPLIT_COLS:
        funnel_rows.append(_funnel_row(df, split_col, "all"))
        for grp in df[split_col].dropna().unique():
            funnel_rows.append(_funnel_row(df, split_col, grp))
    save(pd.DataFrame([r for r in funnel_rows if r]),
         os.path.join(APP_DATA_DIR, "fp_funnel.csv"), "funnel")

    # ── Timing of next pregnancy ──────────────────────────────────────────────
    if "time_before_preferred_pregnancy" in df.columns:
        rows = []
        # Overall
        s = weighted_counts(df, "time_before_preferred_pregnancy", TIME_TO_PREGNANT,
                            exclude=(-88, -99))
        tmp = s.reset_index(); tmp.columns = ["label", "proportion"]
        tmp["split"] = "none"; tmp["group"] = "all"
        rows.append(tmp)
        # By split
        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, "time_before_preferred_pregnancy",
                                          split_col, TIME_TO_PREGNANT,
                                          exclude=(-88, -99))
            # melted = frame.reset_index().melt(id_vars="index", var_name="group",
            #                                   value_name="proportion")
            # melted.columns = ["label", "group", "proportion"]
            melted = safe_melt(frame)
            melted["split"] = split_col
            rows.append(melted)
        save(pd.concat(rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "fp_timing.csv"), "timing")

    # ── Methods (known / ever used / current) ─────────────────────────────────
    method_cols = {
        "known": "known_contraceptive_options",
        "ever":  "ever_used_methods",
        "current": "current_use_methods",
    }
    method_rows = []

    for method_type, col in method_cols.items():
        if col not in df.columns:
            print(f"  Column '{col}' not found")
            continue

        print(f"  Processing {method_type} ({col})...")
        
        # Strip hausa to get English only
        df[col] = df[col].apply(_strip_hausa)
        
        # Overall — use text-aware function; values are pipe-delimited English strings
        s = weighted_multiselect_counts_text(df, col, label_map=None, sep="|")

        tmp = s.reset_index(); tmp.columns = ["method", "proportion"]
        tmp["method_type"] = method_type; tmp["split"] = "none"; tmp["group"] = "all"
        method_rows.append(tmp)

        # By split
        for split_col in SPLIT_COLS:
            frame = split_weighted_multiselect_text(df, col, split_col, label_map=None, sep="|")
            melted = safe_melt(frame)
            melted["method_type"] = method_type
            melted["split"] = split_col
            method_rows.append(melted)
            
    if method_rows:
        result = pd.concat(method_rows, ignore_index=True)
        print(f"  Total fp_methods rows: {len(result)}")
        save(result, os.path.join(APP_DATA_DIR, "fp_methods.csv"), "methods")
    else:
        print("  [family_planning] WARNING: No method data found.")

    # ── Reason for use ────────────────────────────────────────────────────────
    if "reason_current_use" in df.columns:
        rows = []
        s = weighted_counts(df, "reason_current_use", REASON_USE, exclude=(-88, -99))
        tmp = s.reset_index(); tmp.columns = ["label", "proportion"]
        tmp["split"] = "none"; tmp["group"] = "all"
        rows.append(tmp)
        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, "reason_current_use", split_col,
                                          REASON_USE, exclude=(-88, -99))
            # melted = frame.reset_index().melt(id_vars="index", var_name="group",
            #                                   value_name="proportion")
            # melted.columns = ["label", "group", "proportion"]

            melted = safe_melt(frame)
            melted["split"] = split_col
            rows.append(melted)
        save(pd.concat(rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "fp_reason_use.csv"), "reason for use")

    # ── Future intent + considered use ───────────────────────────────────────
    intent_rows = []
    for col, label in [("future_intent", "future_intent"),
                       ("considered_use", "considered_use")]:
        if col not in df.columns:
            continue
        s = weighted_counts(df, col, YESNO, exclude=(-88, -99))
        tmp = s.reset_index(); tmp.columns = ["response", "proportion"]
        tmp["question"] = label; tmp["split"] = "none"; tmp["group"] = "all"
        intent_rows.append(tmp)
        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, col, split_col, YESNO, exclude=(-88, -99))
            # melted = frame.reset_index().melt(id_vars="index", var_name="group",
            #                                   value_name="proportion")
            # melted.columns = ["response", "group", "proportion"]

            melted = safe_melt(frame)
            melted["question"] = label; melted["split"] = split_col
            intent_rows.append(melted)
    if intent_rows:
        save(pd.concat(intent_rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "fp_intent.csv"), "intent")

    # ── Non-use reasons (free text — aggregated counts only, no raw text) ─────
    if "reason_current_nonuse" in df.columns:
        counts = (
            df[df["reason_current_nonuse"].notna()]["reason_current_nonuse"]
            .value_counts()
            .head(20)
            .reset_index()
        )
        counts.columns = ["Reason", "Unweighted count"]

        # translate to ENG
        counts["Reason"] = counts["Reason"].apply(_translate_hausa)
        
        save(counts, os.path.join(APP_DATA_DIR, "fp_nonuse_reasons.csv"),
             "non-use reasons")

    print("  [family_planning] done.")
