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
  fp_unmet.csv              — unmet need / unmet demand by split group
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

# 2026-09-04 (ported from benin_app's etl_family_planning.py): "current_use"
# is just "are you doing *anything* to avoid pregnancy" -- that counts
# withdrawal, the calendar method, etc. as "using", which potentially
# inflates the headline rate relative to actually-effective methods.
# CONTRACEPTIVE_METHODS' keys 1-9 and 15 are WHO's modern-method categories
# (sterilisation, implants, pills, IUD, injectables, condoms, ring, patch,
# vaginal barrier methods, emergency pills); 10-14 (withdrawal, abstinence,
# calendar method, standard days, LAM) are the traditional/less-effective
# methods -- see the CONTRACEPTIVE_METHODS comment in pipeline/config.py for
# the source list.
MODERN_METHOD_KEYS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15}
MODERN_METHODS = {CONTRACEPTIVE_METHODS[k] for k in MODERN_METHOD_KEYS if k in CONTRACEPTIVE_METHODS}

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

def _mentions_modern_method(cell):
    """True if a pipe-delimited multi-select cell names >=1 modern/effective
    method (MODERN_METHODS). Shared by effective_use (current_use_methods)
    and aware_modern_method (known_contraceptive_options) below -- same
    check, different column. Ported from benin_app's etl_family_planning.py
    (2026-09-04)."""
    if pd.isna(cell):
        return False
    return any(m.strip() in MODERN_METHODS for m in str(cell).split("|"))


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

    # % using a modern/effective method specifically -- narrower than
    # "current_use" above, which counts traditional methods too.
    # (2026-09-04, ported from benin_app)
    effective = np.nan
    if "current_use_methods" in sub.columns:
        effective = wp(sub["current_use_methods"].apply(_mentions_modern_method))

    # Stricter awareness check: confirms self-reported "aware" (birth_spacing
    # Yes/No) actually names a real modern method, via known_contraceptive_options,
    # the XLSForm's own "if yes" follow-up to birth_spacing. Denominator is
    # still `w`, the full `sub` group -- respondents skipped past this
    # question (said "Non" to birth_spacing) count as "not aware of a modern
    # method" in the numerator while staying in the denominator.
    # (2026-09-04, ported from benin_app)
    aware_modern = np.nan
    if "known_contraceptive_options" in sub.columns:
        aware_modern = wp(sub["known_contraceptive_options"].apply(_mentions_modern_method))

    return {
        "split": split_col, "group": group_val,
        "aware": aware, "ever_used": ever, "current_use": curr,
        "effective_use": effective, "aware_modern_method": aware_modern,
    }


# 2026-09-04 (ported from benin_app's etl_family_planning.py): unmet need /
# unmet demand, added alongside the funnel above. Both use the same "all
# women" denominator as _funnel_row's current_use / effective_use, so mCPR
# (effective_use), unmet need and unmet demand line up under the standard
# FP2030-style "total demand" framework:
#   total demand      = mCPR + unmet need
#   demand satisfied   = mCPR / total demand
#
# time_before_preferred_pregnancy codes (see TIME_TO_PREGNANT above): 1/2 =
# wants a child within the year (no unmet need), 3/4 = wants to delay 1+
# years, 6 = wants no more children. 5 (already pregnant) and -88/-99
# (don't know / prefer not to say) are excluded from "wants to delay or
# limit" -- they simply don't match WANTS_TO_DELAY_CODES below.
#
# 2026-09-04 fix: time_before_preferred_pregnancy does NOT hold
# TIME_TO_PREGNANT's numeric codes -- like every other select_one field in
# this dataset it holds bilingual "Hausa / English" label text (verified by
# inspecting the column's full distinct value set in
# fem_survey_niger_mapped.csv -- schema-level text only, same kind of check
# already used for FORMATIVE_LABEL_TO_METHOD in 3_linkage/config.py).
# pd.to_numeric().isin([3,4,6]) against that text silently coerced every
# value to NaN, so unmet_need/unmet_demand were 0.0 for every group. Fixed
# by mapping the exact raw strings to their TIME_TO_PREGNANT code below.
TIME_TO_PREGNANT_RAW_TO_CODE = {
    "A cikin watanni 6 masu zuwa / Within the next 6 months": 1,
    "A cikin wattani 6 zuwa 12 / 6–12 months from now": 2,
    "A cikin shekara daya zuwa shekara biyu / 1–2 years from now": 3,
    "Sama da shekara biyu / More than 2 years from now": 4,
    "Ina da/ tuni tana da ciki / I am/she is already pregnant": 5,
    "Ba ni da niyar samun wasu yara / I am not planning on having more children": 6,
    "Ban sani ba / I dont know": -88,
    "Na fi son inyi shiru / Prefer not to say": -99,
}
WANTS_TO_DELAY_CODES = [3, 4, 6]


def _unmet_row(df, split_col, group_val):
    """Unmet need / unmet demand for one subgroup."""
    sub = df if group_val == "all" else df[df[split_col] == group_val]
    w = sub[WEIGHT_COL].sum()
    if w == 0:
        return None
    if "time_before_preferred_pregnancy" not in sub.columns or "current_use" not in sub.columns:
        return None

    def wp(mask):
        return sub.loc[mask, WEIGHT_COL].sum() / w

    # Unmet need: wants to delay/limit births (see WANTS_TO_DELAY_CODES)
    # but isn't currently using *any* method. This checks current_use, not
    # current_use_methods/MODERN_METHODS -- a traditional-method user still
    # counts as having her need "met" here, same convention DHS uses (it's
    # a separate question from whether that method is effective).
    wants_to_delay = (
        sub["time_before_preferred_pregnancy"]
        .map(TIME_TO_PREGNANT_RAW_TO_CODE)
        .isin(WANTS_TO_DELAY_CODES)
    )
    not_using = ~sub["current_use"].fillna("NA").str.contains('Oui')
    unmet_need_mask = wants_to_delay & not_using

    # Unmet demand: the narrower, preference-grounded cut of the above --
    # only women in the unmet-need group who also say they're open to /
    # interested in using contraception in future (future_intent). Left as
    # NaN (not 0) if future_intent isn't in this dataset, so it's visibly
    # missing rather than silently wrong.
    unmet_demand = np.nan
    if "future_intent" in sub.columns:
        open_to_future = sub["future_intent"].fillna("NA").str.contains('Oui')
        unmet_demand = wp(unmet_need_mask & open_to_future)

    return {
        "split": split_col, "group": group_val,
        "wants_to_delay_or_limit": wp(wants_to_delay),
        "unmet_need": wp(unmet_need_mask),
        "unmet_demand": unmet_demand,
    }


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [family_planning] running...")

    # Strip Hausa before _funnel_row's effective-method/aware-modern-method
    # checks (below) need it -- the "Methods" section further down strips
    # current_use_methods again for its own purposes, which is a harmless
    # no-op on already-English text. (2026-09-04, ported from benin_app)
    if "current_use_methods" in df.columns:
        df["current_use_methods"] = df["current_use_methods"].apply(_strip_hausa)
    if "known_contraceptive_options" in df.columns:
        df["known_contraceptive_options"] = df["known_contraceptive_options"].apply(_strip_hausa)

    # ── Funnel ────────────────────────────────────────────────────────────────
    funnel_rows = []
    for split_col in SPLIT_COLS:
        funnel_rows.append(_funnel_row(df, split_col, "all"))
        for grp in df[split_col].dropna().unique():
            funnel_rows.append(_funnel_row(df, split_col, grp))
    save(pd.DataFrame([r for r in funnel_rows if r]),
         os.path.join(APP_DATA_DIR, "fp_funnel.csv"), "funnel")

    # ── Unmet need / unmet demand (2026-09-04, ported from benin_app) ────────
    unmet_rows = []
    for split_col in SPLIT_COLS:
        unmet_rows.append(_unmet_row(df, split_col, "all"))
        for grp in df[split_col].dropna().unique():
            unmet_rows.append(_unmet_row(df, split_col, grp))
    save(pd.DataFrame([r for r in unmet_rows if r]),
         os.path.join(APP_DATA_DIR, "fp_unmet.csv"), "unmet need/demand")

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
