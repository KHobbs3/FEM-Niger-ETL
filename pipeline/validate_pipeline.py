"""
validate_pipeline.py
====================
Run BEFORE and AFTER the pipeline to catch data issues early.

Usage:
    python pipeline/validate_pipeline.py --raw          # validate raw survey CSV
    python pipeline/validate_pipeline.py --outputs      # validate generated CSVs
    python pipeline/validate_pipeline.py --all          # both (default)

The script never reads or prints PII — it only prints column names,
value distributions, counts, and structural summaries.
"""

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd

# ── Allow running from project root or pipeline/ ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import (
    DIR_MAPPED_DATA, APP_DATA_DIR, WEIGHT_COL,
    USER_GROUPS, NONUSER_GROUPS, ALL_USE_GROUPS,
    SPLIT_COLS, WTT_MAP,
    CONTRACEPTIVE_METHODS, LIFE_GOALS, ROLE_MODELS,
    LIKEABLE_TRAITS, FORMING_BELIEFS, DECISION_CONFIDENT,
)

# ── ANSI colours for terminal output ─────────────────────────────────────────
RED   = "\033[91m"
YEL   = "\033[93m"
GRN   = "\033[92m"
BOLD  = "\033[1m"
RESET = "\033[0m"

_errors   = []
_warnings = []


def _err(msg):
    _errors.append(msg)
    print(f"  {RED}✗ ERROR  {RESET}{msg}")


def _warn(msg):
    _warnings.append(msg)
    print(f"  {YEL}⚠ WARN   {RESET}{msg}")


def _ok(msg):
    print(f"  {GRN}✓ ok     {RESET}{msg}")


def _section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_raw(df):
    print(f"\n{BOLD}RAW SURVEY DATA VALIDATION{RESET}")
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    _validate_raw_core(df)
    _validate_raw_statements(df)
    _validate_raw_family_planning(df)
    _validate_raw_access(df)
    _validate_raw_personality(df)


def _validate_raw_core(df):
    _section("Core / shared columns")

    # Weight column
    if WEIGHT_COL not in df.columns:
        _err(f"Weight column '{WEIGHT_COL}' not found. All weighted calculations will fail.")
    else:
        nulls = df[WEIGHT_COL].isna().sum()
        zeros = (df[WEIGHT_COL] == 0).sum()
        neg   = (df[WEIGHT_COL] < 0).sum()
        if nulls:
            _warn(f"'{WEIGHT_COL}' has {nulls:,} null values — those rows excluded from all weighted stats.")
        if zeros:
            _warn(f"'{WEIGHT_COL}' has {zeros:,} zero values — those rows contribute nothing to weighted stats.")
        if neg:
            _err(f"'{WEIGHT_COL}' has {neg:,} negative values — this will corrupt all weighted stats.")
        _ok(f"'{WEIGHT_COL}' present. min={df[WEIGHT_COL].min():.4f}  max={df[WEIGHT_COL].max():.4f}  "
            f"mean={df[WEIGHT_COL].mean():.4f}")

    # 'use' column
    if "use" not in df.columns:
        _err("'use' column not found. All split-by-user-group charts will be blank.")
    else:
        vc = df["use"].value_counts(dropna=False)
        print(f"\n  'use' column value counts:")
        for val, cnt in vc.items():
            flag = "" if val in ALL_USE_GROUPS else f"  {YEL}⚠ not in ALL_USE_GROUPS{RESET}"
            print(f"    {str(val):<20} {cnt:>6,}{flag}")
        missing_groups = [g for g in ALL_USE_GROUPS if g not in df["use"].values]
        if missing_groups:
            _warn(f"Expected use groups not found in data: {missing_groups}")
        else:
            _ok("All expected use groups present.")

    # Split columns
    for col in SPLIT_COLS:
        if col not in df.columns:
            _warn(f"Split column '{col}' not found — split-by-{col} charts will be blank.")
        else:
            nuniq = df[col].nunique()
            nulls = df[col].isna().sum()
            _ok(f"'{col}': {nuniq} unique values, {nulls:,} nulls")
            if col == "age_group":
                vals = sorted(df[col].dropna().unique())
                print(f"    age_group values: {vals}")
                # Check for mixed dash styles (en-dash vs hyphen)
                has_endash  = any("–" in str(v) for v in vals)
                has_hyphen  = any("-" in str(v) for v in vals)
                if has_endash and has_hyphen:
                    _warn("age_group mixes en-dash (–) and hyphen (-) — this creates duplicate groups "
                          "in aggregations. Standardise to one style in the cleaning step.")


def _validate_raw_statements(df):
    _section("Statements ETL inputs")

    stmt_cols = [c for c in df.columns if c.startswith("statement_")]
    if not stmt_cols:
        _err("No columns starting with 'statement_' found. "
             "The statements ETL will produce no output.")
        return
    _ok(f"{len(stmt_cols)} statement columns found.")

    # Check response values
    sample_col = stmt_cols[0]
    vc = df[sample_col].value_counts(dropna=False).head(10)
    print(f"\n  Top values in '{sample_col}':")
    for val, cnt in vc.items():
        print(f"    {str(val):<40} {cnt:>6,}")

    has_agree    = df[stmt_cols].apply(lambda c: c.astype(str).str.contains("Agree").any()).any()
    has_disagree = df[stmt_cols].apply(lambda c: c.astype(str).str.contains("Disagree").any()).any()
    if not has_agree:
        _err("No 'Agree' values found in any statement column. "
             "The ETL maps agreement by str.contains('Agree') — check response encoding.")
    if not has_disagree:
        _warn("No 'Disagree' values found. Check response encoding.")
    if has_agree and has_disagree:
        _ok("'Agree' and 'Disagree' responses found.")


def _validate_raw_family_planning(df):
    _section("Family planning ETL inputs")

    # Funnel columns — the most likely source of all-zero output
    funnel_cols = {
        "birth_spacing":  "Awareness (funnel step 1) — expected integer 1=aware",
        "ever_use":       "Ever used (funnel step 2) — expected integer 1=Oui",
        "current_use":    "Currently using (funnel step 3) — expected integer 1=Oui",
    }
    print()
    for col, desc in funnel_cols.items():
        if col not in df.columns:
            _err(f"'{col}' not found. {desc}. Funnel will show 0% for this step.")
        else:
            vc = df[col].value_counts(dropna=False).head(8)
            n_ones = (df[col].fillna("NA").str.contains("Oui")).sum()
            pct = n_ones / len(df) * 100
            print(f"  '{col}' ({desc})")
            print(f"    → rows with value==1: {n_ones:,} ({pct:.1f}%)")
            if n_ones == 0:
                _err(f"    ZERO rows have {col}==1. This is why the funnel shows 0%. "
                     f"Check encoding — raw values are: {sorted(df[col].dropna().unique().tolist())[:10]}")
            else:
                _ok(f"    '{col}' has {n_ones:,} rows == 1")

    # Method columns
    method_cols = ["known_contraceptive_options", "ever_used_methods", "current_use_methods"]
    for col in method_cols:
        if col not in df.columns:
            _warn(f"'{col}' not found — method charts will be blank.")
        else:
            sample = df[col].dropna().head(3).tolist()
            _ok(f"'{col}' present. Sample values: {sample}")

    # Other FP columns
    for col in ["time_before_preferred_pregnancy", "reason_current_use",
                "future_intent", "considered_use", "reason_current_nonuse"]:
        if col not in df.columns:
            _warn(f"'{col}' not found — that FP chart will be blank.")
        else:
            _ok(f"'{col}' present.")


def _validate_raw_access(df):
    _section("Access ETL inputs")

    # --- Stockouts ---
    print()
    for col, desc in [
        ("stockouts_users",               "Users stockout flag"),
        ("nonusers_seek_contraceptives",   "Non-users seeking flag  ← KEY for 'sought' chart"),
        ("stockouts_nonusers",             "Non-users stockout flag"),
        ("stockouts_response",             "User stockout response"),
        ("stockouts_nonusers_response",    "Non-user stockout response"),
    ]:
        if col not in df.columns:
            _warn(f"'{col}' not found.")
        else:
            vc = df[col].value_counts(dropna=False).head(6)
            # Check whether Oui-detection will work
            has_Oui_str = df[col].astype(str).str.lower().str.contains("Oui").any()
            print(f"  '{col}' ({desc})")
            for val, cnt in vc.items():
                print(f"    {str(val):<35} {cnt:>6,}")
            if not has_Oui_str:
                _err(f"    No 'Oui' string found in '{col}'. "
                     f"ETL uses str.contains('Oui') — if values are 0/1 or Oui/Non, "
                     f"the column will always evaluate to False and produce 0% results.")
            else:
                _ok(f"    'Oui' strings found in '{col}'")

    # --- Travel ---
    print()
    for col in ["travel_time_users", "travel_time_nonusers", "willingness_to_travel"]:
        if col not in df.columns:
            _warn(f"'{col}' not found — travel section will be blank.")
        else:
            if col == "willingness_to_travel":
                vc = df[col].value_counts(dropna=False).head(6)
                unmapped = [v for v in df[col].dropna().unique() if v not in WTT_MAP]
                _ok(f"'{col}' present.")
                if unmapped:
                    _warn(f"  Values not in WTT_MAP (will become NaN): {unmapped[:10]}")
            else:
                _ok(f"'{col}' present. mean={df[col].mean():.1f}  "
                    f"nulls={df[col].isna().sum():,}")

    # --- Costs ---
    for col in ["user_costs", "nonuser_cost"]:
        if col not in df.columns:
            _warn(f"'{col}' not found — affordability section will be blank.")
        else:
            _ok(f"'{col}' present. mean={df[col].mean():.1f}  zeros={(df[col]==0).sum():,}")


def _validate_raw_personality(df):
    _section("Personality ETL inputs")

    multiselect_cols = {
        "life_goals":       LIFE_GOALS,
        "role_models":      ROLE_MODELS,
        "likeable_traits":  LIKEABLE_TRAITS,
        "forming_beliefs":  FORMING_BELIEFS,
    }
    for col, label_map in multiselect_cols.items():
        if col not in df.columns:
            _warn(f"'{col}' not found.")
        else:
            sample = df[col].dropna().head(3).tolist()
            _ok(f"'{col}' present. Sample: {sample}")

    for col in ["decision_confident", "decision_confident_3",
                "life_goals_main", "life_goals_achievable"]:
        status = "present" if col in df.columns else "NOT FOUND"
        fn = _ok if col in df.columns else _warn
        fn(f"'{col}': {status}")

    # Wellbeing — critical: check column names and value encoding
    print()
    for col in ["happiness", "satisfaction"]:
        if col not in df.columns:
            _err(f"Wellbeing column '{col}' NOT FOUND. "
                 f"The wellbeing ETL has a known bug where it references "
                 f"an undefined variable 'frame' in the split loop — "
                 f"this will crash even if the column exists.")
        else:
            vc = df[col].value_counts(dropna=False).head(8)
            print(f"  '{col}' values:")
            for val, cnt in vc.items():
                print(f"    {str(val):<6} {cnt:>6,}")
            _ok(f"'{col}' present.")


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT CSV VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_outputs():
    print(f"\n{BOLD}OUTPUT CSV VALIDATION{RESET}")

    _validate_out_statements()
    _validate_out_family_planning()
    _validate_out_access()
    _validate_out_personality()
    _validate_out_personas()


def _load_out(filename):
    path = os.path.join(APP_DATA_DIR, filename)
    if not os.path.exists(path):
        _warn(f"Output file not found: {path}")
        return None
    df = pd.read_csv(path)
    _ok(f"Loaded {filename}: {len(df):,} rows × {df.shape[1]} cols")
    return df


def _check_splits(df, filename, value_col="proportion"):
    """Check that each SPLIT_COL has data and no all-zero groups."""
    for split in SPLIT_COLS:
        sub = df[df["split"] == split] if "split" in df.columns else pd.DataFrame()
        if sub.empty:
            _warn(f"{filename}: no rows for split='{split}'")
            continue
        groups = sub["group"].unique().tolist()
        if value_col in sub.columns:
            all_zero = (sub[value_col] == 0).all()
            if all_zero:
                _err(f"{filename}: ALL values are 0 for split='{split}'. "
                     f"Groups: {groups}. Check raw data encoding.")
            else:
                _ok(f"{filename}: split='{split}' has {len(groups)} groups, "
                    f"values range [{sub[value_col].min():.3f}, {sub[value_col].max():.3f}]")


def _validate_out_statements():
    _section("statements_heatmap.csv")
    df = _load_out("statements_heatmap.csv")
    if df is None:
        return

    # ── Schema ───────────────────────────────────────────────────────────────
    required = [
        "label", "split", "group", "weighted_agreement",
        "agree_n", "agree_pct", "agree_weighted_n",
        "disagree_n", "disagree_pct", "disagree_weighted_n",
        "total_n", "total_weighted_n",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _err(f"Missing columns: {missing}")
        return
    _ok("All required columns present.")

    # ── Nulls in key columns ──────────────────────────────────────────────────
    for col in ["label", "split", "group", "weighted_agreement", "agree_pct",
                "disagree_pct", "total_n", "total_weighted_n"]:
        n_null = df[col].isna().sum()
        if n_null:
            _err(f"'{col}' has {n_null:,} null values.")
        else:
            _ok(f"'{col}': no nulls.")

    # ── Label count ───────────────────────────────────────────────────────────
    n_labels = df["label"].nunique()
    _ok(f"{n_labels} unique statement labels.")
    if n_labels < 3:
        _warn("Fewer than 3 statement labels — check that statement_ columns exist in raw data.")

    # ── weighted_agreement in [-1, 1] ─────────────────────────────────────────
    wa = df["weighted_agreement"].dropna()
    _ok(f"weighted_agreement range: [{wa.min():.3f}, {wa.max():.3f}]")
    if (wa == 0).all():
        _err("ALL weighted_agreement values are 0 — check Agree/Disagree encoding in raw data.")
    out_of_range = ((wa < -1) | (wa > 1)).sum()
    if out_of_range:
        _err(f"{out_of_range} weighted_agreement values outside [-1, 1]. "
             f"Score must be (weighted agrees − weighted disagrees) / total weighted n.")

    # ── agree_pct / disagree_pct in [0, 1] ───────────────────────────────────
    for col in ["agree_pct", "disagree_pct"]:
        p = df[col].dropna()
        _ok(f"{col} range: [{p.min():.3f}, {p.max():.3f}]")
        bad = ((p < 0) | (p > 1)).sum()
        if bad:
            _err(f"{bad} rows have {col} outside [0, 1].")

    # ── agree_pct + disagree_pct == 1 ────────────────────────────────────────
    # Denominator is agree_n + disagree_n (don't-know excluded), so the two
    # must sum to exactly 1 for any row where someone gave an opinion.
    pct_sum = df["agree_pct"] + df["disagree_pct"]
    opinioned = (df["agree_n"] + df["disagree_n"]) > 0
    not_one = opinioned & ((pct_sum - 1.0).abs() > 0.001)
    if not_one.sum():
        _err(f"{not_one.sum()} rows where agree_pct + disagree_pct ≠ 1 "
             f"(expected: denominator is agree_n + disagree_n).")
    else:
        _ok("agree_pct + disagree_pct = 1 for all rows with opinioned respondents.")

    # ── agree_n + disagree_n <= total_n ──────────────────────────────────────
    n_over = ((df["agree_n"] + df["disagree_n"]) > df["total_n"] + 0.5).sum()
    if n_over:
        _err(f"{n_over} rows where agree_n + disagree_n > total_n.")
    else:
        _ok("agree_n + disagree_n <= total_n in all rows.")

    # ── total_n / total_weighted_n positive ───────────────────────────────────
    zero_n = (df["total_n"] == 0).sum()
    if zero_n:
        _warn(f"{zero_n} rows with total_n == 0 (no respondents answered).")
    zero_wn = (df["total_weighted_n"] <= 0).sum()
    if zero_wn:
        _err(f"{zero_wn} rows with total_weighted_n <= 0.")

    # ── Overall (split=none, group=all) present for every label ──────────────
    overall = df[(df["split"] == "none") & (df["group"] == "all")]
    overall_labels = set(overall["label"].unique())
    all_labels     = set(df["label"].unique())
    missing_overall = all_labels - overall_labels
    if missing_overall:
        _err(f"{len(missing_overall)} labels have no overall (split=none/group=all) row: "
             f"{sorted(missing_overall)[:5]}{'...' if len(missing_overall) > 5 else ''}")
    else:
        _ok(f"All {n_labels} labels have an overall (split=none/group=all) row.")

    # ── Expected splits present ───────────────────────────────────────────────
    splits_present = set(df["split"].unique())
    expected_splits = {"none"} | set(SPLIT_COLS)
    missing_splits = expected_splits - splits_present
    if missing_splits:
        _err(f"Missing splits in output: {missing_splits}")
    else:
        _ok(f"All expected splits present: {sorted(splits_present)}")

    # ── use split: only user / nonuser groups (no past_user / future_user) ────
    use_rows = df[df["split"] == "use"]
    if use_rows.empty:
        _warn("No rows for split='use'.")
    else:
        use_groups = set(use_rows["group"].unique())
        unexpected = use_groups - {"user", "nonuser"}
        if unexpected:
            _err(f"Unexpected groups in use split: {unexpected}. "
                 f"past_user and future_user should have been collapsed into nonuser.")
        else:
            _ok(f"use split groups: {sorted(use_groups)} (user / nonuser only — correct).")
        for grp in ["user", "nonuser"]:
            if grp not in use_groups:
                _err(f"Expected group '{grp}' not found in use split.")

    # ── No duplicate label × split × group rows ───────────────────────────────
    dupes = df.duplicated(subset=["label", "split", "group"]).sum()
    if dupes:
        _err(f"{dupes} duplicate label × split × group rows — aggregation may have run twice.")
    else:
        _ok("No duplicate label × split × group rows.")


def _validate_out_family_planning():
    _section("fp_funnel.csv")
    df = _load_out("fp_funnel.csv")
    if df is None:
        return

    required = ["split", "group", "aware", "ever_used", "current_use"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        _err(f"Missing columns: {missing}")

    for col in ["aware", "ever_used", "current_use"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if (vals == 0).all():
            _err(f"FUNNEL: '{col}' is ALL ZEROS. "
                 f"This means the raw column (birth_spacing / ever_use / current_use) "
                 f"either doesn't exist or never equals 1. "
                 f"Run --raw to check raw value distributions.")
        else:
            _ok(f"funnel '{col}': range [{vals.min():.3f}, {vals.max():.3f}]")

    # Check for duplicate groups (mixed dash styles in age_group)
    age_groups = df[df["split"] == "age_group"]["group"].unique().tolist()
    endash = [v for v in age_groups if "–" in str(v)]
    hyphen = [v for v in age_groups if "-" in str(v) and "–" not in str(v)]
    if endash and hyphen:
        _warn(f"age_group has BOTH en-dash and hyphen variants: {endash + hyphen}. "
              f"This creates duplicate rows for the same age band.")

    _section("fp_methods.csv")
    df = _load_out("fp_methods.csv")
    if df is not None:
        for mt in ["known", "ever", "current"]:
            sub = df[df["method_type"] == mt] if "method_type" in df.columns else pd.DataFrame()
            if sub.empty:
                _warn(f"No rows for method_type='{mt}'")
            else:
                _ok(f"method_type='{mt}': {len(sub):,} rows")

    for fname in ["fp_timing.csv", "fp_reason_use.csv", "fp_intent.csv", "fp_nonuse_reasons.csv"]:
        _section(fname)
        df = _load_out(fname)
        if df is not None:
            _check_splits(df, fname)


def _validate_out_access():
    _section("access_stockouts.csv")
    df = _load_out("access_stockouts.csv")
    if df is None:
        return

    metrics = df["metric"].unique().tolist() if "metric" in df.columns else []
    _ok(f"Metrics present: {metrics}")

    expected_metrics = ["stockout_users", "sought_contraceptives", "stockout_nonusers_sought"]
    for m in expected_metrics:
        if m not in metrics:
            _warn(f"Expected metric '{m}' not found in access_stockouts.csv")

    for m in metrics:
        sub = df[df["metric"] == m]
        vals = sub["value"].dropna()
        if vals.empty or (vals == 0).all():
            _err(f"Metric '{m}': ALL values are 0 or empty. "
                 f"Groups present: {sub['group'].unique().tolist()}. "
                 f"Splits present: {sub['split'].unique().tolist()}.")
        else:
            _ok(f"Metric '{m}': {len(sub)} rows, "
                f"range [{vals.min():.3f}, {vals.max():.3f}]")

    # Check that nonuser appears under sought_contraceptives
    if "sought_contraceptives" in metrics:
        sought_groups = df[df["metric"] == "sought_contraceptives"]["group"].unique().tolist()
        if "nonuser" not in sought_groups:
            _warn(f"'nonuser' not in sought_contraceptives groups: {sought_groups}. "
                  f"Likely because NONUSER_GROUPS filter produced no nonuser rows, "
                  f"or the raw 'use' column spells it differently (e.g. 'non-user').")

    for fname in ["access_travel.csv", "access_affordability.csv",
                  "access_composite.csv", "access_stockout_responses.csv"]:
        _section(fname)
        df = _load_out(fname)
        if df is not None and "value" in df.columns:
            vals = df["value"].dropna()
            if vals.empty or (vals == 0).all():
                _warn(f"{fname}: all values are 0 or empty.")
            else:
                _ok(f"Values range: [{vals.min():.3f}, {vals.max():.3f}]")


def _validate_out_personality():
    outputs = [
        ("personality_life_goals.csv",          ["label", "split", "group", "proportion"]),
        ("personality_role_models.csv",          ["label", "split", "group", "proportion"]),
        ("personality_likeable_traits.csv",      ["label", "split", "group", "proportion"]),
        ("personality_forming_beliefs.csv",      ["label", "split", "group", "proportion"]),
        ("personality_decision_confident.csv",   ["label", "split", "group", "proportion"]),
        ("personality_wellbeing.csv",            ["label", "split", "group", "proportion", "question"]),
        ("personality_goals_achievable.csv",     ["label", "split", "group", "proportion"]),
    ]
    for fname, required_cols in outputs:
        _section(fname)
        df = _load_out(fname)
        if df is None:
            continue
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            _err(f"Missing columns: {missing}")
        else:
            _ok("Required columns present.")
        _check_splits(df, fname)
        if "proportion" in df.columns:
            p = df["proportion"].dropna()
            if (p == 0).all():
                _err(f"{fname}: ALL proportions are 0.")
            elif p.max() > 1.0:
                _warn(f"{fname}: some proportions > 1.0 (max={p.max():.3f}). "
                      f"Expected 0–1 range.")

    # Specific wellbeing check
    _section("Wellbeing ETL bug check")
    wb = _load_out("personality_wellbeing.csv")
    if wb is not None and "question" in wb.columns:
        for q in ["happiness", "satisfaction"]:
            sub = wb[wb["question"] == q]
            if sub.empty:
                _err(f"No rows for wellbeing question='{q}'. "
                     f"Known bug in etl_personality.py: the split loop references "
                     f"an undefined variable 'frame' instead of calling "
                     f"split_weighted_counts(). Fix: replace the split loop body with:\n"
                     + textwrap.indent(textwrap.dedent("""
                         from pipeline.utils import split_weighted_counts
                         for split_col in SPLIT_COLS:
                             frame = split_weighted_counts(df, col, split_col, LIKERT,
                                                           exclude=())
                             melted = frame.reset_index().melt(
                                 id_vars=frame.index.name or "index",
                                 var_name="group", value_name="proportion")
                             melted.columns = ["label", "group", "proportion"]
                             melted["question"] = tag
                             melted["split"] = split_col
                             wellbeing_rows.append(melted)
                     """).strip(), "                     "))
            else:
                splits_present = sub["split"].unique().tolist()
                _ok(f"question='{q}': {len(sub)} rows, splits: {splits_present}")


def _validate_out_personas():
    _section("personas_centroids.csv / personas_profile.csv")
    centroids = _load_out("personas_centroids.csv")
    if centroids is not None:
        _ok(f"{len(centroids)} personas found.")
        if "count" in centroids.columns:
            _ok(f"Persona sizes: {centroids['count'].tolist()}")
        if "weighted_count" in centroids.columns:
            _ok(f"Weighted sizes: {centroids['weighted_count'].round(1).tolist()}")

    profile = _load_out("personas_profile.csv")
    if profile is not None:
        vars_present = profile["variable"].unique().tolist() if "variable" in profile.columns else []
        _ok(f"Profile variables: {vars_present}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary():
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    if _errors:
        print(f"\n  {RED}{BOLD}{len(_errors)} ERROR(S):{RESET}")
        for e in _errors:
            print(f"    {RED}✗{RESET} {e}")
    if _warnings:
        print(f"\n  {YEL}{BOLD}{len(_warnings)} WARNING(S):{RESET}")
        for w in _warnings:
            print(f"    {YEL}⚠{RESET} {w}")
    if not _errors and not _warnings:
        print(f"\n  {GRN}{BOLD}All checks passed.{RESET}")
    elif not _errors:
        print(f"\n  {GRN}No errors.{RESET} Review warnings above.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate FEM pipeline inputs and outputs."
    )
    parser.add_argument("--raw",        action="store_true", help="Validate raw survey CSV")
    parser.add_argument("--outputs",    action="store_true", help="Validate all generated CSVs")
    parser.add_argument("--statements", action="store_true", help="Validate statements_heatmap.csv only")
    parser.add_argument("--all",        action="store_true", help="Run all validations (default)")
    args = parser.parse_args()

    only_statements = args.statements and not args.raw and not args.outputs and not args.all
    run_raw     = args.raw  or args.all or (not any([args.raw, args.outputs, args.statements, args.all]))
    run_outputs = args.outputs or args.all or (not any([args.raw, args.outputs, args.statements, args.all]))

    if run_raw:
        if not os.path.exists(DIR_MAPPED_DATA):
            print(f"{RED}Raw data file not found: {DIR_MAPPED_DATA}{RESET}")
            print("Set FEM_MAPPED_DATA env var or update DIR_MAPPED_DATA in config.py")
        else:
            print(f"Loading raw data from: {DIR_MAPPED_DATA}")
            df = pd.read_csv(DIR_MAPPED_DATA, low_memory=False)
            df.columns = df.columns.str.strip()
            validate_raw(df)

    if only_statements:
        print(f"\n{BOLD}OUTPUT CSV VALIDATION (statements only){RESET}")
        _validate_out_statements()
    elif run_outputs:
        validate_outputs()

    _print_summary()
    sys.exit(1 if _errors else 0)


if __name__ == "__main__":
    main()