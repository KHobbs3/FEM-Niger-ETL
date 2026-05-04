"""
validate_weights.py
===================
Validates the three-stage weight construction in funcs.py /
01_calculate_weights_KH.ipynb.

Checks:
  • Input data integrity (required columns, GPS coords, location codes)
  • settlements_eligible & settlements_chosen consistency
  • Stage 1 — cluster inclusion probabilities and stratum weights
  • Stage 2 — household coverage per cluster
  • Stage 3 — eligible adults per household
  • Combined weight and non-response adjustment
  • Final combined_weight_adjusted — including NULL diagnosis
  • Normalised weight diagnostics

Usage (from project root):
    python pipeline/validate_weights.py

Or point at specific files:
    python pipeline/validate_weights.py \\
        --data     path/to/cleaned_data.csv \\
        --eligible path/to/niger_preselected_minpop_733.csv \\
        --chosen   path/to/settlements_chosen.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import DIR_MAPPED_DATA, APP_DATA_DIR, WEIGHT_COL

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED   = "\033[91m"
YEL   = "\033[93m"
GRN   = "\033[92m"
BOLD  = "\033[1m"
RESET = "\033[0m"

_errors   = []
_warnings = []
_infos    = []


def _err(msg):
    _errors.append(msg)
    print(f"  {RED}✗ ERROR  {RESET}{msg}")


def _warn(msg):
    _warnings.append(msg)
    print(f"  {YEL}⚠ WARN   {RESET}{msg}")


def _ok(msg):
    print(f"  {GRN}✓ ok     {RESET}{msg}")


def _info(msg):
    print(f"  {'·':>8} {msg}")


def _section(title):
    print(f"\n{BOLD}{'─'*70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*70}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_raw_inputs(data, settlements_eligible, settlements_chosen):
    _section("INPUT DATA INTEGRITY")

    # ── data ──────────────────────────────────────────────────────────────────
    _info(f"data: {len(data):,} rows × {data.shape[1]} cols")

    required_data = ['location', 'hh_member_index_list', 'eligible_adults_in_hh',
                     'deviceid', 'KEY']
    for col in required_data:
        if col not in data.columns:
            _warn(f"data: column '{col}' not found — may affect weight construction")
        else:
            nulls = data[col].isna().sum()
            if nulls:
                _warn(f"data['{col}']: {nulls:,} nulls")

    # GPS columns
    gps_cols = ['gps-Latitude', 'gps-Longitude', 'gps-Altitude']
    for col in gps_cols:
        if col not in data.columns:
            _warn(f"data: GPS column '{col}' missing — household_id will be sequential")
        else:
            nulls = data[col].isna().sum()
            if nulls:
                _warn(f"data['{col}']: {nulls:,} null GPS values → those HH IDs may collide")

    # ── settlements_eligible ──────────────────────────────────────────────────
    _info(f"settlements_eligible: {len(settlements_eligible):,} rows")
    for col in ['state', 'population', 'proportion', 'geom_id']:
        if col not in settlements_eligible.columns:
            _err(f"settlements_eligible: required column '{col}' missing")
        else:
            nulls = settlements_eligible[col].isna().sum()
            if nulls:
                _warn(f"settlements_eligible['{col}']: {nulls} nulls")

    if 'proportion' in settlements_eligible.columns:
        p = settlements_eligible['proportion']
        if not pd.api.types.is_numeric_dtype(p):
            _warn("settlements_eligible['proportion'] is not numeric — may need .astype(float)")
        elif p.max() > 1:
            _warn(f"settlements_eligible['proportion'] max={p.max():.4f} > 1. "
                  f"If these are percentages (0–100), divide by 100 before use.")
        else:
            _ok(f"settlements_eligible['proportion']: range [{p.min():.6f}, {p.max():.6f}]")

    # ── settlements_chosen ────────────────────────────────────────────────────
    _info(f"settlements_chosen: {len(settlements_chosen):,} rows")
    for col in ['state', 'population', 'sample_number', 'geom_id']:
        if col not in settlements_chosen.columns:
            _err(f"settlements_chosen: required column '{col}' missing")
        else:
            nulls = settlements_chosen[col].isna().sum()
            if nulls:
                _warn(f"settlements_chosen['{col}']: {nulls} nulls")

    if 'population' in settlements_chosen.columns:
        if not pd.api.types.is_numeric_dtype(settlements_chosen['population']):
            _warn("settlements_chosen['population'] is not numeric. "
                  "The notebook strips commas and casts to float — ensure this has been done.")

    # Duplicate sample_number check — causes households to be double-counted
    # in Stage 2 weight calculation
    if 'sample_number' in settlements_chosen.columns:
        dups = settlements_chosen[settlements_chosen['sample_number'].duplicated(keep=False)]
        if not dups.empty:
            dup_codes = sorted(dups['sample_number'].unique().tolist())
            _warn(
                f"settlements_chosen has {len(dups)} rows with duplicate sample_number "
                f"values: {dup_codes}. "
                f"These are likely replacement settlements surveyed when the original "
                f"cluster was inaccessible. The weight calculation merges on sample_number "
                f"so duplicates will assign multiple weight rows to respondents. "
                f"Resolution: assign unique sample_number codes to replacement settlements "
                f"(e.g. 6a/6b or 60/61) and update the location codes in the survey data, "
                f"OR collapse to a single row by summing households_visited before merging."
            )
            _info("Duplicate rows:")
            for _, row in dups.iterrows():
                _info(f"  sample_number={row['sample_number']}  "
                      f"name={row.get('name', '?')}  "
                      f"state={row.get('state', '?')}  "
                      f"population={row.get('population', '?')}")

    # ── Location code alignment ───────────────────────────────────────────────
    _section("LOCATION CODE ALIGNMENT")
    if 'location' in data.columns and 'sample_number' in settlements_chosen.columns:
        data_locs    = set(data['location'].dropna().unique())
        chosen_locs  = set(settlements_chosen['sample_number'].dropna().unique())

        in_data_not_chosen = data_locs - chosen_locs
        in_chosen_not_data = chosen_locs - data_locs

        if in_data_not_chosen:
            _err(f"{len(in_data_not_chosen)} location codes in data NOT in settlements_chosen: "
                 f"{sorted(in_data_not_chosen)}  "
                 f"→ those respondents will get NULL weights after the merge.")
        else:
            _ok("All data location codes found in settlements_chosen.")

        if in_chosen_not_data:
            _warn(f"{len(in_chosen_not_data)} settlements_chosen codes NOT in data "
                  f"(no respondents surveyed there): {sorted(in_chosen_not_data)}")
        else:
            _ok("All chosen settlements have at least one respondent.")

        # This is the direct cause of NULL combined_weight_adjusted
        n_null_weight = data[~data['location'].isin(chosen_locs)].shape[0]
        if n_null_weight:
            _err(f"{n_null_weight:,} respondents will receive NULL weights because their "
                 f"location code doesn't match any sample_number. "
                 f"Fix: ensure location codes in data match sample_number in settlements_chosen "
                 f"(check for leading zeros, string vs integer types, etc.).")
            _info("Type check:")
            _info(f"  data['location'].dtype        = {data['location'].dtype}")
            _info(f"  settlements_chosen['sample_number'].dtype = "
                  f"{settlements_chosen['sample_number'].dtype}")
            if data['location'].dtype != settlements_chosen['sample_number'].dtype:
                _warn("Dtype mismatch — try: "
                      "settlements_chosen['sample_number'] = "
                      "settlements_chosen['sample_number'].astype(data['location'].dtype)")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE-BY-STAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_stage1(settlements_eligible, settlements_chosen):
    _section("STAGE 1: CLUSTER SELECTION WEIGHTS")

    if not {'state', 'proportion', 'geom_id'}.issubset(settlements_eligible.columns):
        _err("Cannot validate Stage 1 — missing columns in settlements_eligible")
        return
    if not {'state', 'geom_id', 'sample_number'}.issubset(settlements_chosen.columns):
        _err("Cannot validate Stage 1 — missing columns in settlements_chosen")
        return

    # Check every chosen settlement is in eligible list
    chosen_geoms   = set(settlements_chosen['geom_id'].dropna())
    eligible_geoms = set(settlements_eligible['geom_id'].dropna())
    missing = chosen_geoms - eligible_geoms
    if missing:
        _err(f"{len(missing)} chosen settlements not in eligible list: {missing}. "
             f"Inclusion probabilities cannot be calculated for these.")
    else:
        _ok("All chosen settlements found in eligible list.")

    # Check proportion column sums to ~1 per stratum (or globally)
    for state, grp in settlements_eligible.groupby('state'):
        prop_sum = grp['proportion'].sum()
        if abs(prop_sum - 1.0) > 0.01:
            _warn(f"settlements_eligible state='{state}': "
                  f"proportions sum to {prop_sum:.4f} (expected ~1.0). "
                  f"This affects within-stratum inclusion probabilities.")
        else:
            _ok(f"State '{state}': proportions sum to {prop_sum:.4f} ✓")

    # Clusters per stratum
    counts = settlements_chosen['state'].value_counts()
    _info("Clusters per stratum:")
    for state, k in counts.items():
        _info(f"  {state}: {k} clusters selected")

    # If stage1_weight already computed
    if 'stage1_weight' in settlements_chosen.columns:
        w = settlements_chosen['stage1_weight'].dropna()
        _ok(f"stage1_weight: mean={w.mean():.2f}  "
            f"range=[{w.min():.2f}, {w.max():.2f}]")
        if w.max() / w.min() > 20:
            _warn(f"Very high stage1 weight ratio ({w.max()/w.min():.1f}x) — "
                  f"extreme clusters will dominate the weighted estimates.")


def validate_stage2(data, settlements_chosen):
    _section("STAGE 2: HOUSEHOLD SELECTION WEIGHTS (APPROXIMATION)")

    if 'location' not in data.columns:
        _err("'location' column missing — cannot validate Stage 2")
        return

    # Households per cluster
    hh_per_cluster = data.groupby('location')['household_id'].nunique() \
        if 'household_id' in data.columns else data.groupby('location').size()

    _info(f"Households surveyed per cluster:")
    _info(f"  mean={hh_per_cluster.mean():.1f}  "
          f"min={hh_per_cluster.min()}  max={hh_per_cluster.max()}")

    if hh_per_cluster.min() < 10:
        _warn(f"Some clusters have very few households (<10): "
              f"{hh_per_cluster[hh_per_cluster < 10].to_dict()}. "
              f"Stage 2 weights will be very large for these.")

    # Population available?
    if 'population' in settlements_chosen.columns and 'sample_number' in settlements_chosen.columns:
        pop = settlements_chosen.set_index('sample_number')['population']
        missing_pop = [loc for loc in hh_per_cluster.index if loc not in pop.index]
        if missing_pop:
            _warn(f"No population data for locations: {missing_pop[:10]}")

        if 'stage2_weight' in settlements_chosen.columns:
            w = settlements_chosen['stage2_weight'].dropna()
            _ok(f"stage2_weight: mean={w.mean():.2f}  "
                f"range=[{w.min():.2f}, {w.max():.2f}]")
            extreme = settlements_chosen[settlements_chosen['stage2_weight'] > 100]
            if not extreme.empty:
                _warn(f"{len(extreme)} clusters with stage2_weight > 100. "
                      f"High convenience-sampling weights add variance. "
                      f"Locations: {extreme['sample_number'].tolist()}")


def validate_stage3(data):
    _section("STAGE 3: RESPONDENT SELECTION WEIGHTS")

    if 'eligible_adults_in_hh' not in data.columns:
        _err("'eligible_adults_in_hh' column missing — Stage 3 weights cannot be computed")
        return

    ea = data['eligible_adults_in_hh']
    nulls = ea.isna().sum()
    zeros = (ea == 0).sum()

    if nulls:
        _warn(f"eligible_adults_in_hh: {nulls:,} null values → those rows get NULL stage3_weight")
    if zeros:
        _err(f"eligible_adults_in_hh: {zeros:,} zero values. "
             f"stage3_weight = eligible adults, so 0 is impossible — "
             f"check parse_n() parsing (it subtracts 1 from last digit, "
             f"so a raw value ending in '1' → 0 eligible adults).")
    else:
        _ok(f"eligible_adults_in_hh: min={ea.min():.0f}  "
            f"max={ea.max():.0f}  mean={ea.mean():.2f}")

    # Distribution
    vc = ea.value_counts().sort_index()
    _info("Distribution of eligible adults per HH:")
    for n_elig, cnt in vc.items():
        _info(f"  {int(n_elig):>2} eligible: {cnt:>4} households ({cnt/len(data)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED WEIGHT & NULL DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_combined_weight(data):
    _section("COMBINED WEIGHT & combined_weight_adjusted")

    weight_cols = ['stage1_weight', 'stage2_weight', 'stage3_weight',
                   'combined_weight', 'combined_weight_adjusted',
                   'normalized_weight']

    for col in weight_cols:
        if col not in data.columns:
            _warn(f"'{col}' column not present in data.")
            continue

        nulls = data[col].isna().sum()
        neg   = (data[col].dropna() < 0).sum()
        zeros = (data[col].dropna() == 0).sum()
        w     = data[col].dropna()

        if nulls:
            _err(f"'{col}': {nulls:,} NULL values ({nulls/len(data)*100:.1f}% of sample). "
                 f"Those respondents are EXCLUDED from all weighted statistics.")
        else:
            _ok(f"'{col}': no nulls")

        if neg:
            _err(f"'{col}': {neg:,} NEGATIVE values — will corrupt all stats.")
        if zeros:
            _warn(f"'{col}': {zeros:,} zero values — those respondents contribute nothing.")

        if not w.empty:
            _info(f"  range=[{w.min():.4f}, {w.max():.4f}]  "
                  f"mean={w.mean():.4f}  std={w.std():.4f}")

    # ── NULL root-cause diagnosis ─────────────────────────────────────────────
    _section("NULL WEIGHT ROOT-CAUSE DIAGNOSIS")

    target = WEIGHT_COL  # 'combined_weight_adjusted'
    if target not in data.columns:
        _warn(f"'{target}' not in data — skipping null diagnosis")
        return

    null_rows = data[data[target].isna()]
    if null_rows.empty:
        _ok(f"No null values in '{target}'.")
        return

    _err(f"{len(null_rows):,} rows have null '{target}'.")
    _info("Diagnosing which stage introduced the nulls:")

    for stage_col in ['stage1_weight', 'stage2_weight', 'stage3_weight']:
        if stage_col in data.columns:
            n = null_rows[stage_col].isna().sum()
            _info(f"  {stage_col}: {n} nulls among the {len(null_rows)} affected rows")

    # Most likely cause: location not in settlements_chosen
    if 'location' in data.columns and 'stage1_weight' in data.columns:
        null_locs = null_rows[null_rows['stage1_weight'].isna()]['location'].value_counts()
        if not null_locs.empty:
            _err(f"These location codes produced null Stage 1 weights "
                 f"(not matched in settlements_chosen):")
            for loc, cnt in null_locs.items():
                _info(f"    location={loc!r}: {cnt} respondents")

    # eligible_adults_in_hh = 0 causes null stage3
    if 'eligible_adults_in_hh' in data.columns:
        zero_ea = null_rows[null_rows.get('stage3_weight', pd.Series(dtype=float)).isna()
                            if 'stage3_weight' in null_rows.columns
                            else null_rows['eligible_adults_in_hh'] == 0]
        if not zero_ea.empty:
            _warn(f"{len(zero_ea)} null weights may be from eligible_adults_in_hh == 0")

    # non-response column missing
    if 'household_response_rate' in data.columns:
        null_rr = null_rows['household_response_rate'].isna().sum()
        if null_rr:
            _err(f"{null_rr} rows have null household_response_rate. "
                 f"combined_weight_adjusted = combined_weight / response_rate, "
                 f"so a null response rate → null final weight.")
    else:
        _info("'household_response_rate' not in data — non-response "
              "adjustment was skipped (combined_weight used as-is).")


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISED WEIGHT DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_normalised_weight(data):
    _section("NORMALISED WEIGHT DIAGNOSTICS")

    col = 'normalized_weight'
    if col not in data.columns:
        _warn(f"'{col}' not in data — run calculate_three_stage_weights first.")
        return

    w = data[col].dropna()
    n = len(data)

    # Sum should equal sample size
    wsum = w.sum()
    if abs(wsum - n) > 1:
        _warn(f"normalized_weight sums to {wsum:.2f} but sample size is {n}. "
              f"Expected them to be equal — check if null weights are excluded.")
    else:
        _ok(f"normalized_weight sums to {wsum:.2f} ≈ n={n} ✓")

    # Mean should be 1.0
    if abs(w.mean() - 1.0) > 0.05:
        _warn(f"Mean normalized_weight = {w.mean():.4f} (expected 1.0). "
              f"Likely caused by {data[col].isna().sum()} null weights excluded from sum.")
    else:
        _ok(f"Mean normalized_weight = {w.mean():.4f} ✓")

    # Design effect
    n_eff  = (w.sum() ** 2) / (w ** 2).sum()
    deff   = len(w) / n_eff
    cv     = w.std() / w.mean()

    _info(f"Design effect (DEFF):     {deff:.3f}")
    _info(f"Effective sample size:    {n_eff:.0f} of {len(w)}")
    _info(f"Efficiency:               {(n_eff/len(w))*100:.1f}%")
    _info(f"Coeff. of variation (CV): {cv:.3f}")

    if deff > 3.0:
        _warn(f"Very high DEFF ({deff:.2f}) — standard errors are >√3 times larger than SRS. "
              f"Consider trimming extreme weights.")
    elif deff > 2.0:
        _warn(f"Elevated DEFF ({deff:.2f})")
    else:
        _ok(f"DEFF {deff:.2f} is acceptable.")

    if cv > 0.5:
        _warn(f"High CV ({cv:.3f}) — large weight variation increases variance of estimates. "
              f"Consider trimming weights above the 99th percentile "
              f"({np.percentile(w, 99):.3f}).")

    # Extreme weights
    p99 = np.percentile(w, 99)
    p1  = np.percentile(w, 1)
    extreme_hi = (w > p99).sum()
    extreme_lo = (w < p1).sum()

    _info(f"Weights above 99th pctile ({p99:.3f}): {extreme_hi}")
    _info(f"Weights below  1st pctile ({p1:.3f}): {extreme_lo}")

    # Stratum-level check
    if 'state' in data.columns:
        _info("\nWeighted vs population proportions by stratum:")
        wt_sum = w.sum()
        for state, grp in data.dropna(subset=[col]).groupby('state'):
            wp = grp[col].sum() / wt_sum
            up = len(grp) / len(w)
            _info(f"  {state:<10} unweighted={up:.1%}  weighted={wp:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary():
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  WEIGHT VALIDATION SUMMARY{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    if _errors:
        print(f"\n  {RED}{BOLD}{len(_errors)} ERROR(S):{RESET}")
        for e in _errors:
            print(f"    {RED}✗{RESET} {e}")
    if _warnings:
        print(f"\n  {YEL}{BOLD}{len(_warnings)} WARNING(S):{RESET}")
        for w in _warnings:
            print(f"    {YEL}⚠{RESET} {w}")
    if not _errors and not _warnings:
        print(f"\n  {GRN}{BOLD}All weight checks passed.{RESET}")
    elif not _errors:
        print(f"\n  {GRN}No errors.{RESET} Review warnings above.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Validate FEM survey weight construction.")
    parser.add_argument("--data",     default=DIR_MAPPED_DATA,
                        help="Path to cleaned/weighted survey CSV")
    parser.add_argument("--eligible", default=None,
                        help="Path to niger_preselected_minpop_733.csv")
    parser.add_argument("--chosen",   default=None,
                        help="Path to settlements_chosen.csv")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"{RED}Survey data not found: {args.data}{RESET}")
        print("Set FEM_MAPPED_DATA env var or pass --data path/to/file.csv")
        sys.exit(1)

    print(f"Loading: {args.data}")
    data = pd.read_csv(args.data, low_memory=False)
    data.columns = data.columns.str.strip()

    # Settlements files — try common relative paths if not provided.
    # Do NOT pass index_col=0: settlements_chosen has sample_number as a plain
    # column; loading with index_col=0 silently promotes it to the index and
    # the validator then reports it as missing.
    def _try_load(path, candidates, label):
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loading {label}: {path}  ({len(df)} rows)")
            return df
        for c in candidates:
            if os.path.exists(c):
                df = pd.read_csv(c)
                print(f"Loading {label}: {c}  ({len(df)} rows)")
                return df
        print(f"{YEL}⚠  {label} not found — some checks will be skipped.{RESET}")
        return None

    eligible_candidates = [
        "pipeline/data/niger_preselected_minpop_733.csv",
        "../table_analysis/data/meta/niger_preselected_minpop_733.csv",
    ]
    chosen_candidates = [
        "pipeline/data/settlements_chosen.csv",
        "../table_analysis/data/meta/settlements_chosen.csv",
    ]

    settlements_eligible = _try_load(args.eligible, eligible_candidates, "settlements_eligible")
    settlements_chosen   = _try_load(args.chosen,   chosen_candidates,   "settlements_chosen")

    # Fix population dtype (notebook strips commas)
    if settlements_chosen is not None and 'population' in settlements_chosen.columns:
        settlements_chosen['population'] = (
            settlements_chosen['population']
            .astype(str).str.replace(",", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    # ── Run validations ────────────────────────────────────────────────────────
    if settlements_eligible is not None and settlements_chosen is not None:
        validate_raw_inputs(data, settlements_eligible, settlements_chosen)
        validate_stage1(settlements_eligible, settlements_chosen)
        validate_stage2(data, settlements_chosen)
    else:
        _section("INPUT DATA INTEGRITY (partial — settlement files not found)")
        _warn("Settlement files not provided — Stage 1 and Stage 2 checks skipped.")

    validate_stage3(data)
    validate_combined_weight(data)
    validate_normalised_weight(data)

    _print_summary()
    sys.exit(1 if _errors else 0)


if __name__ == "__main__":
    main()
