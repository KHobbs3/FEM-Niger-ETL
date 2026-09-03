"""
ETL: Personas page
Runs k-modes clustering on the PII dataset and exports only the
cluster centroids + counts — no individual-level data in output.

Output files (overall):
  personas_centroids.csv   — cluster centroid for each variable + count
  personas_profile.csv     — per-persona summary stats (demographics)

Output files (by gender):
  personas_centroids_by_gender.csv — centroids per gender × cluster
  personas_profile_by_gender.csv   — profile per gender × cluster
  personas_elbow.csv               — within-cluster cost for k=1..6 per gender

Output files (by region, 4-way: North-East/North-West/South-South/Mid-South):
  personas_centroids_by_region.csv
  personas_profile_by_region.csv
  personas_elbow_by_region.csv     — also records the auto-selected k per region

Output files (by region, 2-way North/South -- 2026-09-02):
  personas_centroids_by_region_ns.csv
  personas_profile_by_region_ns.csv
  personas_elbow_by_region_ns.csv

Output files (standalone culture-clustering, 2026-09-02):
  culture_clusters_centroids.csv
  culture_clusters_profile.csv     — includes each cluster's region mix, to
                                      check whether cultural clusters concentrate
                                      geographically (region is NOT a clustering
                                      input here -- see CULTURE_CLUSTERING_VARS)
  culture_clusters_elbow.csv
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import (
    WEIGHT_COL, APP_DATA_DIR, VARS_FOR_CLUSTERING, CULTURE_CLUSTERING_VARS,
    PROVINCE_TO_REGION, REGION_TO_NORTH_SOUTH,
)
from pipeline.utils import save

N_CLUSTERS = 3       # fixed k for splits that don't use auto-detection (overall, gender)
K_MAX = 6             # maximum k to test for elbow plot / auto-detection

# When clustering within a single gender, drop gender from features (it's constant
# within the split, so it would be a zero-variance feature). Region isn't itself a
# clustering variable, so no equivalent exclusion is needed for the region split.
VARS_FOR_CLUSTERING_GENDER = [v for v in VARS_FOR_CLUSTERING if v != "gender"]

GENDER_LABELS = {
    "female": "Femme Nyɔnu",
    "male":   "Homme Sunnu",
}


# ── Feature engineering ──────────────────────────────────────────────────────

def _strip_fon(text):
    """French half of a bilingual 'French/Fon' label; handles pipe-delimited
    multi-select values. Same convention as etl_family_planning.py's
    _strip_hausa() / etl_radio.py's strip_fon()."""
    if pd.isna(text):
        return np.nan
    text = str(text).strip()
    if "|" in text:
        parts = [_strip_fon(p) for p in text.split("|")]
        parts = [p for p in parts if pd.notna(p) and p != ""]
        return "|".join(parts) if parts else np.nan
    if "/" in text:
        french = text.split("/", 1)[0].strip()
        return french if french else np.nan
    return text if text else np.nan


def _first_value(text):
    """First pipe-delimited value only -- reason_use_main is oddly a
    select_multiple for what's framed as "your ONE main reason", so this
    takes the first-mentioned answer as the respondent's primary one rather
    than treating each distinct combination as its own category (which would
    fragment the clustering the same way unstripped free text did on the
    radio page)."""
    if pd.isna(text):
        return np.nan
    return str(text).split("|")[0].strip()


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, in place-ish (returns the same df with new columns):
      region      -- 4-way region from province (North-East/North-West/
                      South-South/Mid-South), via PROVINCE_TO_REGION
      top_driver  -- respondent's main reason for using contraception
                      (reason_use_main, users/past_users only; NaN for
                      non-users -- filled to "Unknown" at clustering time,
                      same convention as other clustering vars)
      top_barrier -- respondent's main reason for NOT using contraception
                      (reason_nonuse_main, non-users/future_users only)

    reason_use_main / reason_nonuse_main are confirmed field names from
    table_analysis/03_driver_barrier_table_w_counts.ipynb (the "main_drivers"/
    "main_barriers" columns there) -- NOT the statement_1..62 agreement scale,
    whose "barrier" classification column in statement_labels.csv turned out
    to be entirely empty/unused.
    """
    if "province" in df.columns:
        df["region"] = df["province"].map(PROVINCE_TO_REGION)
    else:
        print("  [personas] WARNING: 'province' column not found — 'region' unavailable.")

    if "reason_use_main" in df.columns:
        df["top_driver"] = df["reason_use_main"].apply(_strip_fon).apply(_first_value)
    else:
        print("  [personas] WARNING: 'reason_use_main' column not found — 'top_driver' unavailable.")
        df["top_driver"] = np.nan

    if "reason_nonuse_main" in df.columns:
        df["top_barrier"] = df["reason_nonuse_main"].apply(_strip_fon).apply(_first_value)
    else:
        print("  [personas] WARNING: 'reason_nonuse_main' column not found — 'top_barrier' unavailable.")
        df["top_barrier"] = np.nan

    return df


# ── Elbow / auto-k ────────────────────────────────────────────────────────────

def _find_elbow_k(elbow_rows: list, k_min: int = 1, k_max: int = K_MAX) -> int:
    """
    Pick k from a k=1..k_max cost curve via the standard "distance to the
    chord" (kneedle-style) method: draw a straight line from (k_min, cost at
    k_min) to (k_max, cost at k_max), then pick the k whose point on the
    curve is furthest (perpendicular distance) from that line -- the most
    pronounced bend. No external dependency (kneed isn't installed in this
    environment) -- this is a direct, standard implementation of the same
    idea kneed's default detector uses.

    elbow_rows: list of {"k": int, "cost": float} dicts (or with extra keys,
    ignored). Falls back to N_CLUSTERS if there's not enough data to detect
    a bend (e.g. small group, k_max not reached).
    """
    pts = sorted({(r["k"], r["cost"]) for r in elbow_rows if k_min <= r["k"] <= k_max})
    if len(pts) < 3:
        return N_CLUSTERS

    ks = np.array([p[0] for p in pts], dtype=float)
    costs = np.array([p[1] for p in pts], dtype=float)

    # Normalize both axes to [0, 1] so the two very different scales (k vs.
    # cost) don't distort which point looks "furthest" from the chord.
    k_range = ks.max() - ks.min()
    cost_range = costs.max() - costs.min()
    if k_range == 0 or cost_range == 0:
        return N_CLUSTERS
    x = (ks - ks.min()) / k_range
    y = (costs - costs.min()) / cost_range

    # Perpendicular distance from each point to the line through the first
    # and last normalized points.
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    line_len = np.hypot(x1 - x0, y1 - y0)
    if line_len == 0:
        return N_CLUSTERS
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / line_len

    # Exclude the endpoints themselves (k_min/k_max, distance 0 by
    # construction) from being picked as "the elbow".
    dist[0] = dist[-1] = -1
    best_idx = int(np.argmax(dist))
    return int(ks[best_idx])


# ── Core clustering ───────────────────────────────────────────────────────────

def _cluster_split(df_split, split_col, split_label, cluster_vars, KModes, auto_k: bool = False):
    """
    Run k-modes for k=1..K_MAX to collect elbow data, pick a final k (either
    the fixed N_CLUSTERS, or auto-detected from the elbow curve if
    auto_k=True -- requested specifically for region splits, see
    _find_elbow_k), then fit the final model, for one group of a split (one
    gender, or one region).
    Returns (centroids_df, profile_rows, elbow_rows), each keyed by `split_col`.
    """
    cluster_df = df_split[cluster_vars].copy().fillna("Unknown")
    X = cluster_df.to_numpy()

    # ── Elbow data ────────────────────────────────────────────────────────────
    elbow_rows = []
    for k in range(1, K_MAX + 1):
        km = KModes(n_clusters=k, init="Cao" if k > 1 else "random",
                    n_init=3, verbose=0, random_state=42)
        km.fit(X)
        elbow_rows.append({split_col: split_label, "k": k, "cost": km.cost_})

    n_clusters = _find_elbow_k(elbow_rows) if auto_k else N_CLUSTERS
    for row in elbow_rows:
        row["chosen_k"] = n_clusters

    # ── Final model ───────────────────────────────────────────────────────────
    km_final = KModes(n_clusters=n_clusters, init="Cao", n_init=5,
                      verbose=0, random_state=42)
    clusters = km_final.fit_predict(X)

    # ── Centroids ─────────────────────────────────────────────────────────────
    centroids = pd.DataFrame(km_final.cluster_centroids_, columns=cluster_vars)
    centroids.insert(0, split_col, split_label)
    centroids.insert(1, "persona", range(n_clusters))

    counts = pd.Series(clusters).value_counts().sort_index()
    centroids["count"] = counts.values

    if WEIGHT_COL in df_split.columns:
        df_c = df_split.copy()
        df_c["_cluster"] = clusters
        weighted_counts = df_c.groupby("_cluster")[WEIGHT_COL].sum()
        centroids["weighted_count"] = weighted_counts.values

    # ── Per-persona profile ───────────────────────────────────────────────────
    df_c = df_split.copy()
    df_c["_cluster"] = clusters

    profile_rows = []
    for persona_id in range(n_clusters):
        sub = df_c[df_c["_cluster"] == persona_id]
        n = len(sub)
        w = sub[WEIGHT_COL].sum() if WEIGHT_COL in sub.columns else n

        for col in cluster_vars:
            if col == "age" and pd.api.types.is_numeric_dtype(sub[col]):
                profile_rows.append({
                    split_col: split_label, "persona": persona_id,
                    "variable": col, "value": "mean",
                    "proportion": sub[col].mean(),
                })
            else:
                vc = sub[col].value_counts(normalize=True).head(3)
                for val, prop in vc.items():
                    profile_rows.append({
                        split_col: split_label, "persona": persona_id,
                        "variable": col, "value": str(val),
                        "proportion": prop,
                    })

        # Additional demographic breakdown -- gender/age_group marginals, plus
        # (2026-09-02) a combined age_group x gender cross-tab specifically
        # for region splits, so "region, then age and gender" reads as a real
        # nested breakdown rather than two separate flat lists that don't tie
        # back to each other.
        for col in ["gender", "age_group", "use", "region"]:
            if col in sub.columns and col not in cluster_vars:
                vc = sub[col].value_counts(normalize=True)
                for val, prop in vc.items():
                    profile_rows.append({
                        split_col: split_label, "persona": persona_id,
                        "variable": col, "value": str(val),
                        "proportion": prop,
                    })

        if split_col in ("region", "region_ns") and "age_group" in sub.columns and "gender" in sub.columns:
            combo = sub.groupby(["age_group", "gender"]).size()
            combo_total = combo.sum()
            if combo_total > 0:
                for (age_val, gender_val), n_combo in combo.items():
                    profile_rows.append({
                        split_col: split_label, "persona": persona_id,
                        "variable": "age_gender_combo",
                        "value": f"{age_val} / {gender_val}",
                        "proportion": n_combo / combo_total,
                    })

        profile_rows.append({
            split_col: split_label, "persona": persona_id,
            "variable": "_count", "value": "n", "proportion": n,
        })
        profile_rows.append({
            split_col: split_label, "persona": persona_id,
            "variable": "_count", "value": "weighted_n", "proportion": w,
        })

    return centroids, profile_rows, elbow_rows


def _cluster_gender(df_gender, gender_label, KModes):
    return _cluster_split(df_gender, "gender", gender_label, VARS_FOR_CLUSTERING_GENDER, KModes)


def _run_region_split(df, region_col, region_values, split_col_name, label, KModes):
    """Shared implementation for both the 4-way and 2-way region splits --
    same logic, different province->region mapping already applied to
    `region_col` by the caller."""
    all_centroids, all_profile_rows, all_elbow_rows = [], [], []

    for region_val in region_values:
        df_g = df[df[region_col] == region_val].copy()
        if df_g.empty:
            continue
        print(f"  [personas] clustering {split_col_name}={region_val} (n={len(df_g)}, auto-k)...")
        c_df, p_rows, e_rows = _cluster_split(
            df_g, split_col_name, region_val, VARS_FOR_CLUSTERING, KModes, auto_k=True,
        )
        chosen_k = e_rows[0]["chosen_k"] if e_rows else N_CLUSTERS
        print(f"    -> auto-selected k={chosen_k}")
        all_centroids.append(c_df)
        all_profile_rows.extend(p_rows)
        all_elbow_rows.extend(e_rows)

    if all_centroids:
        save(pd.concat(all_centroids, ignore_index=True),
             os.path.join(APP_DATA_DIR, f"personas_centroids_by_{label}.csv"),
             f"persona centroids (by {label})")
        save(pd.DataFrame(all_profile_rows),
             os.path.join(APP_DATA_DIR, f"personas_profile_by_{label}.csv"),
             f"persona profiles (by {label})")
        save(pd.DataFrame(all_elbow_rows),
             os.path.join(APP_DATA_DIR, f"personas_elbow_by_{label}.csv"),
             f"persona elbow data (by {label}, incl. auto-selected k)")


def run_culture_clusters(df, KModes):
    """
    Standalone analysis (2026-09-02, requested to test "do respondents
    geographically cluster based on culture?"): cluster on religion, life
    goals, and top driver/barrier ONLY -- region is deliberately excluded
    from the clustering inputs (see CULTURE_CLUSTERING_VARS's comment in
    config.py) -- then report each resulting cluster's region mix as a
    profile variable, so geographic concentration (or the lack of it) shows
    up as a genuine finding rather than being assumed by the clustering
    itself.
    """
    missing = [v for v in CULTURE_CLUSTERING_VARS if v not in df.columns]
    if missing:
        print(f"  [personas] WARNING: culture-clustering variables missing: {missing} — skipping.")
        return

    print(f"  [personas] running standalone culture clustering (n={len(df)}, auto-k)...")
    c_df, p_rows, e_rows = _cluster_split(
        df, "analysis", "culture", CULTURE_CLUSTERING_VARS, KModes, auto_k=True,
    )
    chosen_k = e_rows[0]["chosen_k"] if e_rows else N_CLUSTERS
    print(f"    -> auto-selected k={chosen_k}")

    save(c_df, os.path.join(APP_DATA_DIR, "culture_clusters_centroids.csv"),
         "culture cluster centroids")
    save(pd.DataFrame(p_rows), os.path.join(APP_DATA_DIR, "culture_clusters_profile.csv"),
         "culture cluster profiles (includes region mix per cluster)")
    save(pd.DataFrame(e_rows), os.path.join(APP_DATA_DIR, "culture_clusters_elbow.csv"),
         "culture cluster elbow data")


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [personas] running...")

    try:
        from kmodes.kmodes import KModes
    except ImportError:
        print("  [personas] WARNING: kmodes not installed. Run: pip install kmodes")
        return

    df = add_engineered_columns(df.copy())

    # ── Overall clustering (kept for backwards compatibility) ─────────────────
    cluster_df = df[VARS_FOR_CLUSTERING].copy().fillna("Unknown")
    kmode = KModes(n_clusters=N_CLUSTERS, init="Cao", n_init=5,
                   verbose=0, random_state=42)
    clusters = kmode.fit_predict(cluster_df.to_numpy())

    centroids = pd.DataFrame(kmode.cluster_centroids_, columns=VARS_FOR_CLUSTERING)
    centroids.index.name = "persona"
    counts = pd.Series(clusters).value_counts().sort_index()
    centroids["count"] = counts.values
    if WEIGHT_COL in df.columns:
        df_c = df.copy()
        df_c["_cluster"] = clusters
        weighted_counts = df_c.groupby("_cluster")[WEIGHT_COL].sum()
        centroids["weighted_count"] = weighted_counts.values
    save(centroids, os.path.join(APP_DATA_DIR, "personas_centroids.csv"),
         "persona centroids (overall)")

    df_c = df.copy()
    df_c["_cluster"] = clusters
    profile_rows = []
    for persona_id in range(N_CLUSTERS):
        sub = df_c[df_c["_cluster"] == persona_id]
        n = len(sub)
        w = sub[WEIGHT_COL].sum() if WEIGHT_COL in sub.columns else n
        for col in VARS_FOR_CLUSTERING:
            if col == "age" and pd.api.types.is_numeric_dtype(sub[col]):
                profile_rows.append({"persona": persona_id, "variable": col,
                                     "value": "mean", "proportion": sub[col].mean()})
            else:
                vc = sub[col].value_counts(normalize=True).head(3)
                for val, prop in vc.items():
                    profile_rows.append({"persona": persona_id, "variable": col,
                                         "value": str(val), "proportion": prop})
        for col in ["gender", "age_group", "use"]:
            if col in sub.columns and col not in VARS_FOR_CLUSTERING:
                vc = sub[col].value_counts(normalize=True)
                for val, prop in vc.items():
                    profile_rows.append({"persona": persona_id, "variable": col,
                                         "value": str(val), "proportion": prop})
        profile_rows.append({"persona": persona_id, "variable": "_count",
                              "value": "n", "proportion": n})
        profile_rows.append({"persona": persona_id, "variable": "_count",
                              "value": "weighted_n", "proportion": w})
    save(pd.DataFrame(profile_rows),
         os.path.join(APP_DATA_DIR, "personas_profile.csv"), "persona profiles (overall)")

    # ── Gender-split clustering ───────────────────────────────────────────────
    all_centroids = []
    all_profile_rows = []
    all_elbow_rows = []

    gender_col = "gender"
    if gender_col not in df.columns:
        print("  [personas] WARNING: 'gender' column not found — skipping gender split.")
    else:
        for short_key, gender_val in GENDER_LABELS.items():
            df_g = df[df[gender_col] == gender_val].copy()
            if df_g.empty:
                print(f"  [personas] WARNING: no rows for gender='{gender_val}', skipping.")
                continue
            print(f"  [personas] clustering {short_key} (n={len(df_g)})...")
            c_df, p_rows, e_rows = _cluster_gender(df_g, gender_val, KModes)
            all_centroids.append(c_df)
            all_profile_rows.extend(p_rows)
            all_elbow_rows.extend(e_rows)

    if all_centroids:
        save(pd.concat(all_centroids, ignore_index=True),
             os.path.join(APP_DATA_DIR, "personas_centroids_by_gender.csv"),
             "persona centroids (by gender)")
        save(pd.DataFrame(all_profile_rows),
             os.path.join(APP_DATA_DIR, "personas_profile_by_gender.csv"),
             "persona profiles (by gender)")
        save(pd.DataFrame(all_elbow_rows),
             os.path.join(APP_DATA_DIR, "personas_elbow.csv"),
             "persona elbow data")

    # ── Region-split clustering (4-way: North-East/North-West/South-South/
    #    Mid-South) ─────────────────────────────────────────────────────────────
    # k is now auto-selected per region from its own elbow curve (2026-09-02),
    # not fixed at N_CLUSTERS -- regions differ enough in size/composition
    # that one fixed k for all of them was never well-justified.
    if "region" not in df.columns:
        print("  [personas] WARNING: 'region' column not found — skipping region split.")
    else:
        unmapped = df.loc[df["region"].isna() & df["province"].notna(), "province"].unique().tolist()
        if unmapped:
            print(f"  [personas] WARNING: provinces with no region mapping (excluded): {unmapped}")
        region_values = [r for r in df["region"].dropna().unique()]
        _run_region_split(df, "region", region_values, "region", "region", KModes)

    # ── Region-split clustering (2-way: North/South, 2026-09-02) ──────────────
    if "region" not in df.columns:
        print("  [personas] WARNING: 'region' column not found — skipping North/South split.")
    else:
        df["region_ns"] = df["region"].map(REGION_TO_NORTH_SOUTH)
        unmapped_ns = df.loc[df["region_ns"].isna() & df["region"].notna(), "region"].unique().tolist()
        if unmapped_ns:
            print(f"  [personas] WARNING: regions with no North/South mapping (excluded): {unmapped_ns}")
        ns_values = [r for r in df["region_ns"].dropna().unique()]
        _run_region_split(df, "region_ns", ns_values, "region_ns", "region_ns", KModes)

    # ── Standalone culture clustering (2026-09-02) ─────────────────────────────
    run_culture_clusters(df, KModes)

    print(f"  [personas] done. {N_CLUSTERS} clusters overall/by gender; "
          f"region splits use auto-selected k per group.")
