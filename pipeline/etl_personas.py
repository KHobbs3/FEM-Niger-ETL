"""
ETL: Personas page
Runs k-modes clustering on the PII dataset and exports only the
cluster centroids + counts — no individual-level data in output.

Number of clusters (k) per group is chosen automatically from that group's
own elbow curve (kneedle-style "point of max curvature" — see
_auto_select_k below) rather than a fixed count, so a group whose data
genuinely supports more or fewer distinct personas isn't forced into the
same k as every other group. K_MIN/K_MAX below bound the search.

Output files (overall):
  personas_centroids.csv   — cluster centroid for each variable + count
  personas_profile.csv     — per-persona summary stats (demographics)
  personas_elbow_overall.csv — within-cluster cost for k=1..6, overall

Output files (by gender):
  personas_centroids_by_gender.csv — centroids per gender × cluster
  personas_profile_by_gender.csv   — profile per gender × cluster
  personas_elbow.csv               — within-cluster cost for k=1..6 per
                                      gender, with the auto-selected k flagged

Output files (by FP-use status — same "user" definition as elsewhere in this
pipeline, config.USER_GROUPS/NONUSER_GROUPS collapsing the 4-way `use` column
into a binary Using FP / Not using FP):
  personas_centroids_by_fp_use.csv — centroids per fp_use group × cluster
  personas_profile_by_fp_use.csv   — profile per fp_use group × cluster
  personas_elbow_by_fp_use.csv     — within-cluster cost for k=1..6 per
                                      fp_use group, with the auto-selected k
                                      flagged
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import (
    WEIGHT_COL, APP_DATA_DIR, VARS_FOR_CLUSTERING, USER_GROUPS, NONUSER_GROUPS,
)
from pipeline.utils import save

K_MIN = 2  # never auto-select k=1 (a single "persona" isn't a segmentation)
K_MAX = 6  # maximum k to test for elbow plot / auto-detection

# When clustering within a single gender, drop gender from features (it's
# constant within the group). Full VARS_FOR_CLUSTERING list is used for the
# fp_use split instead, since gender still varies within each fp_use group.
VARS_FOR_CLUSTERING_GENDER = [v for v in VARS_FOR_CLUSTERING if v != "gender"]

GENDER_LABELS = {
    "female": "Mace / Femme",
    "male":   "Namiji / Homme",
}

FP_USE_LABELS = {
    "user":     "Using FP",
    "nonuser":  "Not using FP",
}


def _auto_select_k(ks, costs, k_min=K_MIN):
    """
    Kneedle-style elbow detection, dependency-free: normalize both axes to
    [0, 1], draw the straight line from the first to the last (k, cost)
    point, and pick the k whose point sits furthest below that line (i.e.
    the point of maximum curvature on a convex decreasing cost curve — the
    "elbow"). Falls back to k_min if the curve is flat (no discernible
    elbow) or too short to have one.

    ks, costs: same-length sequences, k=1..K_MAX and each k's within-cluster
    cost (k-modes' cost_). Search is restricted to k >= k_min so a trivial
    k=1 "elbow" (always the steepest single drop) is never selected.
    """
    ks = np.asarray(ks, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if len(ks) < 3:
        return int(ks[np.searchsorted(ks, k_min)]) if k_min in ks else int(ks[0])

    k_range = ks.max() - ks.min()
    cost_range = costs.max() - costs.min()
    k_norm = (ks - ks.min()) / k_range if k_range else np.zeros_like(ks)
    cost_norm = (costs - costs.min()) / cost_range if cost_range else np.zeros_like(costs)

    x1, y1 = k_norm[0], cost_norm[0]
    x2, y2 = k_norm[-1], cost_norm[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0:
        return int(k_min)
    distances = np.abs((y2 - y1) * k_norm - (x2 - x1) * cost_norm + x2 * y1 - y2 * x1) / denom

    candidate_mask = ks >= k_min
    if not candidate_mask.any():
        candidate_mask[:] = True
    idx = np.where(candidate_mask)[0]
    best = idx[np.argmax(distances[idx])]
    return int(ks[best])


def _cluster_group(df_group, group_label, group_col_name, cluster_vars, KModes):
    """
    Run k-modes for k=1..K_MAX to collect elbow data, auto-select k from
    that curve (_auto_select_k), then fit the final model with the selected
    k on df_group using cluster_vars as features. Every output row is
    tagged group_col_name -> group_label (e.g. "gender" -> "Mace / Femme",
    or "fp_use" -> "Using FP") so callers can concatenate results across
    groups and filter by the tag column. Pass group_col_name=None (no split)
    for a single ungrouped run — no tag column is added in that case.
    Returns (centroids_df, profile_rows, elbow_rows).
    """
    cluster_df = df_group[cluster_vars].copy().fillna("Unknown")
    X = cluster_df.to_numpy()

    # ── Elbow data + auto-selected k ─────────────────────────────────────────
    elbow_rows = []
    ks, costs = [], []
    for k in range(1, K_MAX + 1):
        km = KModes(n_clusters=k, init="Cao" if k > 1 else "random",
                    n_init=3, verbose=0, random_state=42)
        km.fit(X)
        ks.append(k)
        costs.append(km.cost_)

    n_clusters = _auto_select_k(ks, costs)
    for k, cost in zip(ks, costs):
        row = {"k": k, "cost": cost, "selected": k == n_clusters}
        if group_col_name:
            row = {group_col_name: group_label, **row}
        elbow_rows.append(row)
    tag = f"{group_col_name}={group_label}" if group_col_name else "overall"
    print(f"    -> auto-selected k={n_clusters} for {tag}")

    # ── Final model ───────────────────────────────────────────────────────────
    km_final = KModes(n_clusters=n_clusters, init="Cao", n_init=5,
                      verbose=0, random_state=42)
    clusters = km_final.fit_predict(X)

    # ── Centroids ─────────────────────────────────────────────────────────────
    centroids = pd.DataFrame(km_final.cluster_centroids_, columns=cluster_vars)
    if group_col_name:
        centroids.insert(0, group_col_name, group_label)
        centroids.insert(1, "persona", range(n_clusters))
    else:
        centroids.insert(0, "persona", range(n_clusters))
    centroids["k_selected"] = n_clusters

    counts = pd.Series(clusters).value_counts().sort_index()
    centroids["count"] = counts.values

    if WEIGHT_COL in df_group.columns:
        df_c = df_group.copy()
        df_c["_cluster"] = clusters
        weighted_counts = df_c.groupby("_cluster")[WEIGHT_COL].sum()
        centroids["weighted_count"] = weighted_counts.values

    # ── Per-persona profile ───────────────────────────────────────────────────
    df_c = df_group.copy()
    df_c["_cluster"] = clusters

    def _row(**kwargs):
        base = {group_col_name: group_label} if group_col_name else {}
        return {**base, **kwargs}

    profile_rows = []
    for persona_id in range(n_clusters):
        sub = df_c[df_c["_cluster"] == persona_id]
        n = len(sub)
        w = sub[WEIGHT_COL].sum() if WEIGHT_COL in sub.columns else n

        for col in cluster_vars:
            if col == "age" and pd.api.types.is_numeric_dtype(sub[col]):
                profile_rows.append(_row(
                    persona=persona_id, variable=col, value="mean",
                    proportion=sub[col].mean(),
                ))
            else:
                vc = sub[col].value_counts(normalize=True).head(3)
                for val, prop in vc.items():
                    profile_rows.append(_row(
                        persona=persona_id, variable=col, value=str(val),
                        proportion=prop,
                    ))

        # Additional demographic breakdown
        for col in ["gender", "age_group", "use"]:
            if col in sub.columns and col not in cluster_vars:
                vc = sub[col].value_counts(normalize=True)
                for val, prop in vc.items():
                    profile_rows.append(_row(
                        persona=persona_id, variable=col, value=str(val),
                        proportion=prop,
                    ))

        # "Other" life-goal free text (2026-09-04): "life_goals" clustering on
        # its raw pipe-joined label text can land "Other" as a persona's modal
        # value, which on its own tells you nothing about what respondents
        # actually meant. Surface the top verbatim "Please specify" answers
        # from anyone in this cluster who picked "Other", so the persona table
        # isn't a dead end. Proportions are of respondents-who-specified, not
        # of the whole cluster (n_specified below is the denominator).
        if "life_goals_other_specify" in sub.columns:
            specify = sub["life_goals_other_specify"].dropna().astype(str).str.strip()
            specify = specify[specify != ""]
            if not specify.empty:
                vc = specify.value_counts(normalize=True).head(5)
                for val, prop in vc.items():
                    profile_rows.append(_row(
                        persona=persona_id, variable="life_goals_other_specify",
                        value=val, proportion=prop,
                    ))
                profile_rows.append(_row(
                    persona=persona_id, variable="_count",
                    value="n_life_goals_specified", proportion=len(specify),
                ))

        profile_rows.append(_row(
            persona=persona_id, variable="_count", value="n", proportion=n,
        ))
        profile_rows.append(_row(
            persona=persona_id, variable="_count", value="weighted_n", proportion=w,
        ))

    return centroids, profile_rows, elbow_rows


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [personas] running...")

    try:
        from kmodes.kmodes import KModes
    except ImportError:
        print("  [personas] WARNING: kmodes not installed. Run: pip install kmodes")
        return

    # ── Overall clustering (kept for backwards compatibility) ─────────────────
    # Auto-selects k from the overall sample's own elbow curve, same as every
    # split below (group_col_name=None -> no group-tag column in the output,
    # preserving personas_centroids.csv's original schema).
    print("  [personas] clustering overall...")
    centroids, profile_rows, overall_elbow_rows = _cluster_group(
        df, None, None, VARS_FOR_CLUSTERING, KModes)
    centroids = centroids.set_index("persona")
    save(centroids, os.path.join(APP_DATA_DIR, "personas_centroids.csv"),
         "persona centroids (overall)")
    save(pd.DataFrame(profile_rows),
         os.path.join(APP_DATA_DIR, "personas_profile.csv"), "persona profiles (overall)")
    save(pd.DataFrame(overall_elbow_rows),
         os.path.join(APP_DATA_DIR, "personas_elbow_overall.csv"), "persona elbow data (overall)")

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
            c_df, p_rows, e_rows = _cluster_group(
                df_g, gender_val, "gender", VARS_FOR_CLUSTERING_GENDER, KModes)
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

    # ── FP-use-split clustering ───────────────────────────────────────────────
    # Same binary Using FP / Not using FP definition used throughout this
    # pipeline (config.USER_GROUPS / NONUSER_GROUPS collapsing the 4-way
    # `use` column: user+past_user -> Using FP, nonuser+future_user -> Not
    # using FP). Clusters on the full VARS_FOR_CLUSTERING (gender included —
    # it still varies within an fp_use group, unlike within a gender group).
    fp_centroids, fp_profile_rows, fp_elbow_rows = [], [], []

    use_col = "use"
    if use_col not in df.columns:
        print("  [personas] WARNING: 'use' column not found — skipping fp_use split.")
    else:
        fp_use_group = df[use_col].apply(
            lambda v: "Using FP" if v in USER_GROUPS
            else ("Not using FP" if v in NONUSER_GROUPS else None)
        )
        for fp_label in ["Using FP", "Not using FP"]:
            df_g = df[fp_use_group == fp_label].copy()
            if df_g.empty:
                print(f"  [personas] WARNING: no rows for fp_use='{fp_label}', skipping.")
                continue
            print(f"  [personas] clustering fp_use={fp_label} (n={len(df_g)})...")
            c_df, p_rows, e_rows = _cluster_group(
                df_g, fp_label, "fp_use", VARS_FOR_CLUSTERING, KModes)
            fp_centroids.append(c_df)
            fp_profile_rows.extend(p_rows)
            fp_elbow_rows.extend(e_rows)

    if fp_centroids:
        save(pd.concat(fp_centroids, ignore_index=True),
             os.path.join(APP_DATA_DIR, "personas_centroids_by_fp_use.csv"),
             "persona centroids (by fp_use)")
        save(pd.DataFrame(fp_profile_rows),
             os.path.join(APP_DATA_DIR, "personas_profile_by_fp_use.csv"),
             "persona profiles (by fp_use)")
        save(pd.DataFrame(fp_elbow_rows),
             os.path.join(APP_DATA_DIR, "personas_elbow_by_fp_use.csv"),
             "persona elbow data (by fp_use)")

    print("  [personas] done. k auto-selected per group (elbow/kneedle) -- see "
          "personas_elbow*.csv 'selected' column for each group's chosen k.")
