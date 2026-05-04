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
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import WEIGHT_COL, APP_DATA_DIR, VARS_FOR_CLUSTERING
from pipeline.utils import save

N_CLUSTERS = 3
K_MAX = 6  # maximum k to test for elbow plot

# When clustering within a single gender, drop gender from features
VARS_FOR_CLUSTERING_GENDER = [v for v in VARS_FOR_CLUSTERING if v != "gender"]

GENDER_LABELS = {
    "female": "Mace / Femme",
    "male":   "Namiji / Homme",
}


def _cluster_gender(df_gender, gender_label, KModes):
    """
    Run k-modes for k=1..K_MAX to collect elbow data, then fit the
    final model with N_CLUSTERS.  Returns (centroids_df, profile_rows, elbow_rows).
    """
    cluster_df = df_gender[VARS_FOR_CLUSTERING_GENDER].copy().fillna("Unknown")
    X = cluster_df.to_numpy()

    # ── Elbow data ────────────────────────────────────────────────────────────
    elbow_rows = []
    for k in range(1, K_MAX + 1):
        km = KModes(n_clusters=k, init="Cao" if k > 1 else "random",
                    n_init=3, verbose=0, random_state=42)
        km.fit(X)
        elbow_rows.append({"gender": gender_label, "k": k, "cost": km.cost_})

    # ── Final model ───────────────────────────────────────────────────────────
    km_final = KModes(n_clusters=N_CLUSTERS, init="Cao", n_init=5,
                      verbose=0, random_state=42)
    clusters = km_final.fit_predict(X)

    # ── Centroids ─────────────────────────────────────────────────────────────
    centroids = pd.DataFrame(km_final.cluster_centroids_,
                             columns=VARS_FOR_CLUSTERING_GENDER)
    centroids.insert(0, "gender", gender_label)
    centroids.insert(1, "persona", range(N_CLUSTERS))

    counts = pd.Series(clusters).value_counts().sort_index()
    centroids["count"] = counts.values

    if WEIGHT_COL in df_gender.columns:
        df_c = df_gender.copy()
        df_c["_cluster"] = clusters
        weighted_counts = df_c.groupby("_cluster")[WEIGHT_COL].sum()
        centroids["weighted_count"] = weighted_counts.values

    # ── Per-persona profile ───────────────────────────────────────────────────
    df_c = df_gender.copy()
    df_c["_cluster"] = clusters

    profile_rows = []
    for persona_id in range(N_CLUSTERS):
        sub = df_c[df_c["_cluster"] == persona_id]
        n = len(sub)
        w = sub[WEIGHT_COL].sum() if WEIGHT_COL in sub.columns else n

        for col in VARS_FOR_CLUSTERING_GENDER:
            if col == "age" and pd.api.types.is_numeric_dtype(sub[col]):
                profile_rows.append({
                    "gender": gender_label, "persona": persona_id,
                    "variable": col, "value": "mean",
                    "proportion": sub[col].mean(),
                })
            else:
                vc = sub[col].value_counts(normalize=True).head(3)
                for val, prop in vc.items():
                    profile_rows.append({
                        "gender": gender_label, "persona": persona_id,
                        "variable": col, "value": str(val),
                        "proportion": prop,
                    })

        # Additional demographic breakdown
        for col in ["gender", "age_group", "use"]:
            if col in sub.columns and col not in VARS_FOR_CLUSTERING_GENDER:
                vc = sub[col].value_counts(normalize=True)
                for val, prop in vc.items():
                    profile_rows.append({
                        "gender": gender_label, "persona": persona_id,
                        "variable": col, "value": str(val),
                        "proportion": prop,
                    })

        profile_rows.append({
            "gender": gender_label, "persona": persona_id,
            "variable": "_count", "value": "n", "proportion": n,
        })
        profile_rows.append({
            "gender": gender_label, "persona": persona_id,
            "variable": "_count", "value": "weighted_n", "proportion": w,
        })

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

    print(f"  [personas] done. {N_CLUSTERS} clusters per gender.")
