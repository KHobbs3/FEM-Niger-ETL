"""
ETL: Statement Agreement page
Produces pre-aggregated CSVs — no PII in outputs.

Output files:
  statements_heatmap.csv  — weighted agreement score per statement x split group
                            (long format: statement, split, group, weighted_agreement,
                             agree_n, agree_pct, agree_weighted_n, disagree_n, disagree_pct,
                             disagree_weighted_n, total_n, total_weighted_n)
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import WEIGHT_COL, SPLIT_COLS, APP_DATA_DIR
from pipeline.utils import save, load_raw


def run(df, statement_labels_path=None):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [statements] running...")

    # Identify statement columns
    statement_columns = [c for c in df.columns if c.startswith("statement_")]
    if not statement_columns:
        print("  [statements] WARNING: no statement_ columns found. Skipping.")
        return

    descriptive_cols = ["age_group", "gender", "use", WEIGHT_COL]
    df_s = df[statement_columns + descriptive_cols].copy()

    # Collapse past_user and future_user into nonuser
    df_s["use"] = df_s["use"].replace({"past_user": "nonuser", "future_user": "nonuser"})

    # Melt to long format
    df_melted = df_s.melt(
        id_vars=descriptive_cols,
        value_vars=statement_columns,
        var_name="statement",
        value_name="response",
    )

    # Quantify agreement: Agree=1, Disagree=-1, else=0
    df_melted["agreement"] = (
        df_melted["response"]
        .fillna("")
        .map(lambda x: 1 if "Agree" in str(x) else (-1 if "Disagree" in str(x) else 0))
    )
    
    # Flag agrees and disagrees separately
    df_melted["is_agree"] = df_melted["response"].fillna("").map(lambda x: "Agree" in str(x))
    df_melted["is_disagree"] = df_melted["response"].fillna("").map(lambda x: "Disagree" in str(x))

    # Load statement labels if available
    label_map = {}
    if statement_labels_path and os.path.exists(statement_labels_path):
        ldf = pd.read_csv(statement_labels_path, encoding="ISO-8859-1")
        ldf.columns = ldf.columns.str.strip()
        ldf.dropna(how="all", inplace=True)
        if "statement" in ldf.columns and "label_en" in ldf.columns:
            label_map = ldf.set_index("statement")["label_en"].to_dict()

    df_melted["label"] = df_melted["statement"].map(label_map).fillna(df_melted["statement"])

    # ── Aggregate: weighted agreement, n_agree, n_disagree, weighted_n_agree, weighted_n_disagree ──
    def _agg(g):
        """Compute weighted agreement, agree/disagree counts."""
        valid = g.dropna(subset=[WEIGHT_COL])
        
        # Respondents who agreed
        agree = valid[valid["is_agree"] == True]
        # Respondents who disagreed
        disagree = valid[valid["is_disagree"] == True]
        
        n_total    = len(valid)
        n_agree    = len(agree)
        n_disagree = len(disagree)
        # Denominator for % matches the notebook: only respondents who gave an
        # opinion (agreed or disagreed); don't-know / prefer-not-to-say excluded.
        n_opinioned = n_agree + n_disagree
        return pd.Series({
            "weighted_agreement":  (valid["agreement"] * valid[WEIGHT_COL]).sum() / valid[WEIGHT_COL].sum() if n_total > 0 else 0,
            "agree_n":             n_agree,
            "agree_pct":           n_agree / n_opinioned if n_opinioned > 0 else 0,
            "agree_weighted_n":    agree[WEIGHT_COL].sum() if n_agree > 0 else 0,
            "disagree_n":          n_disagree,
            "disagree_pct":        n_disagree / n_opinioned if n_opinioned > 0 else 0,
            "disagree_weighted_n": disagree[WEIGHT_COL].sum() if n_disagree > 0 else 0,
            "total_n":             n_total,
            "total_weighted_n":    valid[WEIGHT_COL].sum(),
        })

    rows = []

    # Overall (no split)
    overall = df_melted.groupby("label").apply(_agg).reset_index()
    overall["split"] = "none"
    overall["group"] = "all"
    rows.append(overall[["label", "split", "group", "weighted_agreement",
                          "agree_n", "agree_pct", "agree_weighted_n",
                          "disagree_n", "disagree_pct", "disagree_weighted_n",
                          "total_n", "total_weighted_n"]])

    # By each split column
    for split_col in SPLIT_COLS:
        agg = df_melted.groupby(["label", split_col]).apply(_agg).reset_index()
        agg.columns = ["label", "group", "weighted_agreement",
                       "agree_n", "agree_pct", "agree_weighted_n",
                       "disagree_n", "disagree_pct", "disagree_weighted_n",
                       "total_n", "total_weighted_n"]
        agg["split"] = split_col
        rows.append(agg[["label", "split", "group", "weighted_agreement",
                          "agree_n", "agree_pct", "agree_weighted_n",
                          "disagree_n", "disagree_pct", "disagree_weighted_n",
                          "total_n", "total_weighted_n"]])

    result = pd.concat(rows, ignore_index=True)
    save(result, os.path.join(APP_DATA_DIR, "statements_heatmap.csv"),
         "statement heatmap data")

    print("  [statements] done.")