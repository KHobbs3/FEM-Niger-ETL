"""
etl_radio.py — produces the radio summary CSV consumed by page_radio.py

Run from the niger_app directory:
    python pipeline/run_pipeline.py --pages radio

Reads:   FEM_MAPPED_DATA env var, or default path below (raw respondent-level CSV)
Writes:  data/Niger_radio_table_YY_MM_DD.csv (station-level)
         data/Niger_radio_table_by_state_YY_MM_DD.csv (state-level)

Output format
──────────��──
Rows  = one per question/metric (radio listening questions)
Cols  = one per station/state, with three variants:
            <Station_ID>        prevalence %  (parsed by parse_radio_cell)
            <Station_ID>_n      raw n
            <Station_ID>_wn     weighted n
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
from datetime import datetime
from pathlib import Path
import warnings
from pipeline.config import (
    WEIGHT_COL, SPLIT_COLS, APP_DATA_DIR, DIR_MAPPED_DATA, station_path
)


warnings.filterwarnings("ignore", category=FutureWarning)

# ── Station mapping ───────────────────────────────────────────────────────────

STATION_STATE = {
    "0227200006_3_mouriyarkarkara_1":    "Tahoua",
    "0227200215_7_bidizguiri_1":         "Tahoua",
    "0227200415_11_takarkara_1":         "Tahoua",
    "0227200517_13_zoumuntchi__1":       "Maradi",
    "0227200859_20_alheri_1":            "Maradi",
    "0227200931_21_annour_1":            "Maradi",
    "0227201211_26_kutukum":             "Zinder",
    "0227201414_30_kitari_1":            "Zinder",
    "0227201446_31_tsirkau_1":           "Zinder",
    "0227201641_35_maitama_1":           "Zinder",
    "0227201713_36_alternative_1":       "Zinder",
}

STATE_ORDER = ["Maradi", "Tahoua", "Zinder"]

# Radio listening questions to analyze
RADIO_QUESTIONS = [
    'station_most_listened',
    'station_listened_yesterday',
    'station_past_7_days',
    'media_type',
    'radio_consumption_method',
    'favourite_radio_format',
    'radio_ad_saturation',
    'radio_language',
    'radio_when',
    'radio_what',
    'radio_trust',
    'radio_influence',
]


# ── Spatial station labeling ─────────────────────────────────────────────────

def add_station_labels(data: pd.DataFrame, station_path: str, max_distance: int = 1000) -> pd.DataFrame:
    """
    Add station_label column to data using spatial join and nearest-neighbor snapping.
    
    Args:
        data: DataFrame with GPS coordinates (gps-Longitude, gps-Latitude)
        station_path: Glob pattern to station .gpkg files
        max_distance: Max distance (meters) for snapping to nearest station
    
    Returns:
        DataFrame with added 'station_label' column
    """
    print("  [spatial] Adding station labels via spatial join...")
    
    if 'gps-Longitude' not in data.columns or 'gps-Latitude' not in data.columns:
        print("  Warning: GPS columns not found. Skipping spatial join.")
        return data
    
    # Create GeoDataFrame
    data_gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['gps-Longitude'], data['gps-Latitude']),
        crs='epsg:4326'
    )
    
    # Initialize station label column
    if 'station_label' not in data.columns:
        data['station_label'] = None
    
    # Convert to projected CRS for accurate distance calculations
    data_gdf_projected = data_gdf.to_crs('epsg:32632')
    
    all_stations = []
    
    # Iterate through each station file
    for station_file in glob.glob(station_path + "*.gpkg"):
        station_gdf = gpd.read_file(station_file)
        
        if station_gdf.crs != data_gdf.crs:
            station_gdf = station_gdf.to_crs(data_gdf.crs)
        
        station_gdf_projected = station_gdf.to_crs('epsg:32632')
        station_name = station_file.split('/')[-1].replace('.gpkg', '')
        station_gdf_projected['station_name'] = station_name
        
        all_stations.append(station_gdf_projected)
        
        # Perform spatial join (within polygons)
        points_in_station = gpd.sjoin(
            data_gdf_projected,
            station_gdf_projected,
            how='inner',
            predicate='within'
        )
        
        if len(points_in_station) > 0:
            data.loc[points_in_station.index, 'station_label'] = station_name
        print(f"    Assigned {len(points_in_station)} points to {station_name}")
    
    # Snap unmatched points to nearest station
    unmatched_indices = data[data['station_label'].isna()].index
    unmatched_points = data_gdf_projected.loc[unmatched_indices]
    
    if len(unmatched_points) > 0 and len(all_stations) > 0:
        print(f"    Snapping {len(unmatched_points)} unmatched points (within {max_distance}m)...")
        all_stations_gdf = pd.concat(all_stations, ignore_index=True)
        all_stations_gdf = all_stations_gdf.reset_index(drop=True)
        
        snapped_count = 0
        for idx in unmatched_indices:
            point_geom = data_gdf_projected.loc[idx, 'geometry']
            all_stations_gdf['distance'] = all_stations_gdf.geometry.distance(point_geom)
            min_dist_position = all_stations_gdf['distance'].argmin()
            nearest_station = all_stations_gdf.iloc[min_dist_position]
            
            if nearest_station['distance'] <= max_distance:
                data.loc[idx, 'station_label'] = nearest_station['station_name']
                snapped_count += 1
        
        print(f"    Snapped {snapped_count} points; {len(unmatched_points) - snapped_count} beyond threshold")
    
    print(f"    Total labeled: {data['station_label'].notna().sum()} / {len(data)}")
    
    # Drop respondents without station labels
    data = data.dropna(subset=['station_label']).copy()
    return data


# ── Cell formatters ───────────────────────────────────────────────────────────

def _fmt_prevalence(series: pd.Series) -> str:
    """Format {label: pct} Series as 'label\\nvalue\\n...' (parse_radio_cell compat)."""
    lines = []
    for label, val in series.items():
        if pd.notna(val) and val != "":
            lines += [str(label), f"{val:.1f}"]
    return "\n".join(lines)


def _fmt_counts(series: pd.Series, weighted: bool = False) -> str:
    """Format {label: count} Series as 'label\\nvalue\\n...' (parse_radio_cell compat)."""
    lines = []
    for label, val in series.items():
        if pd.notna(val) and val != "":
            try:
                val_numeric = float(val)
                formatted = f"{val_numeric:.1f}" if weighted else f"{int(round(val_numeric))}"
                lines += [str(label), formatted]
            except (ValueError, TypeError) as e:
                print(f"  ⚠️  Could not format value: {label}={val} ({type(val).__name__})")
                continue
    return "\n".join(lines)


# ── Core computation ──────────────────────────────────────────────────────────

def compute_question_stats(data: pd.DataFrame, station_name: str, question_col: str,
                           weight_col: str = WEIGHT_COL) -> tuple:
    """
    For one station, compute distribution of answers to a question.
    
    Used for: station_most_listened, media_type, favourite_radio_format, etc.
    Handles multi-select (pipe-delimited) answers.
    
    Returns (prev_str, n_str, wn_str).
    """
    valid = data[[question_col, weight_col]].dropna(subset=[question_col]).copy()
    
    if len(valid) == 0:
        return "", "", ""
    
    # Convert weight to numeric
    valid[weight_col] = pd.to_numeric(valid[weight_col], errors='coerce')
    valid = valid.dropna(subset=[weight_col])
    
    if len(valid) == 0:
        return "", "", ""
    
    # Handle multi-select answers (pipe-delimited)
    valid['answer'] = valid[question_col].astype(str).str.split('|')
    valid = valid.explode('answer', ignore_index=True)
    valid['answer'] = valid['answer'].str.strip()
    
    # Calculate weighted prevalence by answer
    def calculate_weighted_prevalence(group):
        """Calculate percentage of total weight for this answer"""
        total_weight_all = valid[weight_col].sum()
        group_weight = group[weight_col].sum()
        if total_weight_all == 0:
            return np.nan
        return (group_weight / total_weight_all) * 100.0
    
    # Prevalence: weighted % of total respondents
    prev = valid.groupby('answer').apply(calculate_weighted_prevalence)
    
    # Raw count: number of responses per answer
    n = valid.groupby('answer').size()
    
    # Weighted count: sum of weights per answer
    wn = valid.groupby('answer')[weight_col].sum()
    
    # Sort by prevalence (descending)
    prev = prev.sort_values(ascending=False)
    n = n.reindex(prev.index)
    wn = wn.reindex(prev.index)
    
    return _fmt_prevalence(prev), _fmt_counts(n), _fmt_counts(wn, weighted=True)


def compute_state_stats(data: pd.DataFrame, state_name: str, question_col: str,
                        weight_col: str = WEIGHT_COL) -> tuple:
    """
    For one state, compute distribution of answers to a question.
    Aggregates across all stations in that state.
    
    Returns (prev_str, n_str, wn_str).
    """
    valid = data[[question_col, weight_col]].dropna(subset=[question_col]).copy()
    
    if len(valid) == 0:
        return "", "", ""
    
    # Convert weight to numeric
    valid[weight_col] = pd.to_numeric(valid[weight_col], errors='coerce')
    valid = valid.dropna(subset=[weight_col])
    
    if len(valid) == 0:
        return "", "", ""
    
    # Handle multi-select answers (pipe-delimited)
    valid['answer'] = valid[question_col].astype(str).str.split('|')
    valid = valid.explode('answer', ignore_index=True)
    valid['answer'] = valid['answer'].str.strip()
    
    # Calculate weighted prevalence by answer
    def calculate_weighted_prevalence(group):
        """Calculate percentage of total weight for this answer"""
        total_weight_all = valid[weight_col].sum()
        group_weight = group[weight_col].sum()
        if total_weight_all == 0:
            return np.nan
        return (group_weight / total_weight_all) * 100.0
    
    # Prevalence: weighted % of total respondents
    prev = valid.groupby('answer').apply(calculate_weighted_prevalence)
    
    # Raw count: number of responses per answer
    n = valid.groupby('answer').size()
    
    # Weighted count: sum of weights per answer
    wn = valid.groupby('answer')[weight_col].sum()
    
    # Sort by prevalence (descending)
    prev = prev.sort_values(ascending=False)
    n = n.reindex(prev.index)
    wn = wn.reindex(prev.index)
    
    return _fmt_prevalence(prev), _fmt_counts(n), _fmt_counts(wn, weighted=True)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(merged: pd.DataFrame, station_names: list) -> bool:
    """Validate output table structure and content."""
    errors = []
    
    print(f"  [validation] Output shape: {merged.shape}")
    assert len(merged) > 0, "Empty output"
    
    # Expected columns present
    for name in station_names:
        for suffix in ("", "_n", "_wn"):
            col = f"{name}{suffix}"
            if col not in merged.columns:
                errors.append(f"Missing column: {col}")
    
    # No completely empty cells
    empty_cells = 0
    for col in merged.columns:
        for val in merged[col]:
            if pd.isna(val) or val == "":
                empty_cells += 1
    
    if empty_cells > 0:
        print(f"  [validation] Warning: {empty_cells} empty cells")
    
    if errors:
        print(f"\n{'='*60}")
        print(f"{len(errors)} VALIDATION ERROR(S):")
        for e in errors:
            print(f"  {e}")
        print('='*60)
        return False
    else:
        print(f"  [validation] Passed")
        return True


# ── Table builders ────────────────────────────────────────────────────────────

def build_radio_table(data: pd.DataFrame, weight_col: str = WEIGHT_COL) -> tuple:
    """Build the full radio summary table with stations as columns."""
    
    # Get unique stations
    if 'station_label' not in data.columns:
        print("  Warning: station_label column not found in data")
        return pd.DataFrame(), []
    
    stations = sorted([s for s in data['station_label'].dropna().unique() if s])
    
    if not stations:
        print("  Warning: No stations found in data")
        return pd.DataFrame(), []
    
    print(f"  [radio] Building station-level table for {len(stations)} stations...")
    
    rows = []
    
    # Radio listening questions
    print("  Adding radio listening questions...")
    for question_col in RADIO_QUESTIONS:
        if question_col not in data.columns:
            print(f"    Skipping question '{question_col}' — column not in data")
            continue

        question_fmt = question_col.replace('_', ' ').title()
        row = {"Question": question_fmt}
        
        # For each station geographic area
        for station in stations:
            station_filter = data[data['station_label'] == station]
            
            try:
                prev_str, n_str, wn_str = compute_question_stats(
                    station_filter, station, question_col, weight_col
                )
            except Exception as exc:
                print(f"    Error computing {station} / {question_col}: {exc}")
                import traceback
                traceback.print_exc()
                prev_str = n_str = wn_str = ""
            
            row[station]        = prev_str
            row[f"{station}_n"]  = n_str
            row[f"{station}_wn"] = wn_str
        
        rows.append(row)
    
    df_out = pd.DataFrame(rows).set_index("Question")
    
    # Add state columns for each station
    state_columns = {}
    for station in stations:
        state = STATION_STATE.get(station, "Unknown")
        state_columns[f"{station}_state"] = state
    
    # Insert state columns right after each station's data
    final_cols = []
    for station in stations:
        final_cols.append(station)
        final_cols.append(f"{station}_n")
        final_cols.append(f"{station}_wn")
        final_cols.append(f"{station}_state")
    
    # Reorder columns and add state data
    for state_col, state_val in state_columns.items():
        df_out[state_col] = state_val
    
    df_out = df_out[final_cols]
    
    return df_out, stations


def build_radio_table_by_state(data: pd.DataFrame, weight_col: str = WEIGHT_COL) -> pd.DataFrame:
    """Build radio summary table with states as columns instead of individual stations."""
    
    print(f"  [radio] Building state-level table...")
    
    rows = []
    
    # Radio listening questions
    print("  Adding radio listening questions (state-level)...")
    for question_col in RADIO_QUESTIONS:
        if question_col not in data.columns:
            print(f"    Skipping question '{question_col}' — column not in data")
            continue

        question_fmt = question_col.replace('_', ' ').title()
        row = {"Question": question_fmt}
        
        # For each state
        for state in STATE_ORDER:
            # Filter to respondents in this state
            state_filter = data[data['station_label'].map(lambda x: STATION_STATE.get(x) == state)]
            
            try:
                prev_str, n_str, wn_str = compute_state_stats(
                    state_filter, state, question_col, weight_col
                )
            except Exception as exc:
                print(f"    Error computing {state} / {question_col}: {exc}")
                import traceback
                traceback.print_exc()
                prev_str = n_str = wn_str = ""
            
            row[state]        = prev_str
            row[f"{state}_n"]  = n_str
            row[f"{state}_wn"] = wn_str
        
        rows.append(row)
    
    df_out = pd.DataFrame(rows).set_index("Question")
    return df_out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(df, station_path: str = None):
    """Main ETL pipeline for radio data."""
    print("  [radio] running...")
    
    data = df.copy()
    data.columns = data.columns.str.strip()
    
    # Check and set weight column
    if WEIGHT_COL not in data.columns:
        candidates = [c for c in data.columns if "weight" in c.lower()]
        if candidates:
            print(f"  Using '{candidates[0]}' as weight column")
            data[WEIGHT_COL] = data[candidates[0]]
        else:
            print(f"  No weight column found — using uniform weights")
            data[WEIGHT_COL] = 1.0
    
    # Ensure weight column is numeric
    print(f"  Converting {WEIGHT_COL} to numeric...")
    data[WEIGHT_COL] = pd.to_numeric(data[WEIGHT_COL], errors='coerce')
    
    # Add station labels via spatial join
    if station_path:
        data = add_station_labels(data, station_path)
    else:
        print("  Warning: station_path not provided — skipping spatial join")
    
    # Build station-level table
    df_out, station_names = build_radio_table(data, WEIGHT_COL)
    
    if df_out.empty:
        print("  Error: Output table is empty")
        return
    
    # Build state-level table
    df_out_state = build_radio_table_by_state(data, WEIGHT_COL)
    
    # Validate
    print("\n  [validation] Running...")
    validate(df_out, station_names)
    
    # Ensure all cells are strings
    df_out = df_out.astype(str)
    df_out_state = df_out_state.astype(str)
    
    # Save station-level
    today_date = datetime.today()
    date_string = today_date.strftime("%y_%m_%d")
    out_path = f"{APP_DATA_DIR}/Niger_question_table_by_station_{date_string}.csv"
    df_out.to_csv(out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df_out.shape}  —  {len(station_names)} stations × {len(df_out)} rows")
    
    # Save state-level
    out_path_state = f"{APP_DATA_DIR}/Niger_question_table_by_state_{date_string}.csv"
    df_out_state.to_csv(out_path_state)
    print(f"  Saved: {out_path_state}")
    print(f"  Shape: {df_out_state.shape}  —  3 states × {len(df_out_state)} rows")
    