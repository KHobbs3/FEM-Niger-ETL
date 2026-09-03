"""
etl_radio.py — produces the radio summary CSV consumed by page_radio.py

Run from the benin_app directory:
    python pipeline/run_pipeline.py --pages radio

Reads:   FEM_MAPPED_DATA env var, or default path below (raw respondent-level CSV)
Writes:  data/Benin_radio_table_YY_MM_DD.csv (station-level)
         data/Benin_radio_table_by_state_YY_MM_DD.csv (state-level)

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
import unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
from shapely.geometry import Point
from datetime import datetime
from pathlib import Path
import warnings
from pipeline.config import (
    WEIGHT_COL, SPLIT_COLS, APP_DATA_DIR, DIR_MAPPED_DATA, station_path, COUNTRY
)


warnings.filterwarnings("ignore", category=FutureWarning)

# ── Free-text station-name canonicalization ─────────────────────────────────
# 2026-08-26: station_most_listened / station_listened_yesterday /
# station_past_7_days are `select_multiple station` in the XLSForm, but the
# raw mapped-data column holds literal write-in text, not resolved
# choice-list values -- so the same real station shows up under dozens of
# spelling/case/punctuation variants ("Royal FM", "ROYALE FM", "Radio royal
# FM"...), confirmed by inspecting the pre-aggregated (non-PII) output, not
# raw respondent rows. This section normalizes those mechanically -- accent/
# case fold, strip trailing frequency numbers and parentheticals, strip the
# bare words "radio"/"fm" -- and groups matches into one canonical label per
# cluster, preferring the exact XLSForm 'station' choice-list spelling
# whenever a cluster matches one of the 23 official station names (confirmed
# against [Fon]+Benin+Questionnaire+d'Enquete.xlsx's choices sheet, not
# guessed). Only single-letter/word variants that mechanical normalization
# can't distinguish from a genuinely different station are left alone.
OFFICIAL_STATIONS = [
    "Solidarité FM", "Deeman FM", "Nanto FM", "Maranatha FM", "Radio Tonnasse",
    "Capp FM", "Radio Carrefour", "Ekaaro Ejiire", "Ocean FM", "Gerdes FM",
    "Radio Sedohoun", "Bip Radio", "Tokpa FM", "ADJA OUERE FM", "Kiff FM",
    "Radio Bénin Culture", "Couffo FM", "Radio Tonignon", "Frisson", "ORTB FM",
    "Radio Parakou", "La Voix de la Vallée", "Crystal News",
]

# Two merges confirmed directly by the user (2026-08-26) that mechanical
# normalization can't detect on its own: "Royal"/"Royale" differ by a
# genuine letter, not just case/accent/filler-words, and "ALAFIA FM" is
# short for what's more often written "Radio Bénin Alafia". Everything else
# stays split unless it's an exact mechanical match -- e.g. "Lama" vs "La
# Voix de la Lama" are NOT merged here, since there's no way to confirm from
# spelling alone whether that's the same station or a genuinely different
# one; that's a call only someone who knows the local station landscape can
# make.
STATION_KEY_MERGES = {
    "royale": "royal",
    "alafia": "benin alafia",
}

STATION_TEXT_QUESTIONS = ['station_most_listened', 'station_listened_yesterday', 'station_past_7_days']

_STATION_FREQ_RE = re.compile(r'\b\d{2,3}([.,]\d+)?\s*(fm)?\.?\s*$', re.IGNORECASE)
_STATION_PAREN_RE = re.compile(r'\([^)]*\)')
_STATION_DELIM_RE = re.compile(r'\s+et\s+|,|;|&|/|\||\n', re.IGNORECASE)
# Several stations named in one answer are usually just concatenated with no
# punctuation (e.g. "Lama fm Alliance fm Golf fm", "Radio wêkè Radio
# Attakè") -- the write-in convention repeats "Radio "/"...FM " between each
# name, so treat those repeats as implicit separators too, in addition to
# the explicit delimiters above.
_STATION_IMPLICIT_SPLIT_RE = re.compile(r'(?<=fm)\s+(?=\S)|(?<=\S)\s+(?=radio\s)', re.IGNORECASE)


def _strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def _station_key(piece):
    """Normalize one station-name mention to a clustering key."""
    s = _STATION_PAREN_RE.sub('', piece.strip())
    s = _STATION_FREQ_RE.sub('', s).strip()
    s = _strip_accents(s).lower()
    s = re.sub(r"[.\-':]", ' ', s)
    s = re.sub(r'\bradio\b', ' ', s)
    s = re.sub(r'\bfm\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return STATION_KEY_MERGES.get(s, s)


def split_station_answer(raw):
    """Split one free-text station-name answer into its component mentions."""
    pieces = []
    for chunk in _STATION_DELIM_RE.split(str(raw)):
        pieces.extend(p for p in _STATION_IMPLICIT_SPLIT_RE.split(chunk) if p.strip())
    return [p.strip() for p in pieces if p.strip()]


def canonicalize_station_columns(data: pd.DataFrame, question_cols=STATION_TEXT_QUESTIONS) -> dict:
    """
    Rewrite the free-text station-name columns in place: each cell becomes
    '|'-joined canonical station labels (splitting compound mentions,
    folding spelling variants). Downstream compute_question_stats() already
    handles '|'-joined multi-select values, so no other code needs to change.

    Builds one canonical map shared across all 3 questions, so e.g. "Royal
    FM" resolves the same way regardless of which question it appears in.
    Returns the key->canonical map for logging/inspection.
    """
    official_by_key = {_station_key(o): o for o in OFFICIAL_STATIONS}

    variant_counts = {}
    for col in question_cols:
        if col not in data.columns:
            continue
        for raw in data[col].dropna():
            for piece in split_station_answer(raw):
                key = _station_key(piece)
                if not key:
                    continue
                variant_counts.setdefault(key, {}).setdefault(piece.strip(), 0)
                variant_counts[key][piece.strip()] += 1

    canonical_map = {}
    for key, variants in variant_counts.items():
        if key in official_by_key:
            canonical_map[key] = official_by_key[key]
        else:
            # Most frequent exact spelling; ties broken alphabetically for determinism.
            canonical_map[key] = sorted(variants, key=lambda v: (-variants[v], v))[0]

    for col in question_cols:
        if col not in data.columns:
            continue

        def _rewrite(raw):
            if pd.isna(raw):
                return raw
            labels = []
            for piece in split_station_answer(raw):
                key = _station_key(piece)
                label = canonical_map.get(key)
                if label and label not in labels:
                    labels.append(label)
            return '|'.join(labels) if labels else raw

        data[col] = data[col].apply(_rewrite)

    return canonical_map

# ── Station mapping ───────────────────────────────────────────────────────────
STATION_STATE = {
    "2026-01-29_160356_maranatha_GW_50":                                 "South-South: Littoral, Ouémé, Atlantique",
    "2026-01-29_165417_Solidarité FM_GW_50.gpkg":                        "Donga",
    "2026-01-29_161633_deeman radio_GW_50.gpkg":                         "Borgou",
    "2026-01-29_162016_Radio TONASSE (higher antenna)_GW_50.gpkg":       "Centre-South: Zou and partially Plateaux, Couffo",
    "2026-01-29_161436_nanto_GW_50.gpkg":                                "Atacora"
}

STATE_ORDER = ["Atlantique", "Borgou", "Couffo", "Donga", "Oueme", "Plateau", "Zou"]

# Radio listening questions to analyze
#
# 2026-08-26: added radio_tv_figures and favourite_radio_drama, requested
# from the "radio_habits"/"radio_favourite" XLSForm groups. Of the 5
# questions asked for, 3 (favourite_radio_format, radio_when, radio_what)
# were already here -- column names confirmed against the survey sheet of
# [Fon] Benin Questionnaire d'Enquete.xlsx (question labels/types, not
# respondent data):
#   radio_tv_figures      select_multiple public_figures_benin — "Quel
#                          personnage de radio ou de télévision admirez-vous ?"
#   favourite_radio_drama text — "Quels sont vos cinq programmes radio
#                          préférés ?" — free text, not a coded choice list,
#                          so each distinct answer becomes its own row here
#                          (parse_radio_cell/value_counts treats it like any
#                          other category) rather than resolving to a
#                          handful of clean options like the select_* fields
#                          above.
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
    'radio_tv_figures',
    'favourite_radio_drama',
]

# 2026-08-27: questions whose raw answers are bilingual "French/Fon" choice-
# list labels (e.g. "Réseaux sociaux/ xójlawema") -- these get the French
# half stripped before tallying, same convention etl_family_planning.py's
# _strip_hausa() already applies (Benin's order is French first, unlike
# Niger's Hausa/French this helper was originally named for). Without this,
# the app's "Show in English" toggle has nothing to translate against: its
# lookup dict (see benin_app/data/radio_translations.csv) is keyed on the
# clean French label, not the raw bilingual string. Excludes the 3
# station-name questions (handled by canonicalize_station_columns() instead,
# not a fixed choice list) and favourite_radio_drama (genuine free text --
# stripping on "/" would wrongly truncate any answer that happens to
# contain one).
RADIO_BILINGUAL_QUESTIONS = [
    'media_type',
    'radio_consumption_method',
    'favourite_radio_format',
    'radio_ad_saturation',
    'radio_language',
    'radio_when',
    'radio_what',
    'radio_trust',
    'radio_influence',
    'radio_tv_figures',
]


def strip_fon(text):
    """Return only the French half of a bilingual 'French/Fon' label.
    Handles pipe-delimited multi-select values. Preserves NaN."""
    if pd.isna(text):
        return text
    text = str(text).strip()
    if "|" in text:
        parts = [strip_fon(p) for p in text.split("|")]
        parts = [p for p in parts if pd.notna(p) and p != ""]
        return "|".join(parts) if parts else np.nan
    if "/" in text:
        french = text.split("/", 1)[0].strip()
        return french if french else text
    return text


# ── Spatial station labeling ─────────────────────────────────────────────────

def add_station_labels(data: pd.DataFrame, station_path: str, max_distance: int = 1000) -> pd.DataFrame:
    """
    Add station_label column to data using spatial join and nearest-neighbor snapping.
    
    Args:
        data: DataFrame with GPS coordinates (gps-Longitude, gps-Latitude)
        station_path: Glob pattern to station .gpkg files
        max_distance: Max distance (meters) for snapping to nearest station
    
    Returns:
        (data, match_df) -- data has the added 'station_label' column
        (each respondent's single NEAREST station); match_df is every
        (respondent, station) pair that matched at all, before that
        resolution -- feed it to build_station_overlap_matrix() to see
        respondents shared across overlapping stations.
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
    # 2026-08-26: collect every polygon match per point across ALL stations
    # before assigning station_label, instead of assigning inside the loop.
    # The previous code called `data.loc[points_in_station.index,
    # 'station_label'] = station_name` on every iteration, so when a point
    # fell inside two overlapping station coverage polygons, whichever
    # station was processed LAST in glob.glob()'s (arbitrary, filesystem-
    # dependent) file order silently won -- confirmed via a real case:
    # Maranatha's own iteration logged "Assigned 34 points" but ended up
    # with 0 in the final table, because a later-processed overlapping
    # station's polygon reassigned every one of those 34 respondents with no
    # trace. Now: a point inside multiple polygons is assigned to whichever
    # station's coverage-polygon centroid it's closest to -- deterministic,
    # independent of file iteration order.
    match_candidates = []  # (row_index, station_name, distance_to_centroid)

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

        print(f"    {len(points_in_station)} points fall within {station_name}'s coverage area")
        if len(points_in_station) > 0:
            # Bounding-box center, not a true polygon union+centroid -- these
            # station files are building-level coverage grids (thousands of
            # small polygons each), so a real union_all() over all of them is
            # far too slow for what's only needed as a rough tie-breaker
            # between overlapping stations, not a precise geometric centroid.
            minx, miny, maxx, maxy = station_gdf_projected.total_bounds
            centroid = Point((minx + maxx) / 2, (miny + maxy) / 2)
            for idx in points_in_station.index.unique():
                dist = data_gdf_projected.loc[idx, 'geometry'].distance(centroid)
                match_candidates.append((idx, station_name, dist))

    # Resolve overlaps: each point goes to its closest-centroid match.
    if match_candidates:
        match_df = pd.DataFrame(match_candidates, columns=['idx', 'station_name', 'dist'])
        best = match_df.loc[match_df.groupby('idx')['dist'].idxmin()]
        overlap_count = len(match_df) - len(best)
        if overlap_count > 0:
            print(f"    Resolved {overlap_count} points that fell inside more than one "
                  f"station's coverage area (assigned to nearest centroid)")
        data.loc[best['idx'], 'station_label'] = best['station_name'].values
        for station_name, n in best['station_name'].value_counts().items():
            print(f"    Assigned {n} points to {station_name}")
    else:
        match_df = pd.DataFrame(columns=['idx', 'station_name', 'dist'])

    # Snap unmatched points to nearest station
    unmatched_indices = data[data['station_label'].isna()].index
    unmatched_points = data_gdf_projected.loc[unmatched_indices]

    if len(unmatched_points) > 0 and len(all_stations) > 0:
        print(f"    Snapping {len(unmatched_points)} unmatched points (within {max_distance}m)...")
        all_stations_gdf = pd.concat(all_stations, ignore_index=True)
        all_stations_gdf = all_stations_gdf.reset_index(drop=True)

        snapped_count = 0
        snapped_rows = []
        for idx in unmatched_indices:
            point_geom = data_gdf_projected.loc[idx, 'geometry']
            all_stations_gdf['distance'] = all_stations_gdf.geometry.distance(point_geom)
            min_dist_position = all_stations_gdf['distance'].argmin()
            nearest_station = all_stations_gdf.iloc[min_dist_position]

            if nearest_station['distance'] <= max_distance:
                data.loc[idx, 'station_label'] = nearest_station['station_name']
                snapped_rows.append((idx, nearest_station['station_name'], nearest_station['distance']))
                snapped_count += 1

        # 2026-08-26: fold snapped points into match_df too, so the overlap
        # matrix built from it (see build_station_overlap_matrix) reflects
        # every respondent's station membership, not just the ones that fell
        # directly inside a polygon. A snapped point is single-station by
        # definition (it wasn't inside ANY polygon), so it never contributes
        # an off-diagonal overlap -- only its own diagonal count.
        if snapped_rows:
            match_df = pd.concat(
                [match_df, pd.DataFrame(snapped_rows, columns=['idx', 'station_name', 'dist'])],
                ignore_index=True,
            )

        print(f"    Snapped {snapped_count} points; {len(unmatched_points) - snapped_count} beyond threshold")

    print(f"    Total labeled: {data['station_label'].notna().sum()} / {len(data)}")

    # Drop respondents without station labels
    data = data.dropna(subset=['station_label']).copy()
    match_df = match_df[match_df['idx'].isin(data.index)]
    return data, match_df


def build_station_overlap_matrix(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Station x station matrix from add_station_labels()'s raw match_df
    (every station whose coverage polygon contains each respondent, before
    resolving to one nearest station). Cell (A, B) = number of respondents
    whose point falls within BOTH A's and B's coverage area; the diagonal
    (A, A) = total respondents matched to A at all (== the "N points fall
    within" counts logged during the spatial join, plus any snapped-in
    points). This is what actually answers "how many of the same
    respondents fall within multiple stations" -- add_station_labels()
    itself only keeps each respondent's single nearest station.
    """
    if match_df.empty:
        return pd.DataFrame()

    stations = sorted(match_df['station_name'].unique())
    membership = match_df.groupby('idx')['station_name'].apply(set)

    mat = pd.DataFrame(0, index=stations, columns=stations, dtype=int)
    for station_set in membership:
        for a in station_set:
            mat.loc[a, a] += 1
        for a in station_set:
            for b in station_set:
                if a != b:
                    mat.loc[a, b] += 1
    return mat


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

def build_radio_table(data: pd.DataFrame, station_members: dict, weight_col: str = WEIGHT_COL) -> tuple:
    """Build the full radio summary table with stations as columns.

    2026-08-27: switched from exclusive "nearest station wins" assignment to
    counting a respondent toward EVERY station whose coverage polygon
    contains them -- requested directly, so a respondent inside overlapping
    coverage doesn't silently vanish from a station just because a
    neighboring one was marginally closer (see the Maranatha case worked
    through in this session's "Overlapping Signals" artifact). This means
    the SAME respondent's answers get counted more than once across
    different station columns -- the `_shared_n` column below says exactly
    how many of each station's respondents that applies to, so the overlap
    is visible rather than silent.

    Args:
        data: respondent-level data (weight column, question columns)
        station_members: {station_name: set(row indices)} -- every
            respondent whose point falls inside that station's coverage
            polygon, from add_station_labels()'s match_df (NOT the single
            nearest-station station_label column).
    """
    if not station_members:
        print("  Warning: no station membership data")
        return pd.DataFrame(), []

    stations = sorted(station_members.keys())

    if not stations:
        print("  Warning: No stations found in data")
        return pd.DataFrame(), []

    print(f"  [radio] Building station-level table for {len(stations)} stations "
          f"(respondents in overlapping coverage areas count toward each)...")

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
            idx = [i for i in station_members[station] if i in data.index]
            station_filter = data.loc[idx]

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

    # Add state + shared-respondent-count columns for each station
    extra_columns = {}
    for station in stations:
        members = station_members[station]
        shared = sum(
            1 for i in members
            if any(i in station_members[other] for other in stations if other != station)
        )
        extra_columns[f"{station}_total_n"] = len(members)
        extra_columns[f"{station}_shared_n"] = shared
        extra_columns[f"{station}_state"] = STATION_STATE.get(station, "Unknown")

    # Insert shared-count + state columns right after each station's data
    final_cols = []
    for station in stations:
        final_cols.append(station)
        final_cols.append(f"{station}_n")
        final_cols.append(f"{station}_wn")
        final_cols.append(f"{station}_total_n")
        final_cols.append(f"{station}_shared_n")
        final_cols.append(f"{station}_state")

    # Reorder columns and add the extra data
    for col, val in extra_columns.items():
        df_out[col] = val

    df_out = df_out[final_cols]

    return df_out, stations


def build_radio_table_by_state(data: pd.DataFrame, weight_col: str = WEIGHT_COL) -> pd.DataFrame:
    """Build radio summary table with states as columns instead of individual stations.

    Groups by the respondent's own recorded province, not by which radio
    station's coverage area they fall into -- several stations cover multiple
    provinces (see STATION_STATE), so station-coverage-based grouping would
    either double-count respondents across provinces or silently drop them
    when a station's value doesn't exactly match a STATE_ORDER entry.
    """

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
            # Filter to respondents recorded as living in this province
            state_filter = data[data['province'] == state]

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
    station_match_df = None
    if station_path:
        data, station_match_df = add_station_labels(data, station_path)
    else:
        print("  Warning: station_path not provided — skipping spatial join")

    # Canonicalize the free-text station-name questions (station_most_listened
    # etc.) before they're tallied -- see canonicalize_station_columns() above.
    print("  [radio] canonicalizing free-text station-name answers...")
    canonical_map = canonicalize_station_columns(data)
    print(f"    {len(canonical_map)} distinct station mentions -> "
          f"{len(set(canonical_map.values()))} canonical stations")

    # Strip the Fon half off the bilingual choice-list questions so the
    # "Show in English" toggle has clean French text to translate against
    # -- see RADIO_BILINGUAL_QUESTIONS/strip_fon() above.
    print("  [radio] stripping Fon half of bilingual choice-list answers...")
    for col in RADIO_BILINGUAL_QUESTIONS:
        if col in data.columns:
            data[col] = data[col].apply(strip_fon)

    # Build station-level table -- every station a respondent's point falls
    # inside, not just the nearest one (see build_radio_table's docstring).
    station_members = {}
    if station_match_df is not None and not station_match_df.empty:
        station_members = station_match_df.groupby('station_name')['idx'].apply(set).to_dict()
    df_out, station_names = build_radio_table(data, station_members, WEIGHT_COL)

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
    out_path = f"{APP_DATA_DIR}/{COUNTRY}_question_table_by_station_{date_string}.csv"
    df_out.to_csv(out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df_out.shape}  —  {len(station_names)} stations × {len(df_out)} rows")

    # Save state-level
    out_path_state = f"{APP_DATA_DIR}/{COUNTRY}_question_table_by_state_{date_string}.csv"
    df_out_state.to_csv(out_path_state)
    print(f"  Saved: {out_path_state}")
    print(f"  Shape: {df_out_state.shape}  —  {len(STATE_ORDER)} states × {len(df_out_state)} rows")

    # Save station-overlap matrix -- how many of the SAME respondents fall
    # within multiple stations' coverage areas (requested 2026-08-26, after
    # the nearest-centroid fix above made stations like Maranatha show 0
    # respondents even though they have real, just fully-overlapping,
    # matches). Non-PII: station names x counts only, same as the other
    # outputs -- no respondent-level data.
    if station_match_df is not None and not station_match_df.empty:
        overlap_matrix = build_station_overlap_matrix(station_match_df)
        out_path_overlap = f"{APP_DATA_DIR}/{COUNTRY}_radio_station_overlap_{date_string}.csv"
        overlap_matrix.to_csv(out_path_overlap)
        print(f"  Saved: {out_path_overlap}")
        print(f"  Shape: {overlap_matrix.shape}  —  {len(overlap_matrix)} stations x {len(overlap_matrix)} stations")
    