"""
Niger FEM Survey — Analysis Pipeline
=====================================
Run this script OUTSIDE the niger_app folder to process the PII dataset
and export pre-aggregated summary CSVs into niger_app/data/.

Usage:
    cd <project_root>
    python pipeline/run_pipeline.py

    # Override paths via environment variables:
    FEM_MAPPED_DATA=/path/to/data.csv \\
    FEM_APP_DATA_DIR=/path/to/niger_app/data \\
    python pipeline/run_pipeline.py

    # Run only specific pages:
    python pipeline/run_pipeline.py --pages access statements

The app reads only the CSVs in niger_app/data/ — the raw PII dataset
never leaves this pipeline folder.
"""

import sys
import os
import argparse
import time
import traceback

sys.tracebacklimit = None  # Show full traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Allow imports from the project root

from pipeline.config import DIR_MAPPED_DATA, APP_DATA_DIR, DIR_STATEMENT_LABELS, station_path
from pipeline.utils import load_raw

import pipeline.etl_respondents     as etl_respondents
import pipeline.etl_access         as etl_access
import pipeline.etl_statements      as etl_statements
import pipeline.etl_family_planning as etl_family_planning
import pipeline.etl_personality     as etl_personality
import pipeline.etl_personas        as etl_personas
import pipeline.etl_radio           as etl_radio

ALL_PAGES = ["respondents", "access", "statements", "family_planning", "personality", "personas", "radio"]

def run_pipeline(pages=None):
    pages = pages or ALL_PAGES
    t0 = time.time()

    print("=" * 60)
    print("Niger FEM Survey — Analysis Pipeline")
    print("=" * 60)
    print(f"  Raw data  : {DIR_MAPPED_DATA}")
    print(f"  Output dir: {APP_DATA_DIR}")
    print(f"  Pages     : {', '.join(pages)}")
    print()

    # Load raw data once — shared across all ETL modules
    print("[1/2] Loading raw dataset...")
    if not os.path.exists(DIR_MAPPED_DATA):
        print(f"  ERROR: File not found: {DIR_MAPPED_DATA}")
        print("  Set the FEM_MAPPED_DATA environment variable or update config.py")
        sys.exit(1)

    df = load_raw(DIR_MAPPED_DATA)
    print()

    # Run selected ETL modules
    print("[2/2] Running ETL modules...")
    errors = []

    for page in pages:
        try:
            if page == "respondents":
                etl_respondents.run(df)
            elif page == "access":
                etl_access.run(df)
            elif page == "statements":
                etl_statements.run(df, statement_labels_path=DIR_STATEMENT_LABELS)
            elif page == "family_planning":
                etl_family_planning.run(df)
            elif page == "personality":
                etl_personality.run(df)
            elif page == "personas":
                etl_personas.run(df)
            elif page == "radio":
                etl_radio.run(df, station_path)
            else:
                print(f"  WARNING: unknown page '{page}' — skipping")
        except Exception as e:
            errors.append((page, str(e)))
            print(f"\n{'='*70}")
            print(f"FULL ERROR TRACEBACK:")
            print(f"{'='*70}")
            traceback.print_exc()
            print(f"{'='*70}")
            raise

    elapsed = round(time.time() - t0, 1)
    print()
    print("=" * 60)

    if errors:
        print(f"Pipeline completed with {len(errors)} error(s) in {elapsed}s:")
        for page, msg in errors:
            print(f"  [{page}] {msg}")
    else:
        print(f"Pipeline completed successfully in {elapsed}s.")
        print(f"Output CSVs written to: {APP_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Niger FEM analysis pipeline")
    parser.add_argument(
        "--pages", nargs="*", choices=ALL_PAGES,
        help=f"Pages to process (default: all). Choices: {ALL_PAGES}",
    )
    args = parser.parse_args()
    run_pipeline(pages=args.pages)
