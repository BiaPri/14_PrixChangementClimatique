#!/usr/bin/env python
"""Launch DuckDB UI for the database."""

import sys
import webbrowser
from pathlib import Path

import duckdb


def main():
    # Get the project root (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    exploration_dir = project_root / "data" / "exploration"

    db_name = "dev.duckdb"
    # Default to odis.duckdb, or use command line argument
    odis_db_name = sys.argv[1] if len(sys.argv) > 1 else "odis.duckdb"
    db_file = exploration_dir / db_name
    odis_db_file = exploration_dir / odis_db_name

    # Check if database exists
    if not odis_db_file.exists():
        print(f"❌ Database not found: {odis_db_file}")
        print(f"\nAvailable databases in {exploration_dir}:")
        for db in exploration_dir.glob("*.duckdb"):
            print(f"  - {db.name}")
        sys.exit(1)

    print(f"🦆 Launching DuckDB UI for {db_file} {odis_db_file}...")

    # Connect to the database and start UI
    conn = duckdb.connect(str(db_file))
    conn.sql(f"ATTACH DATABASE '{odis_db_file}' AS odis;")
    result = conn.sql("CALL start_ui();").fetchone()

    if result and result[0]:
        url = result[0]
        print(f"✅ DuckDB UI started at: {url}")
        print("Opening in browser...")
        webbrowser.open(url)
        print("\nPress Ctrl+C to stop the server")

        # Keep the connection alive
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping DuckDB UI...")
            conn.close()
    else:
        print("❌ Failed to start DuckDB UI")
        sys.exit(1)


if __name__ == "__main__":
    main()
