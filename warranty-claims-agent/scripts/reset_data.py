"""
Reset warranty claims agent data for demos.

Usage:
    python scripts/reset_data.py --full      # Reset everything
    python scripts/reset_data.py --partial   # Keep policies
    python scripts/reset_data.py --dry-run   # Show what would be deleted
"""

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATABASE_PATH,
    CHROMA_PERSIST_DIR,
    INBOX_DIR,
    OUTBOX_DIR,
    LABELS_DIR,
    TEST_CLAIMS_DIR,
)


def reset_full(dry_run=False):
    """Full reset: delete all data including policies."""
    print("=" * 60)
    print("FULL RESET - All data will be deleted")
    print("=" * 60)

    files_to_delete = [
        DATABASE_PATH,
        DATABASE_PATH.with_suffix('.db-shm'),
        DATABASE_PATH.with_suffix('.db-wal'),
    ]

    dirs_to_delete = [
        CHROMA_PERSIST_DIR,
        OUTBOX_DIR,
        LABELS_DIR,
    ]

    # Delete files
    for file in files_to_delete:
        if file.exists():
            print(f"  {'[DRY RUN] ' if dry_run else ''}Deleting file: {file}")
            if not dry_run:
                file.unlink()

    # Delete directories
    for dir in dirs_to_delete:
        if dir.exists():
            print(f"  {'[DRY RUN] ' if dry_run else ''}Deleting directory: {dir}")
            if not dry_run:
                shutil.rmtree(dir)

    # Recreate empty directories
    if not dry_run:
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n✅ Full reset complete!" if not dry_run else "\n[DRY RUN] Preview complete")


def reset_partial(dry_run=False):
    """Partial reset: keep policies, delete processed claims."""
    print("=" * 60)
    print("PARTIAL RESET - Keeping ChromaDB policies")
    print("=" * 60)

    files_to_delete = [
        DATABASE_PATH,
        DATABASE_PATH.with_suffix('.db-shm'),
        DATABASE_PATH.with_suffix('.db-wal'),
    ]

    # Delete claim database
    locked_files = []
    for file in files_to_delete:
        if file.exists():
            print(f"  {'[DRY RUN] ' if dry_run else ''}Deleting: {file}")
            if not dry_run:
                try:
                    file.unlink()
                except PermissionError:
                    locked_files.append(file.name)
                    print(f"    ⚠️  LOCKED: {file.name} (close Streamlit/agent first)")

    if locked_files and not dry_run:
        print(f"\n⚠️  WARNING: {len(locked_files)} file(s) locked:")
        print("    Stop these processes first:")
        print("    - Streamlit dashboard (Ctrl+C)")
        print("    - Watch mode agent (Ctrl+C)")
        print("    Then run reset again.")

    # Delete outbox files
    if OUTBOX_DIR.exists():
        print(f"  {'[DRY RUN] ' if dry_run else ''}Clearing outbox directory")
        if not dry_run:
            for file in OUTBOX_DIR.glob("*"):
                if file.is_file():
                    file.unlink()

    # Delete labels
    if LABELS_DIR.exists():
        print(f"  {'[DRY RUN] ' if dry_run else ''}Clearing labels directory")
        if not dry_run:
            for file in LABELS_DIR.glob("*.pdf"):
                file.unlink()

    # Delete evaluation results
    print(f"  {'[DRY RUN] ' if dry_run else ''}Clearing evaluation results")
    if not dry_run:
        for file in TEST_CLAIMS_DIR.glob("evaluation_results_*.json"):
            file.unlink()

    print("\n✅ Partial reset complete!" if not dry_run else "\n[DRY RUN] Preview complete")
    print("\nChromaDB policies preserved - no need to re-ingest")


def main():
    parser = argparse.ArgumentParser(description="Reset warranty claims agent data")
    parser.add_argument("--full", action="store_true", help="Full reset (delete everything)")
    parser.add_argument("--partial", action="store_true", help="Partial reset (keep policies)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    args = parser.parse_args()

    if not (args.full or args.partial):
        # Default: partial reset
        args.partial = True

    if args.full and args.partial:
        print("Error: Cannot specify both --full and --partial")
        return

    if args.full:
        reset_full(dry_run=args.dry_run)
        if not args.dry_run:
            print("\n⚠️  Remember to re-ingest policies:")
            print("     python src/ingest.py")
    elif args.partial:
        reset_partial(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
