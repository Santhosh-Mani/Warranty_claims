"""
Utility script to view and query the claims database.

This script provides various views of the SQLite database that stores
LangGraph checkpoints and claim processing state.
"""

import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATABASE_PATH


def view_schema():
    """Show the database schema."""
    print("=" * 80)
    print("DATABASE SCHEMA")
    print("=" * 80)

    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        print("-" * 80)

        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        print(f"{'Column':<30} {'Type':<15} {'Not Null':<10} {'PK':<5}")
        print("-" * 80)
        for col in columns:
            print(f"{col[1]:<30} {col[2]:<15} {'YES' if col[3] else 'NO':<10} {'YES' if col[5] else 'NO':<5}")

    conn.close()


def view_all_threads():
    """Show all threads (claims) in the database."""
    print("\n" + "=" * 80)
    print("ALL THREADS (CLAIMS)")
    print("=" * 80)

    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()

    # Get unique threads
    cursor.execute("""
        SELECT DISTINCT thread_id FROM checkpoints
        ORDER BY checkpoint_id DESC
    """)

    threads = cursor.fetchall()

    if not threads:
        print("No claims in database yet.")
        conn.close()
        return

    print(f"\nTotal threads: {len(threads)}\n")
    print(f"{'Thread ID':<40} {'Latest Checkpoint':<20} {'Status':<15}")
    print("-" * 80)

    for thread in threads:
        thread_id = thread[0]

        # Get latest checkpoint for this thread
        cursor.execute("""
            SELECT checkpoint_id, checkpoint
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """, (thread_id,))

        result = cursor.fetchone()
        if result:
            checkpoint_id = result[0]
            # Checkpoint is stored as blob, try to decode
            try:
                checkpoint_data = result[1]
                # The checkpoint is pickled, so we'll just show metadata
                status = "Stored"
            except:
                status = "Unknown"

            print(f"{thread_id:<40} {checkpoint_id:<20} {status:<15}")

    conn.close()


def view_thread_detail(thread_id: str = None):
    """Show detailed information about a specific thread."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()

    if not thread_id:
        # Get the most recent thread
        cursor.execute("""
            SELECT DISTINCT thread_id FROM checkpoints
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        if not result:
            print("No claims in database.")
            conn.close()
            return
        thread_id = result[0]

    print("\n" + "=" * 80)
    print(f"THREAD DETAIL: {thread_id}")
    print("=" * 80)

    # Get all checkpoints for this thread
    cursor.execute("""
        SELECT checkpoint_id, checkpoint, metadata
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id ASC
    """, (thread_id,))

    checkpoints = cursor.fetchall()

    print(f"\nTotal checkpoints: {len(checkpoints)}")
    print("-" * 80)

    for idx, cp in enumerate(checkpoints):
        checkpoint_id = cp[0]
        metadata = cp[2]

        print(f"\nCheckpoint {idx + 1}: {checkpoint_id}")
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
                print(f"Metadata: {json.dumps(metadata_dict, indent=2)}")
            except:
                print(f"Metadata: {metadata}")

    conn.close()


def view_claim_states():
    """Show the current state of all claims using LangGraph API."""
    from src.agent import get_compiled_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    print("\n" + "=" * 80)
    print("CLAIM STATES (via LangGraph API)")
    print("=" * 80)

    # Get checkpointer and graph
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = get_compiled_graph(checkpointer=checkpointer)

    # Get all thread IDs
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
    threads = [row[0] for row in cursor.fetchall()]

    if not threads:
        print("No claims in database yet.")
        conn.close()
        return

    print(f"\nTotal claims: {len(threads)}\n")

    for idx, thread_id in enumerate(threads):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = app.get_state(config)

            if state and state.values:
                values = state.values

                print(f"\n{idx + 1}. Claim ID: {values.get('claim_id', 'Unknown')}")
                print(f"   Thread ID: {thread_id}")
                print(f"   Classification: {values.get('classification', 'N/A')}")
                print(f"   Timestamp: {values.get('timestamp', 'N/A')}")

                # Show extracted data
                extracted = values.get('extracted_data', {})
                if extracted:
                    print(f"   Customer: {extracted.get('customer_name', 'N/A')}")
                    print(f"   Product: {extracted.get('product_name', 'N/A')}")
                    print(f"   Purchase Date: {extracted.get('purchase_date', 'N/A')}")

                # Show validation status
                missing = values.get('missing_fields', [])
                if missing:
                    print(f"   ⚠️  Missing Fields: {', '.join(missing)}")
                    print(f"   Validation Complete: {values.get('validation_complete', False)}")

                # Show adjudication
                adj = values.get('adjudication_result', {})
                if adj:
                    print(f"   AI Recommendation: {adj.get('recommendation', 'N/A')} ({adj.get('confidence', 'N/A')})")

                # Show human decision
                decision = values.get('human_decision')
                if decision:
                    print(f"   ✅ Human Decision: {decision.upper()}")
                    notes = values.get('human_notes')
                    if notes:
                        print(f"   Notes: {notes}")
                else:
                    print(f"   ⏳ Human Decision: PENDING")

                # Show fulfillment
                response_email = values.get('response_email')
                if response_email:
                    print(f"   📧 Email Sent: {response_email.get('subject', 'N/A')}")
                    label = values.get('return_label_path')
                    if label:
                        print(f"   📦 Return Label: {label}")

                print("-" * 80)

        except Exception as e:
            print(f"\n{idx + 1}. Thread {thread_id}: Error - {e}")
            print("-" * 80)

    conn.close()


def export_to_json(output_file: str = None):
    """Export all claim states to JSON file."""
    from src.agent import get_compiled_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    if not output_file:
        output_file = f"data/claims_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = get_compiled_graph(checkpointer=checkpointer)

    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
    threads = [row[0] for row in cursor.fetchall()]

    claims = []
    for thread_id in threads:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = app.get_state(config)

            if state and state.values:
                claims.append({
                    "thread_id": thread_id,
                    "state": state.values
                })
        except:
            pass

    conn.close()

    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(claims, f, indent=2, default=str)

    print(f"\n✅ Exported {len(claims)} claims to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="View Claims Database")
    parser.add_argument("--schema", action="store_true", help="Show database schema")
    parser.add_argument("--threads", action="store_true", help="Show all threads")
    parser.add_argument("--detail", type=str, help="Show detail for specific thread ID")
    parser.add_argument("--states", action="store_true", help="Show claim states (via LangGraph)")
    parser.add_argument("--export", type=str, nargs='?', const='', help="Export to JSON file")
    parser.add_argument("--all", action="store_true", help="Show all information")

    args = parser.parse_args()

    if not any([args.schema, args.threads, args.detail, args.states, args.export, args.all]):
        # Default: show states
        args.states = True

    if args.all:
        view_schema()
        view_all_threads()
        view_claim_states()
    else:
        if args.schema:
            view_schema()

        if args.threads:
            view_all_threads()

        if args.detail:
            view_thread_detail(args.detail)

        if args.states:
            view_claim_states()

        if args.export is not None:
            output_file = args.export if args.export else None
            export_to_json(output_file)


if __name__ == "__main__":
    main()
