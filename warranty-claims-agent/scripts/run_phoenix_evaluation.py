#!/usr/bin/env python3
"""
Phoenix-based Evaluation Script for Warranty Claims Agent.

This script demonstrates how to use Arize Phoenix for:
1. Tracing agent workflows
2. Running LLM-as-Judge evaluations
3. Detecting hallucinations
4. Evaluating reasoning quality

Usage:
    # First, run some claims through the agent to generate traces
    python src/agent.py --file data/test_claims/valid/valid_001.json --no-interrupt

    # Then run this evaluation script
    python scripts/run_phoenix_evaluation.py

    # Or run everything together
    python scripts/run_phoenix_evaluation.py --run-claims
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_sample_claims(num_claims: int = 5):
    """
    Run a sample of claims through the agent to generate traces.
    """
    # Initialize Phoenix tracing BEFORE importing agent
    from src.phoenix_tracing import initialize_phoenix
    initialize_phoenix(project_name="warranty-claims", launch_app=False)

    from src.agent import get_compiled_graph, process_claim_file
    from src.config import TEST_CLAIMS_DIR, DATABASE_PATH

    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Setup checkpointer
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Compile graph without interrupts for automated processing
    app = get_compiled_graph(checkpointer=checkpointer, interrupt_before=None)

    # Collect test files
    test_files = []
    for category in ['valid', 'invalid', 'inquiry', 'spam']:
        category_dir = TEST_CLAIMS_DIR / category
        if category_dir.exists():
            test_files.extend(list(category_dir.glob('*.json'))[:num_claims])

    print(f"\nProcessing {len(test_files)} claims for tracing...")
    print("-" * 40)

    results = []
    for i, filepath in enumerate(test_files[:num_claims * 4], 1):
        print(f"[{i}/{min(len(test_files), num_claims * 4)}] Processing: {filepath.name}")
        try:
            result = process_claim_file(filepath, app)
            results.append({
                "file": filepath.name,
                "claim_id": result['claim_id'],
                "classification": result['result'].get('classification'),
                "recommendation": result['result'].get('adjudication_result', {}).get('recommendation'),
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            results.append({
                "file": filepath.name,
                "status": "error",
                "error": str(e)
            })

    print(f"\nProcessed {len(results)} claims")
    return results


def get_traces_summary():
    """Get a summary of available traces from Phoenix."""
    try:
        from src.phoenix_tracing import get_traces_dataframe

        df = get_traces_dataframe()

        if df is None or df.empty:
            print("No traces found. Run some claims first with:")
            print("  python src/agent.py --file data/test_claims/valid/valid_001.json --no-interrupt")
            return None

        print(f"\nTotal spans: {len(df)}")
        print(f"\nSpan types:")

        if 'name' in df.columns:
            for name in df['name'].unique():
                count = len(df[df['name'] == name])
                print(f"  {name}: {count}")

        return df

    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        return None


def run_evaluations():
    """Run all evaluation types on the traces."""
    try:
        from src.phoenix_tracing import (
            get_traces_dataframe,
            run_adjudication_evaluation,
            run_hallucination_check,
            run_reasoning_quality_eval,
            print_evaluation_summary
        )
    except ImportError as e:
        print(f"Error importing Phoenix modules: {e}")
        print("Install with: pip install arize-phoenix openinference-instrumentation-langchain")
        return

    print("\n" + "=" * 60)
    print("PHOENIX EVALUATION RESULTS")
    print("=" * 60)

    # Get traces
    print("\n[1/4] Fetching traces from Phoenix...")
    try:
        df = get_traces_dataframe()
        if df is None or df.empty:
            print("No traces found. Please run some claims first.")
            return
        print(f"Found {len(df)} spans")
    except Exception as e:
        print(f"Error fetching traces: {e}")
        print("\nMake sure Phoenix is running. Start it with:")
        print("  python -c \"from src.phoenix_tracing import quick_start; quick_start()\"")
        return

    # Run Adjudication Accuracy Evaluation
    print("\n[2/4] Running Adjudication Accuracy Evaluation...")
    try:
        accuracy_results = run_adjudication_evaluation(df, sample_size=10)
        print_evaluation_summary(accuracy_results, "Adjudication Accuracy")
    except Exception as e:
        print(f"Error running accuracy evaluation: {e}")

    # Run Hallucination Detection
    print("\n[3/4] Running Hallucination Detection...")
    try:
        hallucination_results = run_hallucination_check(df, sample_size=10)
        print_evaluation_summary(hallucination_results, "Hallucination Detection")
    except Exception as e:
        print(f"Error running hallucination check: {e}")

    # Run Reasoning Quality Evaluation
    print("\n[4/4] Running Reasoning Quality Evaluation...")
    try:
        reasoning_results = run_reasoning_quality_eval(df, sample_size=10)
        print_evaluation_summary(reasoning_results, "Reasoning Quality")
    except Exception as e:
        print(f"Error running reasoning evaluation: {e}")

    print("\n" + "=" * 60)
    print("Evaluation complete! View detailed traces at: http://localhost:6006")
    print("=" * 60)


def demo_simple_evaluation():
    """
    Demonstrate a simple custom evaluation without running the full agent.
    This is useful for understanding how Phoenix evaluations work.
    """
    print("\n" + "=" * 60)
    print("SIMPLE EVALUATION DEMO")
    print("=" * 60)

    try:
        from phoenix.evals import llm_classify, OpenAIModel
        import pandas as pd
    except ImportError:
        print("Phoenix evals not available. Install with: pip install arize-phoenix")
        return

    # Calculate dates relative to today for realistic demo
    from datetime import date, timedelta
    today = date.today()

    # Sample adjudication outputs to evaluate
    sample_data = [
        {
            "claim_id": "CLM-001",
            "product": "AirFlow Pro",
            "purchase_date": (today - timedelta(days=45)).isoformat(),  # 45 days ago - within warranty
            "issue": "Motor stopped working after 2 months",
            "recommendation": "APPROVE",
            "reasoning": "Claim is within 90-day warranty period. Motor failure is a covered defect. No exclusions apply.",
            "expected": "APPROVE"
        },
        {
            "claim_id": "CLM-002",
            "product": "TravelDry Mini",
            "purchase_date": (today - timedelta(days=180)).isoformat(),  # 180 days ago - expired
            "issue": "Hair dryer overheats",
            "recommendation": "REJECT",
            "reasoning": "Warranty expired. Purchase was 180 days ago, exceeding the 90-day warranty.",
            "expected": "REJECT"
        },
        {
            "claim_id": "CLM-003",
            "product": "SalonMaster 3000",
            "purchase_date": (today - timedelta(days=30)).isoformat(),  # 30 days ago - within warranty but water damage
            "issue": "Unit stopped working after being dropped in water",
            "recommendation": "REJECT",
            "reasoning": "Water damage is explicitly excluded from warranty coverage per policy section 4.2.",
            "expected": "REJECT"
        },
        {
            "claim_id": "CLM-004",
            "product": "QuietBlow Deluxe",
            "purchase_date": (today - timedelta(days=60)).isoformat(),  # 60 days ago - within warranty
            "issue": "Fan makes loud noise",
            "recommendation": "APPROVE",
            "reasoning": "Within warranty period. Mechanical noise indicates manufacturing defect. Covered under warranty.",
            "expected": "APPROVE"
        }
    ]

    df = pd.DataFrame(sample_data)

    # Simple accuracy evaluation template
    eval_template = f"""
    Evaluate if this warranty claim recommendation is correct.

    Product: {{product}}
    Purchase Date: {{purchase_date}}
    Issue: {{issue}}
    AI Recommendation: {{recommendation}}
    AI Reasoning: {{reasoning}}

    Today's date is {today.isoformat()} and standard warranty is 90 days.

    Rules:
    - APPROVE if within 90 days AND issue is a manufacturing defect
    - REJECT if warranty expired (>90 days from purchase)
    - REJECT if issue is caused by water damage, misuse, or accidents

    Is this recommendation correct?
    Respond with ONLY: "correct" or "incorrect"
    """

    print("\n[1/2] Creating evaluation model...")
    model = OpenAIModel(model="gpt-4o-mini")

    print("[2/2] Running LLM-as-Judge evaluation...")
    results = llm_classify(
        dataframe=df,
        model=model,
        template=eval_template,
        rails=["correct", "incorrect"],
        provide_explanation=True
    )

    # Print results
    print("\n" + "-" * 60)
    print("EVALUATION RESULTS")
    print("-" * 60)

    for i, (idx, row) in enumerate(results.iterrows()):
        claim = sample_data[i]
        print(f"\nClaim: {claim['claim_id']} ({claim['product']})")
        print(f"  Recommendation: {claim['recommendation']}")
        print(f"  Expected: {claim['expected']}")
        print(f"  Evaluation: {row.get('label', 'N/A')}")
        if 'explanation' in row:
            print(f"  Explanation: {row['explanation'][:100]}...")

    # Calculate accuracy
    if 'label' in results.columns:
        correct_count = (results['label'] == 'correct').sum()
        total = len(results)
        accuracy = (correct_count / total) * 100
        print(f"\n{'='*60}")
        print(f"Overall Accuracy: {correct_count}/{total} ({accuracy:.1f}%)")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phoenix-based evaluation on warranty claims agent"
    )
    parser.add_argument(
        "--run-claims",
        action="store_true",
        help="Run sample claims through the agent first"
    )
    parser.add_argument(
        "--num-claims",
        type=int,
        default=3,
        help="Number of claims to run per category (default: 3)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a simple demo evaluation without the full agent"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show trace summary, don't run evaluations"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("WARRANTY CLAIMS AGENT - PHOENIX EVALUATION")
    print("=" * 60)

    # Start Phoenix first
    print("\n[SETUP] Starting Phoenix...")
    try:
        from src.phoenix_tracing import initialize_phoenix
        initialize_phoenix(project_name="warranty-claims", launch_app=True)
    except Exception as e:
        print(f"Warning: Could not initialize Phoenix: {e}")
        print("Continuing without tracing UI...")

    if args.demo:
        # Run simple demo
        demo_simple_evaluation()
        return

    if args.run_claims:
        # Run sample claims to generate traces
        print("\n[STEP 1] Running sample claims...")
        run_sample_claims(num_claims=args.num_claims)

    if args.summary_only:
        # Just show trace summary
        print("\n[TRACES] Getting trace summary...")
        get_traces_summary()
        return

    # Run full evaluations
    print("\n[STEP 2] Running evaluations...")
    run_evaluations()


if __name__ == "__main__":
    main()
