"""
Evaluation Script for Warranty Claims Agent.

Runs the agent against the test dataset and calculates metrics:
- Triage accuracy
- Extraction field accuracy
- Recommendation accuracy
- False approval/rejection rates

Outputs a detailed evaluation report.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TEST_CLAIMS_DIR, DATABASE_PATH
from src.agent import get_compiled_graph, process_claim_file

# LangGraph checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


class EvaluationRunner:
    """Runs evaluation against the test dataset."""

    def __init__(self, auto_decision: bool = True):
        """
        Initialize the evaluation runner.

        Args:
            auto_decision: If True, automatically apply expected decisions
        """
        self.auto_decision = auto_decision
        self.results = []
        self.metrics = defaultdict(lambda: {"correct": 0, "total": 0})

        # Create a fresh database for evaluation
        self.eval_db = DATABASE_PATH.parent / "eval_claims.db"
        if self.eval_db.exists():
            self.eval_db.unlink()

        # Create SQLite connection and checkpointer
        self.eval_db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.eval_db), check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        self.app = get_compiled_graph(
            checkpointer=self.checkpointer,
            interrupt_before=["fulfillment"] if not auto_decision else None
        )

    def load_test_dataset(self) -> List[Dict]:
        """Load the test dataset."""
        dataset_file = TEST_CLAIMS_DIR / "test_dataset.json"

        if not dataset_file.exists():
            print(f"Test dataset not found at {dataset_file}")
            print("Run 'python scripts/generate_claims.py' first")
            return []

        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("claims", [])

    def evaluate_triage(self, expected: str, actual: str) -> bool:
        """Evaluate triage classification."""
        # Normalize classifications
        expected = expected.lower() if expected else ""
        actual = actual.lower() if actual else ""

        # Map inquiry to spam for evaluation purposes (both are non-claims)
        if expected in ["inquiry", "spam"]:
            expected_is_claim = False
        else:
            expected_is_claim = True

        actual_is_claim = actual == "claim"

        return expected_is_claim == actual_is_claim

    def evaluate_recommendation(self, expected: str, actual: str) -> bool:
        """Evaluate recommendation."""
        if not expected or not actual:
            return False

        expected = expected.upper()
        actual = actual.upper()

        return expected == actual

    def run_single_claim(self, claim: Dict) -> Dict:
        """
        Run the agent on a single claim and evaluate results.

        Returns evaluation result dict.
        """
        claim_id = claim.get("id", "unknown")
        claim_type = claim.get("type", "unknown")
        expected_classification = claim.get("expected_classification", "")
        expected_recommendation = claim.get("expected_recommendation", "")
        ground_truth = claim.get("ground_truth", {})

        print(f"\n{'='*60}")
        print(f"Processing: {claim_id} (Type: {claim_type})")
        print(f"{'='*60}")

        # Create temporary file for the claim
        temp_file = TEST_CLAIMS_DIR / f"temp_{claim_id}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(claim, f)

        try:
            # Process the claim
            result = process_claim_file(temp_file, self.app)
            state = result["result"]

            # Get actual results
            actual_classification = state.get("classification", "")
            adjudication = state.get("adjudication_result", {})
            actual_recommendation = adjudication.get("recommendation", "") if adjudication else ""

            # Evaluate triage
            triage_correct = self.evaluate_triage(expected_classification, actual_classification)
            self.metrics["triage"]["total"] += 1
            if triage_correct:
                self.metrics["triage"]["correct"] += 1

            # Evaluate recommendation (only for claims)
            rec_correct = None
            if claim_type not in ["spam", "inquiry"]:
                if expected_recommendation and actual_recommendation:
                    rec_correct = self.evaluate_recommendation(expected_recommendation, actual_recommendation)
                    self.metrics["recommendation"]["total"] += 1
                    if rec_correct:
                        self.metrics["recommendation"]["correct"] += 1

                    # Track false positives/negatives
                    if expected_recommendation == "APPROVE" and actual_recommendation != "APPROVE":
                        self.metrics["false_rejections"]["total"] += 1
                    if expected_recommendation == "REJECT" and actual_recommendation == "APPROVE":
                        self.metrics["false_approvals"]["total"] += 1

            # Evaluate extraction
            extracted = state.get("extracted_data", {})
            if ground_truth and extracted:
                # Check product matching
                if ground_truth.get("product"):
                    expected_product = ground_truth["product"]
                    actual_product = extracted.get("product_id", "")
                    self.metrics["product_match"]["total"] += 1
                    if expected_product == actual_product:
                        self.metrics["product_match"]["correct"] += 1

                # Check purchase date
                if ground_truth.get("purchase_date"):
                    expected_date = ground_truth["purchase_date"]
                    actual_date = extracted.get("purchase_date", "")
                    self.metrics["date_extraction"]["total"] += 1
                    if expected_date == actual_date:
                        self.metrics["date_extraction"]["correct"] += 1

            eval_result = {
                "claim_id": claim_id,
                "claim_type": claim_type,
                "expected_classification": expected_classification,
                "actual_classification": actual_classification,
                "triage_correct": triage_correct,
                "expected_recommendation": expected_recommendation,
                "actual_recommendation": actual_recommendation,
                "recommendation_correct": rec_correct,
                "extracted_data": extracted,
                "adjudication": adjudication,
                "ground_truth": ground_truth
            }

            # Print summary
            print(f"  Classification: {actual_classification} (Expected: {expected_classification}) "
                  f"{'✓' if triage_correct else '✗'}")
            if rec_correct is not None:
                print(f"  Recommendation: {actual_recommendation} (Expected: {expected_recommendation}) "
                      f"{'✓' if rec_correct else '✗'}")

            return eval_result

        except Exception as e:
            print(f"  ERROR: {e}")
            return {
                "claim_id": claim_id,
                "claim_type": claim_type,
                "error": str(e)
            }

        finally:
            # Cleanup temp file
            if temp_file.exists():
                temp_file.unlink()

    def run_evaluation(self) -> Dict:
        """
        Run evaluation on all test claims.

        Returns summary metrics.
        """
        print("\n" + "=" * 70)
        print("WARRANTY CLAIMS AGENT - EVALUATION RUN")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")

        # Load test data
        claims = self.load_test_dataset()
        if not claims:
            return {"error": "No test data found"}

        print(f"\nLoaded {len(claims)} test claims")

        # Run evaluation
        for claim in claims:
            result = self.run_single_claim(claim)
            self.results.append(result)

        # Calculate final metrics
        summary = self.calculate_summary()

        # Print summary
        self.print_summary(summary)

        # Save results
        self.save_results(summary)

        return summary

    def calculate_summary(self) -> Dict:
        """Calculate summary metrics."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_claims": len(self.results),
            "metrics": {}
        }

        for metric_name, counts in self.metrics.items():
            if counts["total"] > 0:
                accuracy = counts["correct"] / counts["total"]
                summary["metrics"][metric_name] = {
                    "correct": counts["correct"],
                    "total": counts["total"],
                    "accuracy": round(accuracy * 100, 2)
                }
            else:
                summary["metrics"][metric_name] = {
                    "correct": 0,
                    "total": 0,
                    "accuracy": 0
                }

        # Calculate overall accuracy
        triage_acc = summary["metrics"].get("triage", {}).get("accuracy", 0)
        rec_acc = summary["metrics"].get("recommendation", {}).get("accuracy", 0)
        summary["overall_accuracy"] = round((triage_acc + rec_acc) / 2, 2) if rec_acc else triage_acc

        # False rates
        false_approvals = summary["metrics"].get("false_approvals", {}).get("total", 0)
        false_rejections = summary["metrics"].get("false_rejections", {}).get("total", 0)
        total_recommendations = summary["metrics"].get("recommendation", {}).get("total", 0)

        if total_recommendations > 0:
            summary["false_approval_rate"] = round(false_approvals / total_recommendations * 100, 2)
            summary["false_rejection_rate"] = round(false_rejections / total_recommendations * 100, 2)

        return summary

    def print_summary(self, summary: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        print(f"\nTotal Claims Evaluated: {summary['total_claims']}")
        print(f"\nOverall Accuracy: {summary.get('overall_accuracy', 'N/A')}%")

        print("\n" + "-" * 40)
        print("Detailed Metrics:")
        print("-" * 40)

        for metric_name, values in summary.get("metrics", {}).items():
            if values["total"] > 0:
                print(f"  {metric_name}:")
                print(f"    Correct: {values['correct']}/{values['total']}")
                print(f"    Accuracy: {values['accuracy']}%")

        if "false_approval_rate" in summary:
            print(f"\nFalse Approval Rate: {summary['false_approval_rate']}%")
        if "false_rejection_rate" in summary:
            print(f"False Rejection Rate: {summary['false_rejection_rate']}%")

        print("\n" + "=" * 70)

    def save_results(self, summary: Dict):
        """Save evaluation results to file."""
        output_file = TEST_CLAIMS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        full_results = {
            "summary": summary,
            "detailed_results": self.results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Warranty Claims Agent")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode (requires human decisions)")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of claims to evaluate (0 = all)")
    args = parser.parse_args()

    # Check if test data exists
    if not (TEST_CLAIMS_DIR / "test_dataset.json").exists():
        print("Test dataset not found!")
        print("Generating test data first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_claims.py"], cwd=Path(__file__).parent.parent)

    # Run evaluation
    runner = EvaluationRunner(auto_decision=not args.interactive)
    summary = runner.run_evaluation()

    # Return exit code based on accuracy
    if summary.get("overall_accuracy", 0) >= 70:
        print("\n✓ Evaluation PASSED (accuracy >= 70%)")
        sys.exit(0)
    else:
        print("\n✗ Evaluation FAILED (accuracy < 70%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
