"""
Phoenix Tracing and Evaluation Module.

Provides:
1. Automatic tracing for LangChain/LangGraph workflows
2. LLM-as-Judge evaluations for warranty claim quality
3. Custom evaluators for adjudication accuracy
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global flag to track if Phoenix is available and initialized
PHOENIX_INITIALIZED = False


def check_phoenix_available() -> bool:
    """Check if Phoenix packages are installed."""
    try:
        import phoenix
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        return True
    except ImportError:
        return False


# Set PHOENIX_AVAILABLE at module load time
PHOENIX_AVAILABLE = check_phoenix_available()


def is_phoenix_running(port: int = 6006) -> bool:
    """Check if Phoenix is already running on the specified port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def initialize_phoenix(
    project_name: str = "warranty-claims",
    endpoint: Optional[str] = None,
    launch_app: bool = True
) -> bool:
    """
    Initialize Phoenix tracing for the warranty claims agent.

    Args:
        project_name: Name of the Phoenix project
        endpoint: Phoenix collector endpoint (default: http://localhost:6006)
        launch_app: Whether to launch the Phoenix app locally

    Returns:
        True if initialization succeeded, False otherwise
    """
    global PHOENIX_AVAILABLE, PHOENIX_INITIALIZED

    if PHOENIX_INITIALIZED:
        logger.info("Phoenix already initialized")
        return True

    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        PHOENIX_AVAILABLE = True

        # Launch local Phoenix app if requested and not already running
        if launch_app:
            if is_phoenix_running():
                logger.info("Phoenix already running at http://localhost:6006")
            else:
                try:
                    px.launch_app()
                    logger.info("Phoenix app launched at http://localhost:6006")
                except Exception as e:
                    logger.warning(f"Could not launch Phoenix app: {e}")

        # Register tracer provider
        endpoint = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint
        )

        # Instrument LangChain (which also covers LangGraph)
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        PHOENIX_INITIALIZED = True
        logger.info(f"Phoenix tracing initialized for project: {project_name}")
        logger.info(f"View traces at: http://localhost:6006")

        return True

    except ImportError as e:
        logger.warning(f"Phoenix not available: {e}")
        logger.warning("Install with: pip install arize-phoenix openinference-instrumentation-langchain")
        PHOENIX_AVAILABLE = False
        return False

    except Exception as e:
        logger.error(f"Error initializing Phoenix: {e}")
        return False


def get_phoenix_client():
    """Get the Phoenix client for querying traces and running evaluations."""
    if not PHOENIX_AVAILABLE:
        raise RuntimeError("Phoenix not available. Install with: pip install arize-phoenix")

    import phoenix as px
    return px.Client()


def get_traces_dataframe(project_name: str = "warranty-claims"):
    """
    Get all traces as a pandas DataFrame.

    Returns:
        DataFrame with trace data including spans, inputs, outputs
    """
    client = get_phoenix_client()
    return client.get_spans_dataframe(project_name=project_name)


# ============================================================================
# Custom Evaluation Templates for Warranty Claims
# ============================================================================

ADJUDICATION_ACCURACY_TEMPLATE = """
You are evaluating a warranty claim adjudication decision.

## Claim Details
Customer: {customer_name}
Product: {product_name}
Purchase Date: {purchase_date}
Issue: {issue_description}
Days Since Purchase: {days_since_purchase}

## Warranty Policy (excerpt)
{policy_excerpt}

## AI Adjudication
Recommendation: {recommendation}
Reasoning: {reasoning}
Warranty Valid: {warranty_valid}
Exclusion Triggered: {exclusion_triggered}

## Your Task
Evaluate whether the AI's recommendation is correct based on:
1. Is the warranty still valid (within warranty period)?
2. Does the issue trigger any exclusions in the policy?
3. Is the reasoning logical and well-supported?

Respond with ONLY one of these labels:
- "correct" - The recommendation is accurate based on the policy
- "incorrect" - The recommendation contradicts the policy or facts
- "uncertain" - Not enough information to determine correctness
"""

HALLUCINATION_DETECTION_TEMPLATE = """
You are checking for hallucinations in a warranty claim adjudication.

## Source Information (Ground Truth)
Warranty Policy:
{policy_text}

Claim Data:
{claim_data}

## AI Response to Evaluate
{ai_response}

## Your Task
Check if the AI's response contains any hallucinations:
1. Does it reference policies or rules not in the source?
2. Does it claim facts not present in the claim data?
3. Does it misquote or misinterpret the policy?

Respond with ONLY one of these labels:
- "factual" - All statements are grounded in the source
- "hallucinated" - Contains statements not supported by the source
"""

REASONING_QUALITY_TEMPLATE = """
You are evaluating the quality of reasoning in a warranty adjudication.

## AI Reasoning
{reasoning}

## Facts Cited
{facts}

## Assumptions Made
{assumptions}

## Policy References
{policy_references}

## Your Task
Evaluate the reasoning quality on these criteria:
1. Are facts clearly separated from assumptions?
2. Is the logic sound and easy to follow?
3. Are policy references specific and relevant?
4. Is the conclusion well-supported by the evidence?

Respond with ONLY one of these labels:
- "high_quality" - Clear, logical, well-supported reasoning
- "medium_quality" - Adequate reasoning with minor issues
- "low_quality" - Unclear, illogical, or unsupported reasoning
"""


def run_adjudication_evaluation(
    traces_df,
    eval_model: str = "gpt-4o-mini",
    sample_size: Optional[int] = None
):
    """
    Run LLM-as-Judge evaluation on adjudication traces.

    Args:
        traces_df: DataFrame of traces from Phoenix
        eval_model: Model to use for evaluation
        sample_size: Number of traces to evaluate (None = all)

    Returns:
        DataFrame with evaluation results
    """
    try:
        from phoenix.evals import llm_classify, OpenAIModel
    except ImportError:
        raise RuntimeError("Phoenix evals not available. Install with: pip install arize-phoenix")

    # Filter to adjudication spans
    adj_traces = traces_df[traces_df['name'].str.contains('adjudicate', case=False, na=False)]

    if sample_size:
        adj_traces = adj_traces.head(sample_size)

    if adj_traces.empty:
        logger.warning("No adjudication traces found")
        return None

    # Create evaluation model
    model = OpenAIModel(model=eval_model)

    # Run accuracy evaluation
    accuracy_results = llm_classify(
        dataframe=adj_traces,
        model=model,
        template=ADJUDICATION_ACCURACY_TEMPLATE,
        rails=["correct", "incorrect", "uncertain"],
        provide_explanation=True
    )

    return accuracy_results


def run_hallucination_check(
    traces_df,
    eval_model: str = "gpt-4o-mini",
    sample_size: Optional[int] = None
):
    """
    Check for hallucinations in agent responses.

    Args:
        traces_df: DataFrame of traces from Phoenix
        eval_model: Model to use for evaluation
        sample_size: Number of traces to evaluate

    Returns:
        DataFrame with hallucination detection results
    """
    try:
        from phoenix.evals import llm_classify, OpenAIModel
    except ImportError:
        raise RuntimeError("Phoenix evals not available")

    if sample_size:
        traces_df = traces_df.head(sample_size)

    model = OpenAIModel(model=eval_model)

    results = llm_classify(
        dataframe=traces_df,
        model=model,
        template=HALLUCINATION_DETECTION_TEMPLATE,
        rails=["factual", "hallucinated"],
        provide_explanation=True
    )

    return results


def run_reasoning_quality_eval(
    traces_df,
    eval_model: str = "gpt-4o-mini",
    sample_size: Optional[int] = None
):
    """
    Evaluate the quality of reasoning in adjudications.

    Args:
        traces_df: DataFrame of traces from Phoenix
        eval_model: Model to use for evaluation
        sample_size: Number of traces to evaluate

    Returns:
        DataFrame with reasoning quality scores
    """
    try:
        from phoenix.evals import llm_classify, OpenAIModel
    except ImportError:
        raise RuntimeError("Phoenix evals not available")

    # Filter to adjudication spans
    adj_traces = traces_df[traces_df['name'].str.contains('adjudicate', case=False, na=False)]

    if sample_size:
        adj_traces = adj_traces.head(sample_size)

    if adj_traces.empty:
        logger.warning("No adjudication traces found")
        return None

    model = OpenAIModel(model=eval_model)

    results = llm_classify(
        dataframe=adj_traces,
        model=model,
        template=REASONING_QUALITY_TEMPLATE,
        rails=["high_quality", "medium_quality", "low_quality"],
        provide_explanation=True
    )

    return results


def print_evaluation_summary(eval_results, eval_name: str = "Evaluation"):
    """Print a summary of evaluation results."""
    if eval_results is None:
        print(f"\n{eval_name}: No results")
        return

    print(f"\n{'='*60}")
    print(f"{eval_name} Summary")
    print(f"{'='*60}")

    # Count labels
    if 'label' in eval_results.columns:
        label_counts = eval_results['label'].value_counts()
        total = len(eval_results)

        print(f"\nTotal evaluated: {total}")
        print("\nResults:")
        for label, count in label_counts.items():
            pct = (count / total) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    else:
        print("No labels found in results")


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_start():
    """
    Quick start Phoenix for the warranty claims agent.

    Usage:
        from src.phoenix_tracing import quick_start
        quick_start()

        # Then run your agent - traces will be captured automatically
    """
    print("Starting Arize Phoenix...")
    print("-" * 40)

    success = initialize_phoenix(
        project_name="warranty-claims",
        launch_app=True
    )

    if success:
        print("\nPhoenix is ready!")
        print("View traces at: http://localhost:6006")
        print("\nNow run your agent and traces will be captured automatically.")
    else:
        print("\nFailed to start Phoenix. Check the logs above for details.")

    return success


if __name__ == "__main__":
    # When run directly, start Phoenix
    quick_start()
