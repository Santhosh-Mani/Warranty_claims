"""
LangGraph Warranty Claims Agent.

Implements a stateful, cyclic workflow for processing warranty claims:
1. Triage - Classify email as spam/inquiry/claim
2. Extract - Pull structured data from email
3. Retrieve - Get relevant policy from ChromaDB
4. Adjudicate - Determine claim validity with CoT reasoning
5. Human Review - Pause for human approval (interrupt)
6. Fulfillment - Send response email and generate return label

Uses SQLite checkpointing for state persistence and HITL support.
"""

import sys
import json
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Literal, Annotated
import operator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    DATABASE_PATH,
    INBOX_DIR,
    OUTBOX_DIR,
    PRODUCT_CATALOG,
    PRODUCT_ID_TO_NAME,
    WARRANTY_WINDOW_DAYS,
    EXTENDED_WARRANTY_PRODUCTS,
    EXTENDED_WARRANTY_DAYS,
)
from src.models import (
    ClaimData,
    AdjudicationResult,
    WarrantyClaimState,
)
from src.ingest import PolicyIngester
from src.server import tools_server

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage

# LLM Provider imports
if LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
else:
    from langchain_ollama import ChatOllama

# Pydantic for structured output
from pydantic import BaseModel, Field

# Phoenix tracing (optional)
try:
    from src.phoenix_tracing import initialize_phoenix, PHOENIX_AVAILABLE
except ImportError:
    PHOENIX_AVAILABLE = False
    logger.info("Phoenix tracing module not available")


# ============================================================================
# Pydantic Models for LLM Structured Output
# ============================================================================

class TriageResult(BaseModel):
    """Result of email classification."""
    classification: Literal["spam", "inquiry", "claim"] = Field(
        description="The type of email: spam (marketing/junk), inquiry (question, not a claim), or claim (warranty claim request)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the classification"
    )
    reasoning: str = Field(
        description="Brief explanation for the classification"
    )


class ExtractionResult(BaseModel):
    """Extracted claim data from email."""
    customer_name: str = Field(description="Customer's full name")
    customer_email: Optional[str] = Field(None, description="Customer's email address")
    customer_address: Optional[str] = Field(None, description="Customer's mailing address")
    product_name: str = Field(description="Product name mentioned (e.g., 'AirFlow Pro')")
    serial_number: Optional[str] = Field(None, description="Product serial number if provided")
    purchase_date: Optional[str] = Field(None, description="Purchase date in YYYY-MM-DD format")
    issue_description: str = Field(description="Description of the problem/defect")
    has_proof_of_purchase: bool = Field(description="Whether proof of purchase was mentioned")


class AdjudicationOutput(BaseModel):
    """Structured adjudication output."""
    recommendation: Literal["APPROVE", "REJECT", "NEED_INFO"] = Field(
        description="The recommended action"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Confidence in the recommendation"
    )
    facts: list[str] = Field(description="Verified facts from the claim")
    assumptions: list[str] = Field(description="Assumptions made during analysis")
    reasoning: str = Field(description="Detailed reasoning for the recommendation")
    policy_references: list[str] = Field(description="Specific policy sections cited")
    warranty_valid: bool = Field(description="Whether warranty is still valid based on dates")
    exclusion_triggered: Optional[str] = Field(None, description="Which exclusion was triggered, if any")


# ============================================================================
# Agent State with Reducers
# ============================================================================

def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, right overwrites left."""
    return {**left, **right}


class AgentState(WarrantyClaimState):
    """Extended state with message accumulation."""
    messages: Annotated[list, operator.add] = []


# ============================================================================
# LLM Setup
# ============================================================================

def get_llm():
    """Get the LLM instance based on provider configuration."""
    if LLM_PROVIDER == "openai":
        logger.info(f"Using OpenAI with model: {OPENAI_MODEL}")
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0.1,  # Low temperature for consistent outputs
        )
    else:
        logger.info(f"Using Ollama with model: {OLLAMA_MODEL}")
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.1,  # Low temperature for consistent outputs
        )


# ============================================================================
# Node Functions
# ============================================================================

def triage_node(state: AgentState) -> dict:
    """
    Classify the incoming email as spam, inquiry, or claim.

    This is the first node in the workflow.
    """
    logger.info(f"[TRIAGE] Processing claim: {state.get('claim_id', 'unknown')}")

    email_body = state.get("email_body", "")
    email_subject = state.get("email_subject", "")

    llm = get_llm()

    prompt = f"""You are an email classifier for a hair dryer warranty department.

Classify this email into ONE of these categories:
- spam: Marketing, promotional content, unrelated topics, scams
- inquiry: Questions about products/policies (NOT reporting a defect)
- claim: Customer reporting a product defect and requesting warranty service

EMAIL SUBJECT: {email_subject}

EMAIL BODY:
{email_body}

Respond with JSON in this exact format:
{{
    "classification": "spam" or "inquiry" or "claim",
    "confidence": "high" or "medium" or "low",
    "reasoning": "brief explanation"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Parse JSON from response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())

        logger.info(f"[TRIAGE] Classification: {result['classification']} ({result['confidence']})")

        return {
            "classification": result["classification"],
            "messages": [f"Triage: {result['classification']} - {result['reasoning']}"]
        }

    except Exception as e:
        logger.error(f"[TRIAGE] Error: {e}")
        # Default to claim to be safe
        return {
            "classification": "claim",
            "messages": [f"Triage error, defaulting to claim: {str(e)}"]
        }


def extract_node(state: AgentState) -> dict:
    """
    Extract structured claim data from the email.

    Uses LLM to parse unstructured text into structured fields.
    """
    logger.info(f"[EXTRACT] Extracting data from claim: {state.get('claim_id', 'unknown')}")

    email_body = state.get("email_body", "")
    email_from = state.get("email_from", "")

    llm = get_llm()

    prompt = f"""You are a data extraction specialist for warranty claims.

Extract the following information from this warranty claim email.
If information is not provided, use null.
For dates, convert to YYYY-MM-DD format (today is {date.today().isoformat()}).

SENDER EMAIL: {email_from}

EMAIL CONTENT:
{email_body}

Known products (match to these if possible):
{json.dumps(list(PRODUCT_CATALOG.keys()), indent=2)}

Respond with JSON in this exact format:
{{
    "customer_name": "string",
    "customer_email": "string or null",
    "customer_address": "string or null",
    "product_name": "string (match to known products)",
    "serial_number": "string or null",
    "purchase_date": "YYYY-MM-DD or null",
    "issue_description": "string",
    "has_proof_of_purchase": true/false
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        extracted = json.loads(response_text.strip())

        # Map product name to ID
        product_name = extracted.get("product_name", "")
        product_id = PRODUCT_CATALOG.get(product_name, "")

        # If no exact match, try fuzzy match
        if not product_id:
            for name, pid in PRODUCT_CATALOG.items():
                if name.lower() in product_name.lower() or product_name.lower() in name.lower():
                    product_id = pid
                    extracted["product_name"] = name
                    break

        extracted["product_id"] = product_id

        # Use sender email if not in body
        if not extracted.get("customer_email"):
            extracted["customer_email"] = email_from

        logger.info(f"[EXTRACT] Product: {extracted.get('product_name')} ({product_id})")
        logger.info(f"[EXTRACT] Purchase date: {extracted.get('purchase_date')}")

        return {
            "extracted_data": extracted,
            "product_identified": product_id,
            "messages": [f"Extracted: {extracted.get('product_name')}, Serial: {extracted.get('serial_number')}"]
        }

    except Exception as e:
        logger.error(f"[EXTRACT] Error: {e}")
        return {
            "extracted_data": {"error": str(e)},
            "messages": [f"Extraction error: {str(e)}"]
        }


def validate_fields_node(state: AgentState) -> dict:
    """
    Validate that all required fields are present in extracted data.

    Required fields:
    - customer_name, customer_email, product_name (critical)
    - purchase_date (needed for warranty validation)
    - issue_description (critical)
    - customer_address (needed for return label)
    - serial_number (recommended)
    """
    logger.info(f"[VALIDATE] Checking required fields for: {state.get('claim_id', 'unknown')}")

    extracted_data = state.get("extracted_data", {})
    missing_fields = []

    # Critical fields
    if not extracted_data.get("customer_name"):
        missing_fields.append("customer_name")
    if not extracted_data.get("customer_email"):
        missing_fields.append("customer_email")
    if not extracted_data.get("product_name") or extracted_data.get("product_name") == "Unknown":
        missing_fields.append("product_name")
    if not extracted_data.get("issue_description"):
        missing_fields.append("issue_description")

    # Important for warranty validation
    if not extracted_data.get("purchase_date"):
        missing_fields.append("purchase_date")

    # Important for fulfillment
    if not extracted_data.get("customer_address"):
        missing_fields.append("customer_address")

    # Recommended
    if not extracted_data.get("serial_number"):
        missing_fields.append("serial_number")

    validation_complete = len(missing_fields) == 0

    if validation_complete:
        logger.info(f"[VALIDATE] All required fields present")
    else:
        logger.info(f"[VALIDATE] Missing fields: {', '.join(missing_fields)}")

    return {
        "missing_fields": missing_fields,
        "validation_complete": validation_complete,
        "messages": [
            f"Validation: {'Complete' if validation_complete else f'Missing {len(missing_fields)} fields'}"
        ]
    }


def collect_missing_fields_node(state: AgentState) -> dict:
    """
    Prepare for human input to collect missing fields.

    This node runs but the graph will INTERRUPT after it,
    waiting for human to provide missing data before proceeding.
    """
    logger.info(f"[COLLECT] Awaiting missing fields for: {state.get('claim_id', 'unknown')}")

    missing_fields = state.get("missing_fields", [])

    # The state is prepared - graph will interrupt here
    # Human will provide missing data via Streamlit or CLI
    return {
        "messages": [f"Awaiting missing fields: {', '.join(missing_fields)}"]
    }


def answer_inquiry_node(state: AgentState) -> dict:
    """
    Answer product inquiries using ChromaDB retrieval.

    This node handles customer questions about products by:
    1. Searching ChromaDB for relevant product information
    2. Using LLM to generate a helpful response
    3. Saving the response to outbox
    """
    logger.info(f"[INQUIRY] Answering product inquiry for: {state.get('claim_id', 'unknown')}")

    email_body = state.get("email_body", "")
    customer_email = state.get("email_from", "")
    claim_id = state.get("claim_id", "unknown")

    try:
        # Initialize ChromaDB collection
        from src.ingest import PolicyIngester
        ingester = PolicyIngester()
        collection = ingester.collection

        # Search for products related to the inquiry
        logger.info(f"[INQUIRY] Searching ChromaDB for: {email_body[:100]}...")
        results = collection.query(
            query_texts=[email_body],
            n_results=3,  # Get top 3 relevant product sections
            include=["documents", "metadatas"]
        )

        # Build context from retrieved documents
        context = ""
        products_mentioned = set()

        if results['documents'] and results['documents'][0]:
            for idx, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][idx]
                product_name = metadata.get('product_name', 'Unknown')
                products_mentioned.add(product_name)
                context += f"\n\n=== {product_name} Information ===\n{doc}"

        logger.info(f"[INQUIRY] Found information about: {', '.join(products_mentioned)}")

        # Generate helpful response using LLM
        prompt = f"""You are a helpful customer service agent for a hair dryer company.

Customer Question:
{email_body}

Relevant Product Information from our database:
{context}

Instructions:
- Answer the customer's question clearly and helpfully
- Reference specific product features from the information above
- Be friendly and professional
- If the information covers warranty, mention our warranty policies
- If the information doesn't fully answer their question, politely say so and offer to help further
- Keep the response concise (2-3 paragraphs)

Generate a helpful email response:"""

        # Call LLM
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, 'content') else str(response)

        logger.info(f"[INQUIRY] Generated response ({len(response_text)} chars)")

        # Save response to outbox
        response_email = {
            "to": customer_email,
            "subject": "Re: Product Inquiry",
            "body": response_text,
            "type": "inquiry_response"
        }

        # Write to outbox
        outbox_file = OUTBOX_DIR / f"inquiry_response_{claim_id}.txt"
        with open(outbox_file, 'w', encoding='utf-8') as f:
            f.write(f"To: {customer_email}\n")
            f.write(f"Subject: Re: Product Inquiry\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            f.write(response_text)

        logger.info(f"[INQUIRY] Response saved to: {outbox_file}")

        return {
            "response_email": response_email,
            "messages": [f"Inquiry answered and sent to {customer_email}"]
        }

    except Exception as e:
        logger.error(f"[INQUIRY] Error: {e}")

        # Fallback response
        fallback_text = f"""Thank you for your inquiry about our hair dryer products.

I apologize, but I'm currently unable to retrieve detailed product information.

Please feel free to:
- Visit our website for product specifications
- Contact our customer service team directly
- Reply to this email with your specific questions

We appreciate your interest in our products!

Best regards,
Customer Service Team"""

        outbox_file = OUTBOX_DIR / f"inquiry_response_{claim_id}.txt"
        with open(outbox_file, 'w', encoding='utf-8') as f:
            f.write(f"To: {customer_email}\n")
            f.write(f"Subject: Re: Product Inquiry\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            f.write(fallback_text)

        return {
            "response_email": {"to": customer_email, "subject": "Re: Product Inquiry", "body": fallback_text},
            "messages": [f"Inquiry answered (fallback) - Error: {str(e)}"]
        }


def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve the relevant warranty policy from ChromaDB.
    """
    logger.info(f"[RETRIEVE] Getting policy for: {state.get('product_identified', 'unknown')}")

    product_id = state.get("product_identified", "")
    extracted_data = state.get("extracted_data", {})
    issue = extracted_data.get("issue_description", "warranty claim")

    try:
        # Initialize policy ingester
        ingester = PolicyIngester()

        # Get policy for this product
        if product_id:
            policy_text = ingester.get_policy_for_product(product_id)
        else:
            # Query based on issue description
            results = ingester.query(issue, n_results=3)
            policy_text = "\n\n".join([r["content"] for r in results])

        logger.info(f"[RETRIEVE] Retrieved policy ({len(policy_text)} chars)")

        return {
            "policy_text": policy_text,
            "messages": [f"Retrieved policy for product {product_id}"]
        }

    except Exception as e:
        logger.error(f"[RETRIEVE] Error: {e}")
        return {
            "policy_text": "Policy not found. Please review manually.",
            "messages": [f"Policy retrieval error: {str(e)}"]
        }


def adjudicate_node(state: AgentState) -> dict:
    """
    Evaluate the claim against the warranty policy.

    Uses Chain-of-Thought reasoning to separate facts, assumptions, and reasoning.
    """
    logger.info(f"[ADJUDICATE] Evaluating claim: {state.get('claim_id', 'unknown')}")

    extracted_data = state.get("extracted_data", {})
    policy_text = state.get("policy_text", "")
    product_id = state.get("product_identified", "")

    # Calculate warranty status
    purchase_date_str = extracted_data.get("purchase_date")
    warranty_days = EXTENDED_WARRANTY_DAYS if product_id in EXTENDED_WARRANTY_PRODUCTS else WARRANTY_WINDOW_DAYS

    days_since_purchase = None
    warranty_valid = None
    if purchase_date_str:
        try:
            purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d").date()
            days_since_purchase = (date.today() - purchase_date).days
            warranty_valid = days_since_purchase <= warranty_days
        except (ValueError, TypeError) as e:
            logger.warning(f"[ADJUDICATE] Invalid date format: {purchase_date_str} - {e}")

    llm = get_llm()

    prompt = f"""You are a Warranty Adjudication Officer. Evaluate this warranty claim.

CLAIM DATA:
{json.dumps(extracted_data, indent=2, default=str)}

WARRANTY POLICY:
{policy_text[:3000]}  # Truncate to fit context

TODAY'S DATE: {date.today().isoformat()}
WARRANTY PERIOD: {warranty_days} days
DAYS SINCE PURCHASE: {days_since_purchase if days_since_purchase else "Unknown"}
WARRANTY STILL VALID: {warranty_valid if warranty_valid is not None else "Unknown"}

Analyze this claim step by step:

1. FACTS: List only verifiable data points from the claim
2. ASSUMPTIONS: List inferences you're making
3. POLICY CHECK:
   - WARRANTY WINDOW: Is the claim within {warranty_days} days?
   - EXCLUSION CHECK: Does the issue match any exclusions in the policy?
   - COVERAGE CHECK: Is this type of issue covered?
4. REASONING: Synthesize the above
5. RECOMMENDATION: APPROVE, REJECT, or NEED_INFO

Respond with JSON:
{{
    "recommendation": "APPROVE" or "REJECT" or "NEED_INFO",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "facts": ["fact1", "fact2"],
    "assumptions": ["assumption1", "assumption2"],
    "reasoning": "detailed reasoning",
    "policy_references": ["relevant policy sections"],
    "warranty_valid": true/false,
    "exclusion_triggered": "exclusion name or null"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        adjudication = json.loads(response_text.strip())

        # Add calculated values if not present
        if days_since_purchase is not None:
            adjudication["days_since_purchase"] = days_since_purchase
        if warranty_valid is not None and "warranty_valid" not in adjudication:
            adjudication["warranty_valid"] = warranty_valid

        logger.info(f"[ADJUDICATE] Recommendation: {adjudication['recommendation']} ({adjudication['confidence']})")

        return {
            "adjudication_result": adjudication,
            "reasoning_trace": adjudication.get("reasoning", ""),
            "messages": [f"Adjudication: {adjudication['recommendation']} - {adjudication.get('reasoning', '')[:100]}..."]
        }

    except Exception as e:
        logger.error(f"[ADJUDICATE] Error: {e}")
        return {
            "adjudication_result": {
                "recommendation": "NEED_INFO",
                "confidence": "LOW",
                "reasoning": f"Error during adjudication: {str(e)}",
                "facts": [],
                "assumptions": [],
                "policy_references": []
            },
            "messages": [f"Adjudication error: {str(e)}"]
        }


def human_review_node(state: AgentState) -> dict:
    """
    Prepare the review packet for human review.

    This node runs but the graph will INTERRUPT after it,
    waiting for human decision before proceeding to fulfillment.
    """
    logger.info(f"[HUMAN_REVIEW] Preparing review packet for: {state.get('claim_id', 'unknown')}")

    # The state is already complete - just add a message
    return {
        "messages": ["Awaiting human review..."]
    }


def fulfillment_node(state: AgentState) -> dict:
    """
    Execute the final action based on human decision.

    - If APPROVE: Generate return label and approval email
    - If REJECT: Generate rejection email with reasons
    - If NEED_INFO: Generate info request email
    """
    logger.info(f"[FULFILLMENT] Processing decision for: {state.get('claim_id', 'unknown')}")

    claim_id = state.get("claim_id", "UNKNOWN")
    human_decision = state.get("human_decision", "").lower()
    extracted_data = state.get("extracted_data", {})
    adjudication = state.get("adjudication_result", {})

    customer_email = extracted_data.get("customer_email", "customer@example.com")
    customer_name = extracted_data.get("customer_name", "Customer")
    customer_address = extracted_data.get("customer_address", "Address not provided")
    product_name = extracted_data.get("product_name", "Product")
    serial_number = extracted_data.get("serial_number", "N/A")

    response_email = None
    return_label_path = None

    if human_decision == "approve":
        # Generate return label
        label_result = tools_server.generate_return_label(
            claim_id=claim_id,
            customer_name=customer_name,
            customer_address=customer_address,
            product_name=product_name,
            serial_number=serial_number
        )
        return_label_path = label_result.get("label_path")

        # Send approval email
        email_body = f"""Dear {customer_name},

Great news! Your warranty claim (ID: {claim_id}) has been APPROVED.

Product: {product_name}
Serial Number: {serial_number}

Next Steps:
1. A prepaid return shipping label is attached to this email
2. Package your product securely with all accessories
3. Attach the label to the outside of the package
4. Drop off at any UPS location or schedule a pickup
5. Once we receive your product, a replacement will be shipped within 7-10 business days

If you have any questions, please reply to this email with your claim ID.

Thank you for choosing HairDryer Co.

Best regards,
HairDryer Co. Warranty Team
"""
        email_result = tools_server.send_email(
            to=customer_email,
            subject=f"Warranty Claim {claim_id} - APPROVED",
            body=email_body,
            attachments=[return_label_path] if return_label_path else None
        )

        response_email = {
            "to": customer_email,
            "subject": f"Warranty Claim {claim_id} - APPROVED",
            "body": email_body,
            "is_approval": True
        }

        logger.info(f"[FULFILLMENT] Approval email sent, label generated")

    elif human_decision == "reject":
        # Generate rejection email
        rejection_reason = adjudication.get("reasoning", "Your claim does not meet our warranty requirements.")
        exclusion = adjudication.get("exclusion_triggered")
        policy_refs = adjudication.get("policy_references", [])

        email_body = f"""Dear {customer_name},

Thank you for contacting us regarding your warranty claim (ID: {claim_id}).

After careful review, we regret to inform you that your claim has been DENIED.

Product: {product_name}
Serial Number: {serial_number}

Reason for Denial:
{rejection_reason}

{"Policy Reference: " + exclusion if exclusion else ""}
{chr(10).join(policy_refs) if policy_refs else ""}

What You Can Do:
- Review our warranty terms at www.hairdryer-co.com/warranty
- If you believe this decision is in error, reply to this email with additional documentation
- Consider our out-of-warranty repair service at a reduced cost

We value your business and apologize for any inconvenience.

Best regards,
HairDryer Co. Warranty Team
"""
        email_result = tools_server.send_email(
            to=customer_email,
            subject=f"Warranty Claim {claim_id} - DENIED",
            body=email_body
        )

        response_email = {
            "to": customer_email,
            "subject": f"Warranty Claim {claim_id} - DENIED",
            "body": email_body,
            "is_approval": False
        }

        logger.info(f"[FULFILLMENT] Rejection email sent")

    else:  # need_info or unknown
        # Request more information
        missing = []
        if not extracted_data.get("purchase_date"):
            missing.append("purchase date")
        if not extracted_data.get("serial_number"):
            missing.append("serial number")
        if not extracted_data.get("product_name") or extracted_data.get("product_name") == "Unknown":
            missing.append("specific product model")

        email_body = f"""Dear {customer_name},

Thank you for contacting us regarding your warranty claim (ID: {claim_id}).

To process your claim, we need additional information:

Missing Information:
{chr(10).join(f"- {item}" for item in missing) if missing else "- Please provide more details about your issue"}

Please reply to this email with:
1. Your product's serial number (found on the label on the device)
2. Your purchase date and proof of purchase (receipt or order confirmation)
3. A detailed description of the issue you're experiencing

Once we receive this information, we'll process your claim promptly.

Best regards,
HairDryer Co. Warranty Team
"""
        email_result = tools_server.send_email(
            to=customer_email,
            subject=f"Warranty Claim {claim_id} - Information Needed",
            body=email_body
        )

        response_email = {
            "to": customer_email,
            "subject": f"Warranty Claim {claim_id} - Information Needed",
            "body": email_body,
            "is_approval": False
        }

        logger.info(f"[FULFILLMENT] Info request email sent")

    return {
        "response_email": response_email,
        "return_label_path": return_label_path,
        "messages": [f"Fulfillment complete: {human_decision}"]
    }


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_triage(state: AgentState) -> str:
    """Route based on triage classification."""
    classification = state.get("classification", "claim")

    if classification == "claim":
        logger.info(f"[ROUTE] Routing to extract - warranty claim detected")
        return "extract"
    elif classification == "inquiry":
        logger.info(f"[ROUTE] Routing to answer_inquiry - product question detected")
        return "answer_inquiry"
    else:
        # Spam - end the workflow
        logger.info(f"[ROUTE] Ending workflow - spam detected")
        return END


def route_after_validation(state: AgentState) -> str:
    """Route based on field validation status."""
    validation_complete = state.get("validation_complete", False)
    if validation_complete:
        logger.info(f"[ROUTE] All fields present - proceeding to retrieve")
        return "retrieve"
    else:
        missing = state.get("missing_fields", [])
        logger.info(f"[ROUTE] Missing fields: {missing} - collecting data")
        return "collect_missing_fields"


# ============================================================================
# Graph Construction
# ============================================================================

def create_warranty_graph():
    """Create the LangGraph workflow with field validation and inquiry handling."""
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("triage", triage_node)
    graph.add_node("answer_inquiry", answer_inquiry_node)  # NEW: Handle product inquiries
    graph.add_node("extract", extract_node)
    graph.add_node("validate_fields", validate_fields_node)
    graph.add_node("collect_missing_fields", collect_missing_fields_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("adjudicate", adjudicate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("fulfillment", fulfillment_node)

    # Add edges
    graph.add_edge(START, "triage")

    # Route after triage: claim -> extract, inquiry -> answer_inquiry, spam -> END
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        ["extract", "answer_inquiry", END]
    )

    # Inquiry path: answer and end
    graph.add_edge("answer_inquiry", END)

    # Claim path: extract and validate
    graph.add_edge("extract", "validate_fields")  # Always validate after extraction

    # Conditional routing after validation
    graph.add_conditional_edges(
        "validate_fields",
        route_after_validation,
        ["retrieve", "collect_missing_fields"]
    )

    # After collecting missing fields, go back to validation to re-check
    graph.add_edge("collect_missing_fields", "validate_fields")

    # Continue with normal flow after validation passes
    graph.add_edge("retrieve", "adjudicate")
    graph.add_edge("adjudicate", "human_review")
    graph.add_edge("human_review", "fulfillment")
    graph.add_edge("fulfillment", END)

    return graph


def get_compiled_graph(checkpointer=None, interrupt_before=None):
    """Get a compiled graph with optional checkpointer and interrupts."""
    graph = create_warranty_graph()

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    return graph.compile(**compile_kwargs)


# ============================================================================
# Main Execution
# ============================================================================

def process_claim_file(filepath: Path, app, thread_id: str = None) -> dict:
    """Process a single claim file through the agent."""
    # Load claim data
    with open(filepath, 'r', encoding='utf-8') as f:
        claim_data = json.load(f)

    email = claim_data.get("email", claim_data)

    # Generate claim ID if not provided
    claim_id = claim_data.get("id", f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    thread_id = thread_id or str(uuid.uuid4())

    # Prepare initial state
    initial_state = {
        "claim_id": claim_id,
        "email_source": str(filepath),
        "timestamp": datetime.now().isoformat(),
        "email_body": email.get("body", ""),
        "email_from": email.get("from", ""),
        "email_subject": email.get("subject", ""),
        "attachments": email.get("attachments", []),
        "messages": [f"Processing claim from: {filepath.name}"]
    }

    # Run the graph
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state, config)

    return {
        "thread_id": thread_id,
        "claim_id": claim_id,
        "result": result
    }


def process_manual_claim(claim_text: str, app, thread_id: str = None) -> dict:
    """Process a manually entered claim text through the agent."""
    # Generate claim ID
    claim_id = f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    thread_id = thread_id or str(uuid.uuid4())

    # Prepare initial state with manual entry flag
    initial_state = {
        "claim_id": claim_id,
        "email_source": "manual_entry",
        "timestamp": datetime.now().isoformat(),
        "email_body": claim_text,
        "email_from": "",  # Will be extracted from text
        "email_subject": "Manual Claim Entry",
        "attachments": [],
        "manual_entry_mode": True,
        "messages": [f"Processing manual claim: {claim_id}"]
    }

    # Run the graph
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state, config)

    return {
        "thread_id": thread_id,
        "claim_id": claim_id,
        "result": result
    }


def main():
    """Main entry point for the warranty claims agent."""
    parser = argparse.ArgumentParser(description="Warranty Claims Agent")
    parser.add_argument("--file", type=Path, help="Process a single claim file")
    parser.add_argument("--text", type=str, help="Process manual claim text")
    parser.add_argument("--watch", action="store_true", help="Watch inbox directory for new files")
    parser.add_argument("--no-interrupt", action="store_true", help="Run without HITL interrupts")
    parser.add_argument("--decision", type=str, choices=["approve", "reject", "need_info"],
                       help="Auto-decision for testing (skips human review)")
    parser.add_argument("--resume", type=str, help="Resume a paused claim by thread_id")
    parser.add_argument("--update-fields", type=str, help="Update missing fields (JSON format)")
    args = parser.parse_args()

    # Setup Phoenix tracing if available
    if PHOENIX_AVAILABLE:
        try:
            initialize_phoenix(project_name="warranty-claims", launch_app=True)
        except Exception as e:
            logger.warning(f"Phoenix tracing setup failed: {e}")

    # Create checkpointer
    import sqlite3
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Create compiled graph with interrupts for HITL
    interrupt_before = None if args.no_interrupt else ["collect_missing_fields", "fulfillment"]
    app = get_compiled_graph(checkpointer=checkpointer, interrupt_before=interrupt_before)

    if args.resume:
        # Resume a paused claim
        logger.info(f"Resuming thread: {args.resume}")
        config = {"configurable": {"thread_id": args.resume}}

        # Update state with decision or missing fields
        if args.decision:
            print(f"[RESUME] Applying decision: {args.decision}")
            app.update_state(config, {"human_decision": args.decision})
        elif args.update_fields:
            # Parse JSON with missing fields
            try:
                updated_fields = json.loads(args.update_fields)
                print(f"[RESUME] Updating fields: {list(updated_fields.keys())}")

                # Get current state
                current_state = app.get_state(config)
                extracted_data = current_state.values.get("extracted_data", {})

                # Merge updated fields
                extracted_data.update(updated_fields)

                # Update state
                app.update_state(config, {"extracted_data": extracted_data})
            except Exception as e:
                logger.error(f"Error parsing update-fields JSON: {e}")
                return

        # Continue execution
        final_result = app.invoke(None, config)
        print(f"\n[RESUME] Processing complete!")
        print(json.dumps(final_result, indent=2, default=str))

    elif args.text:
        # Process manual text claim
        logger.info(f"Processing manual claim text")
        result = process_manual_claim(args.text, app)

        print("\n" + "=" * 60)
        print("MANUAL CLAIM PROCESSING RESULT")
        print("=" * 60)
        print(f"Thread ID: {result['thread_id']}")
        print(f"Claim ID: {result['claim_id']}")
        print(f"\nClassification: {result['result'].get('classification', 'N/A')}")

        if result['result'].get('extracted_data'):
            print(f"\nExtracted Data:")
            print(json.dumps(result['result']['extracted_data'], indent=2))

        if result['result'].get('missing_fields'):
            print(f"\n⚠️  MISSING FIELDS: {', '.join(result['result']['missing_fields'])}")
            print(f"\n[PAUSED] Please provide missing fields")
            print(f"Resume with:")
            print(f"  python src/agent.py --resume {result['thread_id']} --update-fields '{{\"field\": \"value\"}}'")
            return

        if result['result'].get('adjudication_result'):
            adj = result['result']['adjudication_result']
            print(f"\nAdjudication:")
            print(f"  Recommendation: {adj.get('recommendation')}")
            print(f"  Confidence: {adj.get('confidence')}")
            print(f"  Reasoning: {adj.get('reasoning', '')[:200]}...")

        # Handle auto-decision for testing
        if args.decision and not args.no_interrupt:
            print(f"\n[AUTO-DECISION] Applying decision: {args.decision}")
            config = {"configurable": {"thread_id": result['thread_id']}}
            app.update_state(config, {"human_decision": args.decision})
            final_result = app.invoke(None, config)
            print(f"\nFulfillment complete!")
            if final_result.get('return_label_path'):
                print(f"Return label: {final_result['return_label_path']}")

        elif not args.no_interrupt:
            print(f"\n[PAUSED] Waiting for human review")
            print(f"Use Streamlit dashboard or resume with:")
            print(f"  python src/agent.py --resume {result['thread_id']} --decision approve")

    elif args.file:
        # Process single file
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            return

        logger.info(f"Processing file: {args.file}")
        result = process_claim_file(args.file, app)

        print("\n" + "=" * 60)
        print("CLAIM PROCESSING RESULT")
        print("=" * 60)
        print(f"Thread ID: {result['thread_id']}")
        print(f"Claim ID: {result['claim_id']}")
        print(f"\nClassification: {result['result'].get('classification', 'N/A')}")

        if result['result'].get('extracted_data'):
            print(f"\nExtracted Data:")
            print(json.dumps(result['result']['extracted_data'], indent=2))

        if result['result'].get('missing_fields'):
            print(f"\n⚠️  MISSING FIELDS: {', '.join(result['result']['missing_fields'])}")
            print(f"\n[PAUSED] Please provide missing fields")
            print(f"Resume with:")
            print(f"  python src/agent.py --resume {result['thread_id']} --update-fields '{{\"field\": \"value\"}}'")
            return

        if result['result'].get('adjudication_result'):
            adj = result['result']['adjudication_result']
            print(f"\nAdjudication:")
            print(f"  Recommendation: {adj.get('recommendation')}")
            print(f"  Confidence: {adj.get('confidence')}")
            print(f"  Reasoning: {adj.get('reasoning', '')[:200]}...")

        # Handle auto-decision for testing
        if args.decision and not args.no_interrupt:
            print(f"\n[AUTO-DECISION] Applying decision: {args.decision}")
            config = {"configurable": {"thread_id": result['thread_id']}}
            app.update_state(config, {"human_decision": args.decision})
            final_result = app.invoke(None, config)
            print(f"\nFulfillment complete!")
            if final_result.get('return_label_path'):
                print(f"Return label: {final_result['return_label_path']}")

        elif not args.no_interrupt:
            print(f"\n[PAUSED] Waiting for human review")
            print(f"Use Streamlit dashboard or resume with:")
            print(f"  python src/agent.py --resume {result['thread_id']} --decision approve")

    elif args.watch:
        # Watch mode
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class ClaimHandler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                if event.src_path.endswith('.json'):
                    logger.info(f"New claim detected: {event.src_path}")
                    try:
                        result = process_claim_file(Path(event.src_path), app)
                        logger.info(f"Processed: {result['claim_id']}")
                    except Exception as e:
                        logger.error(f"Error processing {event.src_path}: {e}")

        observer = Observer()
        observer.schedule(ClaimHandler(), str(INBOX_DIR), recursive=False)
        observer.start()

        logger.info(f"Watching {INBOX_DIR} for new claims...")
        logger.info("Press Ctrl+C to stop")

        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    else:
        # Demo mode
        print("=" * 60)
        print("WARRANTY CLAIMS AGENT")
        print("=" * 60)
        print("\nUsage:")
        print("  python src/agent.py --file <claim.json>           # Process claim file")
        print("  python src/agent.py --text \"claim text here\"      # Process manual text")
        print("  python src/agent.py --watch                       # Watch inbox for new claims")
        print("  python src/agent.py --resume <thread_id> --decision approve  # Resume and approve")
        print("  python src/agent.py --resume <thread_id> --update-fields '{\"field\": \"value\"}'  # Provide missing fields")
        print("\nFor human review and manual entry, use the Streamlit dashboard:")
        print("  streamlit run src/app.py")


if __name__ == "__main__":
    main()
