"""
Pydantic models for the Warranty Claims Agent.

These models define the data structures used throughout the system,
ensuring type safety and validation for claim data, adjudication results,
and the agent state.
"""

from datetime import date, datetime
from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class EmailData(BaseModel):
    """Represents an incoming email."""

    from_address: str = Field(..., alias="from", description="Sender email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    received_at: datetime = Field(default_factory=datetime.now)
    attachments: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class ClaimData(BaseModel):
    """Extracted warranty claim data from an email."""

    customer_name: str = Field(..., description="Customer's full name")
    customer_email: Optional[str] = Field(None, description="Customer's email address")
    customer_address: Optional[str] = Field(None, description="Customer's mailing address")
    product_name: str = Field(..., description="Name of the product (e.g., 'AirFlow Pro')")
    product_id: Optional[str] = Field(None, description="Product ID (e.g., 'HD-002')")
    serial_number: Optional[str] = Field(None, description="Product serial number")
    purchase_date: Optional[date] = Field(None, description="Date of purchase")
    issue_description: str = Field(..., description="Description of the issue/defect")
    has_proof_of_purchase: bool = Field(
        False, description="Whether proof of purchase was mentioned"
    )
    attachments_mentioned: List[str] = Field(
        default_factory=list, description="Attachments mentioned in email"
    )


class AdjudicationResult(BaseModel):
    """Result of the warranty claim adjudication."""

    recommendation: Literal["APPROVE", "REJECT", "NEED_INFO"] = Field(
        ..., description="Recommended action"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Confidence level in the recommendation"
    )
    facts: List[str] = Field(
        default_factory=list, description="Verified facts from the claim"
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions made during analysis"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the recommendation")
    policy_references: List[str] = Field(
        default_factory=list, description="Relevant policy sections cited"
    )
    warranty_valid: Optional[bool] = Field(
        None, description="Whether warranty is still valid"
    )
    exclusion_triggered: Optional[str] = Field(
        None, description="Which exclusion was triggered, if any"
    )
    days_since_purchase: Optional[int] = Field(
        None, description="Days between purchase and claim"
    )


class ReviewPacket(BaseModel):
    """Complete review packet for human reviewer."""

    claim_id: str
    email_data: EmailData
    extracted_data: Optional[ClaimData]
    matched_policy: Optional[str]
    policy_text: Optional[str]
    adjudication: Optional[AdjudicationResult]
    created_at: datetime = Field(default_factory=datetime.now)


class ResponseEmail(BaseModel):
    """Generated response email to customer."""

    to: str
    subject: str
    body: str
    is_approval: bool
    return_label_path: Optional[str] = None


# LangGraph State Definition
class WarrantyClaimState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    # Metadata
    claim_id: str
    email_source: str
    timestamp: str

    # Raw Input
    email_body: str
    email_from: str
    email_subject: str
    attachments: List[str]

    # Processing Results
    classification: str  # 'spam' | 'inquiry' | 'claim'
    extracted_data: Optional[dict]  # Serialized ClaimData
    product_identified: str
    policy_text: str

    # Field Validation (for manual entry)
    missing_fields: Optional[List[str]]  # List of missing required fields
    validation_complete: bool  # Whether all required fields are present
    manual_entry_mode: bool  # Whether this is a manual text entry

    # Adjudication
    adjudication_result: Optional[dict]  # Serialized AdjudicationResult
    reasoning_trace: str

    # Human Interaction
    human_decision: Optional[str]  # 'approve' | 'reject' | 'need_info'
    human_notes: Optional[str]

    # Output
    response_email: Optional[dict]  # Serialized ResponseEmail
    return_label_path: Optional[str]

    # Error tracking
    error: Optional[str]


# Ground Truth for Evaluation
class GroundTruth(BaseModel):
    """Ground truth data for test evaluation."""

    id: str
    type: str  # 'spam', 'inquiry', 'valid_claim', 'invalid_expired', etc.
    expected_classification: Optional[str] = None
    expected_recommendation: Optional[str] = None
    ground_truth: Optional[dict] = None


class TestCase(BaseModel):
    """Complete test case with email and ground truth."""

    id: str
    type: str
    expected_classification: Optional[str] = None
    expected_recommendation: Optional[str] = None
    ground_truth: Optional[dict] = None
    email: EmailData
