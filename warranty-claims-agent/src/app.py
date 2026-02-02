"""
Streamlit Dashboard for Human-in-the-Loop Review.

This dashboard allows human reviewers to:
1. View pending warranty claims
2. See extracted data and AI reasoning
3. Approve or reject claims
4. Resume the LangGraph workflow after decision

The app queries SQLite for interrupted threads and provides
controls to resume the agent workflow.
"""

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.config import DATABASE_PATH, PRODUCT_ID_TO_NAME
from src.agent import get_compiled_graph

# LangGraph checkpoint imports
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Warranty Claims Review Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Styling
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .claim-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .approve-btn {
        background-color: #28a745;
        color: white;
    }
    .reject-btn {
        background-color: #dc3545;
        color: white;
    }
    .fact-box {
        background-color: #e7f3ff;
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .assumption-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-left: 3px solid #ff9800;
        margin: 0.5rem 0;
    }
    .reasoning-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def get_checkpointer():
    """Get the SQLite checkpointer instance."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    return SqliteSaver(conn)


@st.cache_resource
def get_graph():
    """Get the compiled graph with checkpointer."""
    checkpointer = get_checkpointer()
    return get_compiled_graph(checkpointer=checkpointer, interrupt_before=["collect_missing_fields", "fulfillment"])


def get_pending_claims():
    """
    Get all claims that are pending human review.

    These are threads that have been interrupted before the fulfillment node.
    """
    checkpointer = get_checkpointer()
    pending = []

    try:
        # Query the checkpoints table directly
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()

        # Get all thread IDs (ordered by checkpoint_id which contains timestamp)
        cursor.execute("""
            SELECT DISTINCT thread_id FROM checkpoints
            ORDER BY checkpoint_id DESC
        """)

        thread_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Get state for each thread
        app = get_graph()
        for thread_id in thread_ids:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                state = app.get_state(config)

                if state and state.values:
                    # Check if this is a pending claim (has adjudication but no human_decision)
                    values = state.values
                    if (values.get("adjudication_result") and
                        not values.get("human_decision") and
                        values.get("classification") == "claim"):
                        pending.append({
                            "thread_id": thread_id,
                            "claim_id": values.get("claim_id", "Unknown"),
                            "timestamp": values.get("timestamp", ""),
                            "state": values
                        })
            except Exception as e:
                st.warning(f"Error reading thread {thread_id}: {e}")

    except Exception as e:
        st.error(f"Database error: {e}")

    return pending


def resume_claim(thread_id: str, decision: str, notes: str = ""):
    """
    Resume a claim with the human decision.

    Updates the state with the decision and invokes the graph to continue.
    """
    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Update state with human decision
        app.update_state(config, {
            "human_decision": decision,
            "human_notes": notes
        })

        # Resume the graph
        result = app.invoke(None, config)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def process_manual_claim(claim_text: str):
    """
    Process a manually entered claim text.
    """
    from src.agent import process_manual_claim as agent_process_manual
    import uuid

    app = get_graph()
    thread_id = str(uuid.uuid4())

    try:
        result = agent_process_manual(claim_text, app, thread_id)
        return {
            "success": True,
            "thread_id": thread_id,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def update_missing_fields(thread_id: str, updated_fields: dict):
    """
    Update the state with missing fields provided by user.
    """
    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Get current state
        state = app.get_state(config)
        extracted_data = state.values.get("extracted_data", {})

        # Merge updated fields
        extracted_data.update(updated_fields)

        # Update state
        app.update_state(config, {"extracted_data": extracted_data})

        # Resume the graph
        result = app.invoke(None, config)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_claims_with_missing_fields():
    """
    Get all claims that are waiting for missing field input.
    """
    checkpointer = get_checkpointer()
    missing_field_claims = []

    try:
        # Query the checkpoints table directly
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()

        # Get all thread IDs
        cursor.execute("""
            SELECT DISTINCT thread_id FROM checkpoints
            ORDER BY checkpoint_id DESC
        """)

        thread_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Get state for each thread
        app = get_graph()
        for thread_id in thread_ids:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                state = app.get_state(config)

                if state and state.values:
                    values = state.values
                    # Check if this claim has missing fields and no decision yet
                    if (values.get("missing_fields") and
                        not values.get("validation_complete") and
                        not values.get("human_decision")):
                        missing_field_claims.append({
                            "thread_id": thread_id,
                            "claim_id": values.get("claim_id", "Unknown"),
                            "timestamp": values.get("timestamp", ""),
                            "state": values
                        })
            except Exception as e:
                pass

    except Exception as e:
        st.error(f"Database error: {e}")

    return missing_field_claims


def get_inquiries():
    """
    Get all inquiries that have been processed.

    Returns inquiries with their responses.
    """
    inquiries = []

    try:
        # Query the checkpoints table directly
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()

        # Get all thread IDs
        cursor.execute("""
            SELECT DISTINCT thread_id FROM checkpoints
            ORDER BY checkpoint_id DESC
            LIMIT 50
        """)

        thread_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Get state for each thread
        app = get_graph()
        for thread_id in thread_ids:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                state = app.get_state(config)

                if state and state.values:
                    values = state.values
                    # Check if this is an inquiry (classification = inquiry)
                    if values.get("classification") == "inquiry":
                        inquiries.append({
                            "thread_id": thread_id,
                            "claim_id": values.get("claim_id", "Unknown"),
                            "timestamp": values.get("timestamp", ""),
                            "email_from": values.get("email_from", "Unknown"),
                            "email_subject": values.get("email_subject", ""),
                            "email_body": values.get("email_body", ""),
                            "response_email": values.get("response_email", {}),
                            "state": values
                        })
            except Exception as e:
                pass

    except Exception as e:
        st.error(f"Database error: {e}")

    return inquiries


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## Navigation")

    page = st.radio(
        "Select Page",
        ["Manual Entry", "Pending Claims", "Inquiries", "All Claims", "Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### Quick Stats")
    pending_claims = get_pending_claims()
    inquiries = get_inquiries()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pending", len(pending_claims))
    with col2:
        st.metric("Inquiries", len(inquiries))

    st.markdown("---")

    st.markdown("### System Status")
    st.success("Agent: Online")
    st.info(f"Database: {DATABASE_PATH.name}")

    # Refresh button
    if st.button("🔄 Refresh"):
        st.cache_resource.clear()
        st.rerun()


# ============================================================================
# Main Content
# ============================================================================

st.markdown('<p class="main-header">📋 Warranty Claims Review Dashboard</p>', unsafe_allow_html=True)

if page == "Manual Entry":
    st.markdown("### Manual Claim Entry")
    st.markdown("Enter warranty claim text below. The system will extract information and validate required fields.")

    # Tab for new entry vs providing missing fields
    entry_tab, missing_tab = st.tabs(["➕ New Claim", "📝 Provide Missing Fields"])

    with entry_tab:
        st.markdown("#### Submit New Claim")

        claim_text = st.text_area(
            "Claim Text",
            height=200,
            placeholder="""Example:
Hi, my name is John Smith (john@email.com).
I purchased an AirFlow Pro hair dryer on 2024-10-15.
The serial number is HD002-2024-1234.
The heating element stopped working after just 2 weeks.
I have the receipt as proof of purchase.
My address is 123 Main St, New York, NY 10001.
Please help with my warranty claim."""
        )

        if st.button("🚀 Submit Claim", type="primary"):
            if claim_text.strip():
                with st.spinner("Processing claim..."):
                    result = process_manual_claim(claim_text)

                    if result["success"]:
                        res_data = result["result"]["result"]

                        # Check if missing fields
                        if res_data.get("missing_fields"):
                            st.warning(f"⚠️ Missing required fields: {', '.join(res_data['missing_fields'])}")
                            st.info(f"Thread ID: {result['thread_id']}")
                            st.info("Please go to the 'Provide Missing Fields' tab to complete the claim.")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.success("✅ Claim submitted successfully!")
                            st.info(f"Claim ID: {res_data.get('claim_id')}")
                            st.info("Your claim is now pending review. Check the 'Pending Claims' page.")
                            st.cache_resource.clear()
                            st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")
            else:
                st.warning("Please enter claim text")

    with missing_tab:
        st.markdown("#### Provide Missing Fields")

        # Get claims with missing fields
        missing_field_claims = get_claims_with_missing_fields()

        if not missing_field_claims:
            st.info("No claims waiting for missing fields.")
        else:
            for claim in missing_field_claims:
                thread_id = claim["thread_id"]
                claim_id = claim["claim_id"]
                state = claim["state"]
                missing_fields = state.get("missing_fields", [])
                extracted_data = state.get("extracted_data", {})

                with st.expander(f"📋 Claim: {claim_id}", expanded=True):
                    st.markdown(f"**Missing Fields:** {', '.join(missing_fields)}")

                    st.markdown("#### Current Extracted Data")
                    st.json(extracted_data)

                    st.markdown("---")
                    st.markdown("#### Provide Missing Information")

                    # Create form for missing fields
                    with st.form(key=f"form_{thread_id}"):
                        updated_fields = {}

                        if "customer_name" in missing_fields:
                            updated_fields["customer_name"] = st.text_input(
                                "Customer Name *",
                                value=extracted_data.get("customer_name", "")
                            )

                        if "customer_email" in missing_fields:
                            updated_fields["customer_email"] = st.text_input(
                                "Customer Email *",
                                value=extracted_data.get("customer_email", "")
                            )

                        if "customer_address" in missing_fields:
                            updated_fields["customer_address"] = st.text_area(
                                "Customer Address *",
                                value=extracted_data.get("customer_address", ""),
                                height=80
                            )

                        if "product_name" in missing_fields:
                            from src.config import PRODUCT_CATALOG
                            updated_fields["product_name"] = st.selectbox(
                                "Product Name *",
                                options=[""] + list(PRODUCT_CATALOG.keys()),
                                index=0
                            )

                        if "serial_number" in missing_fields:
                            updated_fields["serial_number"] = st.text_input(
                                "Serial Number",
                                value=extracted_data.get("serial_number", "")
                            )

                        if "purchase_date" in missing_fields:
                            purchase_date_input = st.date_input(
                                "Purchase Date *",
                                value=None,
                                key=f"purchase_date_{thread_id}"
                            )
                            updated_fields["purchase_date"] = purchase_date_input.isoformat() if purchase_date_input else ""

                        if "issue_description" in missing_fields:
                            updated_fields["issue_description"] = st.text_area(
                                "Issue Description *",
                                value=extracted_data.get("issue_description", ""),
                                height=100
                            )

                        submit_btn = st.form_submit_button("💾 Save and Continue", type="primary")

                        if submit_btn:
                            # Filter out empty fields
                            filled_fields = {k: v for k, v in updated_fields.items() if v}

                            if filled_fields:
                                with st.spinner("Updating fields and resuming..."):
                                    result = update_missing_fields(thread_id, filled_fields)

                                    if result["success"]:
                                        # Check if still missing fields
                                        res_data = result["result"]
                                        if res_data.get("missing_fields"):
                                            st.warning(f"Still missing: {', '.join(res_data['missing_fields'])}")
                                        else:
                                            st.success("✅ All fields provided! Claim is now pending review.")
                                        st.cache_resource.clear()
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {result['error']}")
                            else:
                                st.warning("Please fill in at least one field")

                    st.markdown("---")

elif page == "Pending Claims":
    st.markdown("### Claims Awaiting Review")

    pending_claims = get_pending_claims()

    if not pending_claims:
        st.info("No pending claims to review. New claims will appear here automatically.")
        st.markdown("""
        **To test the system:**
        1. Generate sample claims: `python scripts/generate_claims.py`
        2. Process a claim: `python src/agent.py --file data/test_claims/valid/valid_001.json`
        3. The claim will appear here for review
        """)
    else:
        for claim in pending_claims:
            thread_id = claim["thread_id"]
            claim_id = claim["claim_id"]
            state = claim["state"]

            with st.expander(f"📧 Claim: {claim_id}", expanded=True):
                # Create columns for layout
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Original Email
                    st.markdown("#### Original Email")
                    st.markdown(f"**From:** {state.get('email_from', 'N/A')}")
                    st.markdown(f"**Subject:** {state.get('email_subject', 'N/A')}")
                    with st.container():
                        st.text_area(
                            "Email Body",
                            state.get('email_body', 'No content'),
                            height=150,
                            disabled=True,
                            key=f"email_{thread_id}"
                        )

                    # Extracted Data
                    st.markdown("#### Extracted Data")
                    extracted = state.get('extracted_data', {})
                    if extracted:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Customer:** {extracted.get('customer_name', 'N/A')}")
                            st.markdown(f"**Email:** {extracted.get('customer_email', 'N/A')}")
                            st.markdown(f"**Product:** {extracted.get('product_name', 'N/A')}")
                        with col_b:
                            st.markdown(f"**Serial:** {extracted.get('serial_number', 'N/A')}")
                            st.markdown(f"**Purchase Date:** {extracted.get('purchase_date', 'N/A')}")
                            st.markdown(f"**Proof of Purchase:** {'Yes' if extracted.get('has_proof_of_purchase') else 'No'}")

                        st.markdown(f"**Issue:** {extracted.get('issue_description', 'N/A')}")

                with col2:
                    # AI Recommendation
                    st.markdown("#### AI Analysis")
                    adj = state.get('adjudication_result', {})

                    if adj:
                        # Recommendation badge
                        rec = adj.get('recommendation', 'UNKNOWN')
                        conf = adj.get('confidence', 'UNKNOWN')

                        if rec == "APPROVE":
                            st.success(f"🟢 Recommend: **{rec}** ({conf})")
                        elif rec == "REJECT":
                            st.error(f"🔴 Recommend: **{rec}** ({conf})")
                        else:
                            st.warning(f"🟡 Recommend: **{rec}** ({conf})")

                        # Warranty status
                        if adj.get('warranty_valid') is not None:
                            if adj['warranty_valid']:
                                st.success("✓ Warranty Valid")
                            else:
                                st.error("✗ Warranty Expired")

                        if adj.get('days_since_purchase'):
                            st.info(f"📅 {adj['days_since_purchase']} days since purchase")

                        if adj.get('exclusion_triggered'):
                            st.warning(f"⚠️ Exclusion: {adj['exclusion_triggered']}")

                # Facts, Assumptions, Reasoning
                st.markdown("#### Analysis Details")

                if adj:
                    facts_tab, assumptions_tab, reasoning_tab, policy_tab = st.tabs(
                        ["📋 Facts", "💭 Assumptions", "🧠 Reasoning", "📄 Policy"]
                    )

                    with facts_tab:
                        facts = adj.get('facts', [])
                        if facts:
                            for fact in facts:
                                st.markdown(f'<div class="fact-box">✓ {fact}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No facts extracted")

                    with assumptions_tab:
                        assumptions = adj.get('assumptions', [])
                        if assumptions:
                            for assumption in assumptions:
                                st.markdown(f'<div class="assumption-box">? {assumption}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No assumptions made")

                    with reasoning_tab:
                        reasoning = adj.get('reasoning', 'No reasoning provided')
                        st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

                    with policy_tab:
                        refs = adj.get('policy_references', [])
                        if refs:
                            for ref in refs:
                                st.markdown(f"- {ref}")
                        else:
                            st.info("No policy references cited")

                        # Show retrieved policy
                        policy_text = state.get('policy_text', '')
                        if policy_text:
                            with st.expander("View Full Policy Text"):
                                st.text(policy_text[:2000] + "..." if len(policy_text) > 2000 else policy_text)

                # Decision buttons
                st.markdown("---")
                st.markdown("#### Make Decision")

                notes = st.text_input(
                    "Review Notes (optional)",
                    key=f"notes_{thread_id}",
                    placeholder="Add any notes about your decision..."
                )

                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("✅ Approve", key=f"approve_{thread_id}", type="primary"):
                        with st.spinner("Processing approval..."):
                            result = resume_claim(thread_id, "approve", notes)
                            if result["success"]:
                                st.success("Claim approved! Email sent to customer.")
                                label_path = result["result"].get("return_label_path")
                                if label_path:
                                    st.info(f"Return label generated: {label_path}")
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error(f"Error: {result['error']}")

                with col_btn2:
                    if st.button("❌ Reject", key=f"reject_{thread_id}", type="secondary"):
                        with st.spinner("Processing rejection..."):
                            result = resume_claim(thread_id, "reject", notes)
                            if result["success"]:
                                st.success("Claim rejected. Rejection email sent to customer.")
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error(f"Error: {result['error']}")

                with col_btn3:
                    if st.button("📝 Request Info", key=f"info_{thread_id}"):
                        with st.spinner("Sending info request..."):
                            result = resume_claim(thread_id, "need_info", notes)
                            if result["success"]:
                                st.success("Information request sent to customer.")
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error(f"Error: {result['error']}")

                st.markdown("---")


elif page == "Inquiries":
    st.markdown("### Product Inquiries")
    st.markdown("View all customer inquiries and AI-generated responses")

    inquiries = get_inquiries()

    if not inquiries:
        st.info("No inquiries have been processed yet.")
        st.markdown("""
        **How inquiries work:**
        1. Customer submits a product question (not a warranty claim)
        2. AI classifies it as "inquiry"
        3. AI searches policy database for relevant information
        4. AI generates a helpful response automatically
        5. Response is saved and can be viewed here

        **Try it:**
        - Go to "Manual Entry" and ask: "What features does the AirFlow Pro have?"
        - Or: "What is your warranty policy?"
        """)
    else:
        st.markdown(f"**Total Inquiries Processed:** {len(inquiries)}")
        st.markdown("---")

        for idx, inquiry in enumerate(inquiries):
            # Status indicator for inquiries
            response_email = inquiry.get('response_email', {})
            if response_email and response_email.get('body'):
                inquiry_status = "✅ ANSWERED"
            else:
                inquiry_status = "⏳ PROCESSING"

            with st.expander(f"{inquiry_status} | {inquiry['claim_id']} - {inquiry['email_subject'] or 'Product Inquiry'}", expanded=(idx == 0)):
                # Display inquiry details in columns
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("#### Customer Question")
                    st.markdown(f"**From:** {inquiry['email_from']}")
                    st.markdown(f"**Date:** {inquiry['timestamp'][:19] if inquiry['timestamp'] else 'N/A'}")
                    st.markdown(f"**Subject:** {inquiry['email_subject'] or 'N/A'}")

                    st.markdown("**Question:**")
                    st.text_area(
                        "Question",
                        value=inquiry['email_body'],
                        height=200,
                        key=f"inquiry_question_{inquiry['thread_id']}",
                        label_visibility="collapsed"
                    )

                with col2:
                    st.markdown("#### AI Response")
                    response_email = inquiry.get('response_email', {})

                    if response_email:
                        st.markdown(f"**To:** {response_email.get('to', 'N/A')}")
                        st.markdown(f"**Subject:** {response_email.get('subject', 'N/A')}")

                        st.markdown("**Response:**")
                        response_body = response_email.get('body', 'No response generated')
                        st.text_area(
                            "Response",
                            value=response_body,
                            height=200,
                            key=f"inquiry_response_{inquiry['thread_id']}",
                            label_visibility="collapsed"
                        )

                        # Show response quality indicator
                        response_length = len(response_body)
                        if response_length > 300:
                            st.success(f"✅ Detailed response ({response_length} chars)")
                        elif response_length > 100:
                            st.info(f"📝 Standard response ({response_length} chars)")
                        else:
                            st.warning(f"⚠️ Short response ({response_length} chars)")

                    else:
                        st.warning("⚠️ No response was generated for this inquiry")

                st.markdown("---")

                # Thread details
                with st.expander("🔍 View Technical Details"):
                    st.markdown(f"**Thread ID:** `{inquiry['thread_id']}`")
                    st.markdown(f"**Claim ID:** `{inquiry['claim_id']}`")
                    st.json(inquiry['state'])


elif page == "All Claims":
    st.markdown("### All Processed Claims")

    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT thread_id FROM checkpoints
            ORDER BY checkpoint_id DESC
            LIMIT 50
        """)
        all_threads = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not all_threads:
            st.info("No claims have been processed yet.")
        else:
            app = get_graph()
            for thread_id in all_threads:
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    state = app.get_state(config)

                    if state and state.values:
                        values = state.values
                        classification = values.get("classification", "Unknown")

                        # FILTER: Only show actual warranty claims
                        if classification != "claim":
                            continue  # Skip inquiries and spam

                        claim_id = values.get("claim_id", "Unknown")
                        decision = values.get("human_decision", "Pending")

                        # Status indicator for claims only
                        if decision:
                            status = f"✅ {decision.upper()}"
                        else:
                            status = "⏳ PENDING"

                        with st.expander(f"{status} | {claim_id}"):
                            st.json(values)

                except Exception as e:
                    st.warning(f"Error loading thread {thread_id}")

    except Exception as e:
        st.error(f"Database error: {e}")


elif page == "Settings":
    st.markdown("### System Settings")

    st.markdown("#### Database")
    st.code(str(DATABASE_PATH))

    st.markdown("#### Clear Database")
    st.warning("This will delete all claim history!")

    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("I understand this will delete all data"):
            try:
                if DATABASE_PATH.exists():
                    DATABASE_PATH.unlink()
                st.success("Database cleared!")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    st.markdown("#### Instructions")
    st.markdown("""
    **How to use this dashboard:**

    **Setup (First Time):**
    1. **Generate Test Data**:
       ```bash
       python scripts/generate_policies.py
       python scripts/generate_claims.py
       ```

    2. **Ingest Policies**:
       ```bash
       python src/ingest.py
       ```

    **Processing Claims:**

    **Option 1: Manual Entry (Recommended)**
    1. Go to "Manual Entry" page
    2. Type or paste claim text
    3. Submit claim
    4. If fields are missing, provide them in the "Provide Missing Fields" tab
    5. Review claim in "Pending Claims" and approve/reject

    **Option 2: Process JSON File**
       ```bash
       python src/agent.py --file data/test_claims/valid/valid_001.json
       ```

    **Option 3: Command Line Manual Entry**
       ```bash
       python src/agent.py --text "Customer text here..."
       ```

    **Check Output**:
       - Emails: `data/outbox/`
       - Return labels: `data/labels/`
       - Evaluation metrics: `data/test_claims/evaluation_results_*.json`
    """)

    st.markdown("---")

    st.markdown("#### About")
    st.markdown("""
    **Warranty Claims Agent v1.0**

    An Agentic AI system for processing warranty claims with human-in-the-loop approval.

    Built with:
    - LangGraph (Agent orchestration & workflow)
    - ChromaDB (Policy retrieval with semantic search)
    - OpenAI API (GPT-4o-mini for LLM)
    - Streamlit (This dashboard)

    **Features:**
    - **Automated Product Inquiry Handling** - AI answers customer questions using policy database
    - **Manual Claim Entry** - With automatic field validation
    - **Automated Triage** - Classifies as spam, inquiry, or warranty claim
    - **Semantic Policy Retrieval** - ChromaDB vector search
    - **AI Adjudication** - Recommends approve/reject with reasoning
    - **Human-in-the-Loop Review** - Two HITL stages:
      1. Missing field collection
      2. Final approval/rejection decision
    - **Email & Label Generation** - Automatic fulfillment after approval
    """)


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Warranty Claims Agent | Human-in-the-Loop Dashboard | "
    f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)
