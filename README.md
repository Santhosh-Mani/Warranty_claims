# Warranty Claims Agentic AI System

An end-to-end Agentic AI system for processing warranty claims and product inquiries with human-in-the-loop approval. Built with LangGraph, ChromaDB, OpenAI, and Streamlit.

## Features

✅ **Automated Triage** - Classifies emails as spam, inquiries, or warranty claims
✅ **Product Inquiry Auto-Response** - AI answers customer questions using policy database
✅ **Data Extraction** - Extracts customer info, product details, and issue descriptions
✅ **Policy Retrieval** - Semantic search against warranty policies using vector embeddings
✅ **AI Adjudication** - Chain-of-Thought reasoning to evaluate claim validity
✅ **Human-in-the-Loop** - Two review stages for quality control
✅ **Bulk Processing** - Watch mode for continuous email processing
✅ **Automated Response** - Generates approval/rejection emails and return shipping labels
✅ **Evaluation Metrics** - 95% accuracy with 0% false approvals

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INBOX (./data/inbox/)                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph Workflow                                                 │
│  ┌─────────┐                                                        │
│  │ Triage  │───────┬─────────────────────────────────────┐         │
│  └─────────┘       │                                     │         │
│                    │                                     │         │
│         ┌──────────▼───────────┐           ┌────────────▼──────┐  │
│         │  INQUIRY              │           │  CLAIM            │  │
│         │  ┌────────────┐       │           │  ┌──────────┐    │  │
│         │  │  Retrieve  │       │           │  │ Extract  │    │  │
│         │  │  Policies  │       │           │  └────┬─────┘    │  │
│         │  └─────┬──────┘       │           │       │          │  │
│         │        ▼               │           │       ▼          │  │
│         │  ┌────────────┐       │           │  ┌──────────┐    │  │
│         │  │  Generate  │       │           │  │ Retrieve │    │  │
│         │  │  Response  │       │           │  │ Policies │    │  │
│         │  └────────────┘       │           │  └────┬─────┘    │  │
│         │  (Auto-complete)      │           │       ▼          │  │
│         └───────────────────────┘           │  ┌──────────┐    │  │
│                                             │  │Adjudicate│    │  │
│                                             │  └────┬─────┘    │  │
│                                             │       ▼          │  │
│                                             │  ┌──────────┐    │  │
│                                             │  │  Human   │    │  │
│                                             │  │  Review  │    │  │
│                                             │  │ (PAUSE)  │    │  │
│                                             │  └────┬─────┘    │  │
│                                             │       ▼          │  │
│                                             │  ┌──────────┐    │  │
│                                             │  │Fulfillment│   │  │
│                                             │  └──────────┘    │  │
│                                             └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | LangGraph + SQLite | Stateful workflows with checkpoints |
| LLM | OpenAI GPT-4o-mini | LLM for triage, extraction, adjudication |
| Document Parsing | Docling / PDFPlumber | PDF to Markdown conversion |
| Vector Store | ChromaDB | Policy embeddings and retrieval |
| Data Validation | Pydantic | Structured output enforcement |
| UI | Streamlit | Human review dashboard |
| MCP Tools | FastMCP | Email and label generation |

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- 4GB+ RAM

### Installation

```bash
# Clone the repository
cd warranty-claims-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Copy .env.example to .env and add your OpenAI API key
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac

# Edit .env and add:
# OPENAI_API_KEY=your_api_key_here
```

### Generate Sample Data

```bash
# Generate 10 warranty policy PDFs
python scripts/generate_policies.py

# Generate 30 test claim emails (valid, invalid, spam, inquiries)
python scripts/generate_claims.py
```

### Ingest Policies into Vector Store

```bash
python src/ingest.py
```

This will create a ChromaDB vector store with embeddings of all warranty policies for semantic search.

### Run the System

**Option 1: Process a single claim**
```bash
python src/agent.py --file data/test_claims/valid/valid_001.json
```

**Option 2: Process an inquiry**
```bash
python src/agent.py --file data/test_claims/inquiry/inquiry_001.json --no-interrupt
```

**Option 3: Watch inbox for new emails (bulk processing)**
```bash
python src/agent.py --watch
```

**Option 4: Use the Streamlit Dashboard**
```bash
# In a separate terminal
streamlit run src/app.py
```
Then open http://localhost:8501 in your browser.

### Run Evaluation

```bash
python scripts/run_evaluation.py
```

This will process all 30 test cases and calculate accuracy metrics.

## Bulk Processing Demo

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for comprehensive demonstrations of:
- Watch mode (continuous processing)
- Batch evaluation with metrics
- Manual bulk processing loops
- Mixed batch processing (claims + inquiries)
- Dashboard walkthrough

## Project Structure

```
warranty-claims-agent/
├── src/
│   ├── config.py          # Configuration settings
│   ├── models.py          # Pydantic models
│   ├── ingest.py          # Policy ingestion (Docling → ChromaDB)
│   ├── server.py          # MCP server (email, labels)
│   ├── agent.py           # LangGraph workflow
│   └── app.py             # Streamlit dashboard
│
├── scripts/
│   ├── generate_policies.py  # Create sample policies
│   ├── generate_claims.py    # Create test emails
│   ├── run_evaluation.py     # Batch evaluation
│   ├── reset_data.py          # Reset data for demos
│   ├── view_claims_db.py      # View claims database
│   └── view_chroma_db.py      # View vector store
│
├── data/
│   ├── policies/          # Warranty policy PDFs
│   ├── inbox/             # Incoming emails (claims & inquiries)
│   ├── outbox/            # Sent emails & inquiry responses
│   ├── labels/            # Return shipping labels
│   ├── chroma_db/         # Vector store
│   ├── claims.db          # LangGraph checkpoints
│   └── test_claims/       # Evaluation dataset
│       ├── valid/         # Valid warranty claims
│       ├── invalid/       # Invalid claims (expired, exclusions)
│       ├── inquiry/       # Product inquiries
│       └── spam/          # Spam/irrelevant emails
│
├── .gitignore
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── README.md
├── DEMO_GUIDE.md          # Bulk processing demonstrations
├── WORKFLOW.md            # Detailed workflow documentation
├── DATABASE_GUIDE.md      # Database schema and queries
├── INQUIRY_GUIDE.md       # Inquiry handling documentation
└── REPORT.md              # Project report
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# OpenAI Settings
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Optional: Adjust other settings
```

## Workflow Overview

### Warranty Claims Flow

1. **Claim arrives** in `data/inbox/` (or via dashboard)
2. **Triage**: Classifies as spam, inquiry, or claim
3. **Extract**: Extracts customer info, product, serial number, purchase date, issue
4. **Retrieve**: Semantic search for relevant warranty policies
5. **Adjudicate**: AI analyzes claim and recommends APPROVE/REJECT/NEED_INFO with reasoning
6. **Human Review**: Workflow pauses at Streamlit dashboard
7. **Fulfillment**: Sends approval/rejection email and generates return label if approved

### Product Inquiry Flow

1. **Inquiry arrives** in `data/inbox/` (or via dashboard)
2. **Triage**: Classifies as inquiry
3. **Retrieve**: Semantic search for relevant product information
4. **Generate Response**: AI generates helpful response using policy context
5. **Auto-complete**: No human review needed, response sent immediately

## Human-in-the-Loop Dashboard

The Streamlit dashboard provides:

- **Manual Entry**: Submit emails directly via web form
- **Pending Claims**: Review claims awaiting human decision
- **Inquiries**: View all product inquiries and AI responses
- **All Claims**: Status overview of all processed warranty claims
- **Settings**: Database management and system configuration

## Data Reset (for Demos)

Reset the system between demonstrations:

```bash
# Quick reset (keep policies, clear processed data)
python scripts/reset_data.py --partial

# Full reset (clean slate, requires re-ingestion)
python scripts/reset_data.py --full
python src/ingest.py

# Preview what would be deleted
python scripts/reset_data.py --dry-run
```

## Evaluation Metrics

The evaluation script calculates:

- **Triage Accuracy**: Correctly classifying claims vs inquiries vs spam
- **Recommendation Accuracy**: APPROVE/REJECT/NEED_INFO matching ground truth
- **False Approval Rate**: Invalid claims wrongly approved (target: 0%)
- **False Rejection Rate**: Valid claims wrongly rejected
- **Product Match Rate**: Correctly identifying the product
- **Date Extraction Rate**: Correctly parsing purchase dates

**Typical Results**:
- Overall Accuracy: ~95%
- Triage Accuracy: ~100%
- False Approval Rate: 0%
- False Rejection Rate: ~4%

## Test Data Distribution

- **6 spam/irrelevant emails** (20%)
- **9 valid claims** (30%)
- **15 invalid claims** (50%)
  - 5 expired warranty
  - 7 exclusion triggered (water damage, commercial use, etc.)
  - 3 missing information
- **3 product inquiries** (10%)

## API Reference

### Agent CLI

```bash
# Process single file
python src/agent.py --file <path>

# Watch directory for bulk processing
python src/agent.py --watch

# Auto-approve for testing (skip human review)
python src/agent.py --file <path> --decision approve

# Process without interruption (for inquiries)
python src/agent.py --file <path> --no-interrupt
```

### MCP Tools (server.py)

```python
# Validate serial number
tools_server.validate_serial("HD-002-583921")

# Send email (mocked)
tools_server.send_email(
    to="customer@example.com",
    subject="Warranty Approved",
    body="Your claim has been approved..."
)

# Generate return label
tools_server.generate_return_label(
    claim_id="CLM-001",
    customer_name="John Smith",
    customer_address="123 Main St",
    product_name="AirFlow Pro",
    serial_number="HD-002-583921"
)
```

## Troubleshooting

### OpenAI API errors
```bash
# Check your API key is set
echo %OPENAI_API_KEY%  # Windows
echo $OPENAI_API_KEY   # Linux/Mac

# Verify in .env file
type .env  # Windows
cat .env   # Linux/Mac
```

### ChromaDB errors
```bash
# Clear and reinitialize
python scripts/reset_data.py --full
python src/ingest.py
```

### Claims not appearing in dashboard
```bash
# Click "🔄 Refresh" button in sidebar
# Or clear database and reprocess
python scripts/reset_data.py --partial
python src/agent.py --file data/test_claims/valid/valid_001.json
```

### Watch mode not detecting files
- Ensure files are `.json` format
- Check correct inbox path: `data/inbox/`
- Verify watchdog is installed: `pip install watchdog`

## Performance Benchmarks

- **Single Claim**: ~10 seconds (excluding human review pause)
- **Single Inquiry**: ~8 seconds (auto-completed)
- **Batch Processing**: 30 claims in ~5 minutes (auto-mode)
- **Watch Mode**: Real-time as files arrive

## Documentation

- [WORKFLOW.md](WORKFLOW.md) - Detailed workflow and state machine
- [DATABASE_GUIDE.md](DATABASE_GUIDE.md) - Database schema and SQL queries
- [INQUIRY_GUIDE.md](INQUIRY_GUIDE.md) - Product inquiry handling
- [DEMO_GUIDE.md](DEMO_GUIDE.md) - Bulk processing demonstrations
- [REPORT.md](REPORT.md) - Project report and design decisions

## Future Improvements

- [ ] OCR for receipt attachments
- [ ] Multi-language support
- [ ] Email integration (IMAP/SMTP)
- [ ] Auto-approval for high-confidence claims
- [ ] Feedback loop for model fine-tuning
- [ ] Docker containerization
- [ ] Multi-product support
- [ ] Advanced analytics dashboard

## License

MIT License

## Contact

For questions about the assessment, please contact the interviewer.
