"""
MCP (Model Context Protocol) Server for Warranty Claims Agent.

Exposes tools for:
- Validating serial numbers
- Sending emails (mocked - writes to outbox)
- Generating return labels

This demonstrates the MCP pattern for decoupling AI agents from infrastructure.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTBOX_DIR, LABELS_DIR, PRODUCT_CATALOG

# Try to import MCP
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP not available, using standalone mode")

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class WarrantyToolsServer:
    """
    MCP Server providing warranty-related tools.

    Tools:
    - validate_serial: Check if a serial number is valid
    - send_email: Send an email to a customer (mocked)
    - generate_return_label: Create a shipping label PDF
    """

    def __init__(self):
        """Initialize the tools server."""
        self.outbox_dir = OUTBOX_DIR
        self.labels_dir = LABELS_DIR
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Email log file
        self.email_log = self.outbox_dir / "email_log.jsonl"

        # Valid serial number prefixes (for mock validation)
        self.valid_prefixes = list(PRODUCT_CATALOG.values())

    def validate_serial(self, serial_number: str) -> dict:
        """
        Validate a product serial number.

        Mock implementation - checks format and prefix.

        Args:
            serial_number: The serial number to validate (e.g., "HD-002-583921")

        Returns:
            dict with 'valid' boolean and 'message'
        """
        if not serial_number:
            return {
                "valid": False,
                "message": "No serial number provided"
            }

        # Check format: XX-XXX-XXXXXX
        parts = serial_number.split('-')
        if len(parts) != 3:
            return {
                "valid": False,
                "message": f"Invalid format. Expected format: HD-XXX-XXXXXX, got: {serial_number}"
            }

        prefix = f"{parts[0]}-{parts[1]}"

        # Check if prefix matches a known product
        if prefix in self.valid_prefixes:
            # Check numeric suffix
            try:
                int(parts[2])
                return {
                    "valid": True,
                    "message": f"Serial number {serial_number} is valid",
                    "product_id": prefix,
                    "product_name": next(
                        (name for name, pid in PRODUCT_CATALOG.items() if pid == prefix),
                        "Unknown"
                    )
                }
            except ValueError:
                return {
                    "valid": False,
                    "message": f"Invalid serial suffix. Expected numeric, got: {parts[2]}"
                }
        else:
            return {
                "valid": False,
                "message": f"Unknown product prefix: {prefix}. Valid prefixes: {self.valid_prefixes}"
            }

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        attachments: Optional[list] = None
    ) -> dict:
        """
        Send an email to a customer (mocked - writes to log file).

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            attachments: Optional list of attachment paths

        Returns:
            dict with 'success' boolean and 'message_id'
        """
        message_id = f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(to) % 10000:04d}"

        email_record = {
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "to": to,
            "subject": subject,
            "body": body,
            "attachments": attachments or [],
            "status": "sent"
        }

        # Append to log file
        with open(self.email_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(email_record) + '\n')

        # Also save as individual file for easy viewing
        email_file = self.outbox_dir / f"{message_id}.json"
        with open(email_file, 'w', encoding='utf-8') as f:
            json.dump(email_record, f, indent=2)

        logger.info(f"Email sent: {message_id} to {to}")

        return {
            "success": True,
            "message_id": message_id,
            "message": f"Email sent successfully to {to}",
            "log_file": str(email_file)
        }

    def generate_return_label(
        self,
        claim_id: str,
        customer_name: str,
        customer_address: str,
        product_name: str,
        serial_number: str
    ) -> dict:
        """
        Generate a return shipping label PDF.

        Args:
            claim_id: The warranty claim ID
            customer_name: Customer's full name
            customer_address: Customer's mailing address
            product_name: Name of the product being returned
            serial_number: Product serial number

        Returns:
            dict with 'success' boolean and 'label_path'
        """
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"return_label_{claim_id}_{timestamp}.pdf"
        label_path = self.labels_dir / filename

        # Create PDF
        c = canvas.Canvas(str(label_path), pagesize=letter)
        width, height = letter

        # Company return address
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, height - 1 * inch, "RETURN SHIPPING LABEL")

        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, height - 1.5 * inch, "FROM:")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, height - 1.75 * inch, customer_name)

        # Handle multi-line address
        c.setFont("Helvetica", 10)
        address_lines = customer_address.split('\n') if customer_address else ["Address not provided"]
        y_pos = height - 2 * inch
        for line in address_lines[:4]:  # Max 4 lines
            c.drawString(1 * inch, y_pos, line.strip())
            y_pos -= 0.2 * inch

        # Ship to address
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, height - 3 * inch, "SHIP TO:")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, height - 3.25 * inch, "HairDryer Co. Warranty Department")
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, height - 3.5 * inch, "1234 Manufacturing Way")
        c.drawString(1 * inch, height - 3.7 * inch, "Warehouse District")
        c.drawString(1 * inch, height - 3.9 * inch, "Cincinnati, OH 45202")

        # Barcode placeholder
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1 * inch, height - 4.5 * inch, f"CLAIM ID: {claim_id}")

        # Draw barcode-like rectangle
        c.rect(1 * inch, height - 5.5 * inch, 4 * inch, 0.75 * inch)
        c.setFont("Courier-Bold", 24)
        c.drawString(1.2 * inch, height - 5.2 * inch, f"||| {claim_id} |||")

        # Product info
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, height - 6 * inch, f"Product: {product_name}")
        c.drawString(1 * inch, height - 6.25 * inch, f"Serial: {serial_number}")
        c.drawString(1 * inch, height - 6.5 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Instructions
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1 * inch, height - 7.25 * inch, "INSTRUCTIONS:")
        c.setFont("Helvetica", 9)
        instructions = [
            "1. Cut out this label along the dotted line",
            "2. Securely package your product with all accessories",
            "3. Tape this label to the outside of the package",
            "4. Drop off at any UPS location or schedule a pickup",
            "5. Keep a copy of this label for your records"
        ]
        y_pos = height - 7.5 * inch
        for instruction in instructions:
            c.drawString(1 * inch, y_pos, instruction)
            y_pos -= 0.2 * inch

        # Dotted border
        c.setDash(3, 3)
        c.rect(0.5 * inch, height - 8.5 * inch, 7.5 * inch, 7.5 * inch)

        c.save()
        logger.info(f"Return label generated: {label_path}")

        return {
            "success": True,
            "label_path": str(label_path),
            "message": f"Return label generated for claim {claim_id}"
        }


# Create server instance
tools_server = WarrantyToolsServer()


def create_mcp_server():
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP is not available. Install 'mcp' package to use this function.")
    server = Server("warranty-tools")

    @server.list_tools()
    async def list_tools():
        """List available tools."""
        return [
            Tool(
                name="validate_serial",
                description="Validate a product serial number",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "serial_number": {
                            "type": "string",
                            "description": "The serial number to validate (e.g., HD-002-583921)"
                        }
                    },
                    "required": ["serial_number"]
                }
            ),
            Tool(
                name="send_email",
                description="Send an email to a customer (writes to outbox log)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject line"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        },
                        "attachments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of attachment paths"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            ),
            Tool(
                name="generate_return_label",
                description="Generate a return shipping label PDF",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "description": "The warranty claim ID"
                        },
                        "customer_name": {
                            "type": "string",
                            "description": "Customer's full name"
                        },
                        "customer_address": {
                            "type": "string",
                            "description": "Customer's mailing address"
                        },
                        "product_name": {
                            "type": "string",
                            "description": "Name of the product being returned"
                        },
                        "serial_number": {
                            "type": "string",
                            "description": "Product serial number"
                        }
                    },
                    "required": ["claim_id", "customer_name", "customer_address", "product_name", "serial_number"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Handle tool calls."""
        if name == "validate_serial":
            result = tools_server.validate_serial(arguments.get("serial_number", ""))
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "send_email":
            result = tools_server.send_email(
                to=arguments.get("to", ""),
                subject=arguments.get("subject", ""),
                body=arguments.get("body", ""),
                attachments=arguments.get("attachments")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "generate_return_label":
            result = tools_server.generate_return_label(
                claim_id=arguments.get("claim_id", ""),
                customer_name=arguments.get("customer_name", ""),
                customer_address=arguments.get("customer_address", ""),
                product_name=arguments.get("product_name", ""),
                serial_number=arguments.get("serial_number", "")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def run_mcp_server():
    """Run the MCP server."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Warranty Tools MCP Server")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no MCP)")
    args = parser.parse_args()

    if args.test or not MCP_AVAILABLE:
        # Test mode - demonstrate tools without MCP
        print("=" * 50)
        print("WARRANTY TOOLS SERVER - TEST MODE")
        print("=" * 50)

        # Test validate_serial
        print("\n1. Testing validate_serial:")
        result = tools_server.validate_serial("HD-002-583921")
        print(f"   Valid serial: {json.dumps(result, indent=2)}")

        result = tools_server.validate_serial("INVALID-123")
        print(f"   Invalid serial: {json.dumps(result, indent=2)}")

        # Test send_email
        print("\n2. Testing send_email:")
        result = tools_server.send_email(
            to="customer@example.com",
            subject="Warranty Claim Approved",
            body="Your warranty claim has been approved. A return label is attached."
        )
        print(f"   Result: {json.dumps(result, indent=2)}")

        # Test generate_return_label
        print("\n3. Testing generate_return_label:")
        result = tools_server.generate_return_label(
            claim_id="CLM-2026-001",
            customer_name="John Smith",
            customer_address="123 Main Street\nBoston, MA 02101",
            product_name="AirFlow Pro",
            serial_number="HD-002-583921"
        )
        print(f"   Result: {json.dumps(result, indent=2)}")

        print("\n" + "=" * 50)
        print("Test complete! Check ./data/outbox and ./data/labels")
        print("=" * 50)

    else:
        # Run actual MCP server
        import asyncio
        logger.info("Starting MCP server...")
        asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
