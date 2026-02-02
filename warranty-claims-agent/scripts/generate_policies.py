"""
Generate synthetic warranty policy PDF documents.

Creates 10 warranty policy PDFs for different hair dryer products,
each with unique terms, exclusions, and coverage rules.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from src.config import POLICIES_DIR, ensure_directories


# Product definitions with their specific warranty terms
PRODUCTS = [
    {
        "id": "HD-001",
        "name": "AirFlow Basic",
        "warranty_days": 90,
        "price": "$29.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Commercial or salon use",
            "Physical damage from dropping or impact",
            "Damage from use with non-standard voltage",
        ],
        "features": ["1800W motor", "2 heat settings", "Cool shot button"],
    },
    {
        "id": "HD-002",
        "name": "AirFlow Pro",
        "warranty_days": 90,
        "price": "$49.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Unauthorized repairs or modifications",
            "Damage from power surges",
            "Use with incompatible attachments",
        ],
        "features": ["2000W motor", "3 heat settings", "Ionic technology", "Concentrator nozzle"],
    },
    {
        "id": "HD-003",
        "name": "TravelDry Mini",
        "warranty_days": 90,
        "price": "$24.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Continuous use exceeding 20 minutes",
            "Use with voltage converters",
            "Damage during travel or shipping",
        ],
        "features": ["1200W motor", "Compact design", "Dual voltage (110/220V)", "Foldable handle"],
    },
    {
        "id": "HD-004",
        "name": "TravelDry Plus",
        "warranty_days": 90,
        "price": "$34.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Physical damage from dropping",
            "Damage to retractable cord from misuse",
            "Use in high-humidity environments (steam rooms)",
        ],
        "features": ["1500W motor", "Retractable cord", "Dual voltage", "Diffuser included"],
    },
    {
        "id": "HD-005",
        "name": "SalonMaster 3000",
        "warranty_days": 180,
        "price": "$149.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Damage from modified voltage or wiring",
            "Normal wear of filters and attachments",
            "Cosmetic damage not affecting function",
        ],
        "features": [
            "2400W AC motor",
            "Professional-grade",
            "Commercial use approved",
            "6 heat/speed settings",
        ],
        "commercial_allowed": True,
    },
    {
        "id": "HD-006",
        "name": "SalonMaster Elite",
        "warranty_days": 180,
        "price": "$199.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Use with modified voltage or non-standard outlets",
            "Damage from chemical exposure (hair products inside motor)",
            "Normal wear on removable filters",
        ],
        "features": [
            "2600W brushless motor",
            "Professional-grade",
            "Commercial use approved",
            "Digital temperature control",
        ],
        "commercial_allowed": True,
    },
    {
        "id": "HD-007",
        "name": "QuietBlow Compact",
        "warranty_days": 90,
        "price": "$39.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Failure to clean lint filter (causing overheating)",
            "Commercial or salon use",
            "Damage from blocked air vents",
        ],
        "features": ["1600W quiet motor", "Noise reduction technology", "Removable lint filter"],
    },
    {
        "id": "HD-008",
        "name": "QuietBlow Deluxe",
        "warranty_days": 90,
        "price": "$59.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Overheating due to blocked vents or dirty filter",
            "Damage from use with wet hair near motor intake",
            "Commercial use",
        ],
        "features": [
            "1800W quiet motor",
            "Advanced noise reduction",
            "Removable lint filter",
            "Multiple attachments",
        ],
    },
    {
        "id": "HD-009",
        "name": "KidSafe Dryer",
        "warranty_days": 90,
        "price": "$34.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Misuse or rough handling",
            "Commercial or professional use",
            "Damage from dropping or throwing",
        ],
        "features": [
            "1400W motor",
            "Auto shut-off safety",
            "Cool exterior",
            "Low noise operation",
        ],
    },
    {
        "id": "HD-010",
        "name": "ProStyle Ionic",
        "warranty_days": 90,
        "price": "$79.99",
        "exclusions": [
            "Water damage or exposure to moisture",
            "Use with unauthorized accessories or attachments",
            "Damage to ionic generator from product buildup",
            "Commercial use without extended warranty",
        ],
        "features": [
            "2200W motor",
            "Advanced ionic technology",
            "Ceramic heating element",
            "Multiple styling attachments",
        ],
    },
]


def create_policy_pdf(product: dict, output_path: Path):
    """Generate a warranty policy PDF for a product."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=20,
        alignment=1,  # Center
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.darkblue,
    )
    body_style = styles["Normal"]

    content = []

    # Title
    content.append(Paragraph(f"{product['name']} Warranty Policy", title_style))
    content.append(Spacer(1, 0.2 * inch))

    # Product Information
    content.append(Paragraph("Product Information", heading_style))
    product_info = [
        ["Model Number:", product["id"]],
        ["Product Name:", product["name"]],
        ["Retail Price:", product["price"]],
        ["Serial Number Format:", f"{product['id'][:2]}X-XXX-XXXXXX"],
    ]
    table = Table(product_info, colWidths=[2 * inch, 4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    content.append(table)
    content.append(Spacer(1, 0.2 * inch))

    # Product Features
    content.append(Paragraph("Product Features", heading_style))
    for feature in product["features"]:
        content.append(Paragraph(f"• {feature}", body_style))
    content.append(Spacer(1, 0.2 * inch))

    # Warranty Coverage
    content.append(Paragraph("Warranty Coverage", heading_style))
    warranty_text = f"""
    This product is covered by a limited warranty for a period of <b>{product['warranty_days']} days</b>
    from the original date of purchase. The warranty covers defects in materials and workmanship
    under normal use conditions.
    """
    content.append(Paragraph(warranty_text, body_style))
    content.append(Spacer(1, 0.1 * inch))

    coverage_info = [
        ["Warranty Duration:", f"{product['warranty_days']} days from purchase date"],
        ["Proof Required:", "Original receipt or order confirmation with purchase date"],
        [
            "Coverage Type:",
            "Repair or replacement at manufacturer's discretion",
        ],
    ]
    if product.get("commercial_allowed"):
        coverage_info.append(["Commercial Use:", "Approved for professional/salon use"])
    else:
        coverage_info.append(["Commercial Use:", "NOT approved - residential use only"])

    table = Table(coverage_info, colWidths=[2 * inch, 4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    content.append(table)
    content.append(Spacer(1, 0.2 * inch))

    # What Is Covered
    content.append(Paragraph("What Is Covered", heading_style))
    covered_items = [
        "Motor malfunction under normal use",
        "Heating element failure during warranty period",
        "Switch or button defects",
        "Power cord defects (excluding physical damage)",
        "Manufacturing defects affecting functionality",
    ]
    for item in covered_items:
        content.append(Paragraph(f"✓ {item}", body_style))
    content.append(Spacer(1, 0.2 * inch))

    # Exclusions
    content.append(Paragraph("Exclusions (What Is NOT Covered)", heading_style))
    content.append(
        Paragraph(
            "<i>The following conditions void the warranty:</i>",
            body_style,
        )
    )
    content.append(Spacer(1, 0.1 * inch))
    for exclusion in product["exclusions"]:
        content.append(Paragraph(f"✗ {exclusion}", body_style))
    content.append(Spacer(1, 0.1 * inch))

    # Standard exclusions
    standard_exclusions = [
        "Normal wear and tear",
        "Cosmetic damage that does not affect functionality",
        "Lost or stolen units",
        "Damage caused by accidents or misuse",
        "Use contrary to product instructions",
    ]
    for exclusion in standard_exclusions:
        content.append(Paragraph(f"✗ {exclusion}", body_style))
    content.append(Spacer(1, 0.2 * inch))

    # Claim Process
    content.append(Paragraph("Warranty Claim Process", heading_style))
    claim_steps = [
        "Contact our customer service team at warranty@hairdryer-co.com with your claim details.",
        "Provide proof of purchase (receipt, order confirmation, or invoice showing purchase date).",
        "Describe the issue or defect in detail.",
        "Include the product serial number (found on the product label).",
        "If approved, you will receive a prepaid shipping label to return the product.",
        "Upon receipt and verification, a replacement or repair will be processed within 7-10 business days.",
    ]
    for i, step in enumerate(claim_steps, 1):
        content.append(Paragraph(f"{i}. {step}", body_style))
    content.append(Spacer(1, 0.2 * inch))

    # Contact Information
    content.append(Paragraph("Contact Information", heading_style))
    contact_info = [
        ["Email:", "warranty@hairdryer-co.com"],
        ["Website:", "www.hairdryer-co.com/warranty"],
        ["Hours:", "Monday - Friday, 9 AM - 5 PM EST"],
    ]
    table = Table(contact_info, colWidths=[1.5 * inch, 4.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    content.append(table)
    content.append(Spacer(1, 0.3 * inch))

    # Legal Notice
    content.append(Paragraph("Legal Notice", heading_style))
    legal_text = """
    This warranty gives you specific legal rights, and you may also have other rights which vary
    from state to state. Some states do not allow limitations on implied warranties or exclusion
    of incidental or consequential damages, so the above limitations may not apply to you.
    The warranty is non-transferable and applies only to the original purchaser.
    """
    content.append(Paragraph(legal_text, body_style))

    # Build PDF
    doc.build(content)
    print(f"Created: {output_path}")


def main():
    """Generate all warranty policy PDFs."""
    ensure_directories()

    print(f"Generating {len(PRODUCTS)} warranty policy PDFs...")
    print(f"Output directory: {POLICIES_DIR}")
    print("-" * 50)

    for product in PRODUCTS:
        filename = f"{product['id']}_{product['name'].replace(' ', '_')}.pdf"
        output_path = POLICIES_DIR / filename
        create_policy_pdf(product, output_path)

    print("-" * 50)
    print(f"Successfully generated {len(PRODUCTS)} policy PDFs!")


if __name__ == "__main__":
    main()
