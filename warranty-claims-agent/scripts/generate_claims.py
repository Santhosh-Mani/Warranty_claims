"""
Generate synthetic warranty claim emails for testing.

Creates 30 test claim emails:
- 6 spam/irrelevant (20%)
- 9 valid claims (30%)
- 15 invalid claims (50%)
  - 5 expired warranty
  - 7 exclusion triggered
  - 3 missing information
"""

import sys
import json
from pathlib import Path
from datetime import date, timedelta
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from faker import Faker
from src.config import TEST_CLAIMS_DIR, ensure_directories

fake = Faker()
Faker.seed(42)  # For reproducibility
random.seed(42)

# Product info for generating claims
PRODUCTS = [
    {"id": "HD-001", "name": "AirFlow Basic", "warranty_days": 90},
    {"id": "HD-002", "name": "AirFlow Pro", "warranty_days": 90},
    {"id": "HD-003", "name": "TravelDry Mini", "warranty_days": 90},
    {"id": "HD-004", "name": "TravelDry Plus", "warranty_days": 90},
    {"id": "HD-005", "name": "SalonMaster 3000", "warranty_days": 180},
    {"id": "HD-006", "name": "SalonMaster Elite", "warranty_days": 180},
    {"id": "HD-007", "name": "QuietBlow Compact", "warranty_days": 90},
    {"id": "HD-008", "name": "QuietBlow Deluxe", "warranty_days": 90},
    {"id": "HD-009", "name": "KidSafe Dryer", "warranty_days": 90},
    {"id": "HD-010", "name": "ProStyle Ionic", "warranty_days": 90},
]

# Issues that are covered under warranty
COVERED_ISSUES = [
    "motor stopped working",
    "motor makes clicking sound",
    "heating element not working",
    "won't turn on",
    "switches don't work",
    "power button is stuck",
    "cord is frayed at the base",
    "makes a burning smell when used",
]

# Today's date for calculating warranty status
TODAY = date.today()


def generate_serial(product_id: str) -> str:
    """Generate a realistic serial number."""
    return f"{product_id}-{random.randint(100000, 999999)}"


def generate_spam_claims():
    """Generate 6 spam/irrelevant emails."""
    claims = []

    # Marketing spam
    claims.append({
        "id": "spam_001",
        "type": "spam",
        "expected_classification": "spam",
        "email": {
            "from": "marketing@amazingdeals.com",
            "subject": "AMAZING HAIR CARE DEALS - 70% OFF TODAY ONLY!",
            "body": """
Dear Valued Customer,

Don't miss our INCREDIBLE sale on professional hair care products!

🔥 70% OFF all styling tools
🔥 FREE shipping on orders over $50
🔥 Buy 2 Get 1 FREE on all accessories

Visit www.amazingdeals.com/haircare NOW!

This offer expires at midnight!

Unsubscribe: click here
            """.strip()
        }
    })

    claims.append({
        "id": "spam_002",
        "type": "spam",
        "expected_classification": "spam",
        "email": {
            "from": "prince.nigeria@email.ng",
            "subject": "URGENT: Business Proposal - $5,000,000 Transfer",
            "body": """
Dear Friend,

I am Prince Adebayo from Nigeria. I have $5,000,000 USD to transfer to your account.
Please send your bank details for this legitimate business opportunity.

God Bless,
Prince Adebayo
            """.strip()
        }
    })

    # General inquiry (not a claim)
    claims.append({
        "id": "inquiry_001",
        "type": "inquiry",
        "expected_classification": "inquiry",
        "email": {
            "from": "jane.doe@gmail.com",
            "subject": "Question about AirFlow Pro features",
            "body": """
Hi there,

I'm considering buying the AirFlow Pro hair dryer and had a few questions:

1. Does it have a cool shot button?
2. What attachments are included?
3. Is it dual voltage for travel?

I haven't purchased yet but want to make sure it's right for me.

Thanks!
Jane
            """.strip()
        }
    })

    claims.append({
        "id": "inquiry_002",
        "type": "inquiry",
        "expected_classification": "inquiry",
        "email": {
            "from": "bob.smith@yahoo.com",
            "subject": "Where can I buy replacement parts?",
            "body": """
Hello,

I've had my QuietBlow Deluxe for about 2 years now and I love it! The concentrator
nozzle broke recently (my fault, I dropped it). Where can I purchase a replacement
nozzle? I don't need a warranty claim, just want to buy the part.

Thanks,
Bob
            """.strip()
        }
    })

    # Job application
    claims.append({
        "id": "spam_003",
        "type": "spam",
        "expected_classification": "spam",
        "email": {
            "from": "recruiter@staffing.com",
            "subject": "Exciting Career Opportunity - Customer Service Manager",
            "body": """
Hello,

We came across your profile and think you'd be a great fit for our Customer Service
Manager position. The salary is $75,000-$95,000 with full benefits.

Please reply with your resume if interested.

Best regards,
HR Team
            """.strip()
        }
    })

    # Newsletter
    claims.append({
        "id": "spam_004",
        "type": "spam",
        "expected_classification": "spam",
        "email": {
            "from": "newsletter@beautymagazine.com",
            "subject": "This Week's Top 10 Hair Styling Tips",
            "body": """
BEAUTY WEEKLY NEWSLETTER

This week's top stories:
1. 10 Tips for Perfect Blowouts
2. Celebrity Hair Trends 2026
3. Best Products for Frizzy Hair
4. Interview with Top Stylist Maria Chen

Read more at beautymagazine.com

To unsubscribe, click here.
            """.strip()
        }
    })

    return claims


def generate_valid_claims():
    """Generate 9 valid warranty claims that should be approved."""
    claims = []

    # Valid claim 1: Recent purchase, motor failure
    purchase_date = TODAY - timedelta(days=45)
    claims.append({
        "id": "valid_001",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-002",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 45,
            "issue": "motor_failure",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "john.smith@gmail.com",
            "subject": "Warranty Claim - AirFlow Pro stopped working",
            "body": f"""
Hi,

My AirFlow Pro hair dryer suddenly stopped working yesterday. The motor makes a
clicking sound and won't spin at all. I've only had it for about 6 weeks.

Product: AirFlow Pro
Serial Number: HD-002-583921
Purchase Date: {purchase_date.strftime('%B %d, %Y')}
Purchased from: Amazon

I have the receipt and can provide it if needed. This is clearly a defect as I've
taken good care of it and only use it normally.

My address for shipping:
John Smith
123 Main Street
Boston, MA 02101

Please let me know how to proceed with the warranty claim.

Thank you,
John Smith
john.smith@gmail.com
            """.strip()
        }
    })

    # Valid claim 2: Heating element failure
    purchase_date = TODAY - timedelta(days=30)
    claims.append({
        "id": "valid_002",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-007",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 30,
            "issue": "heating_element_failure",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "emily.wilson@outlook.com",
            "subject": "QuietBlow Compact - No Heat",
            "body": f"""
Hello,

I purchased a QuietBlow Compact hair dryer about a month ago and it has stopped
producing heat. The motor runs fine and air comes out, but it's only cold air
regardless of the heat setting.

Details:
- Model: QuietBlow Compact (HD-007)
- Serial: HD-007-294817
- Bought on: {purchase_date.strftime('%m/%d/%Y')}
- Store: Target

I've attached a photo of my receipt. Please advise on the warranty process.

Emily Wilson
456 Oak Avenue
Chicago, IL 60601
emily.wilson@outlook.com
            """.strip()
        }
    })

    # Valid claim 3: Switch defect
    purchase_date = TODAY - timedelta(days=60)
    claims.append({
        "id": "valid_003",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-001",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 60,
            "issue": "switch_defect",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "maria.garcia@yahoo.com",
            "subject": "AirFlow Basic power button problem",
            "body": f"""
Hi there,

The power button on my AirFlow Basic is defective. It gets stuck in the ON position
and I have to unplug it to turn it off. This started happening last week.

I bought this on {purchase_date.strftime('%B %d')} from Walmart.
Serial number is HD-001-847293.

This seems like a manufacturing defect. I have the receipt.

Maria Garcia
789 Pine Road
Miami, FL 33101
305-555-1234
            """.strip()
        }
    })

    # Valid claim 4: SalonMaster with extended warranty
    purchase_date = TODAY - timedelta(days=120)  # Within 180 day warranty
    claims.append({
        "id": "valid_004",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-005",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 120,
            "issue": "motor_failure",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "lisa.stylist@salonbeauty.com",
            "subject": "SalonMaster 3000 Warranty Claim - Motor Issue",
            "body": f"""
Hello,

I own a small salon and purchased the SalonMaster 3000 for professional use about
4 months ago. The motor has started making a grinding noise and sometimes cuts out.

I know this is a professional-grade dryer approved for salon use, and I believe
it should still be under the 180-day warranty.

Product: SalonMaster 3000
Serial: HD-005-192847
Purchase Date: {purchase_date.strftime('%Y-%m-%d')}
Order #: AMZ-9876543

Please process this warranty claim. I rely on this dryer for my business.

Lisa Chen
Beauty First Salon
100 Commerce St
San Francisco, CA 94102
lisa.stylist@salonbeauty.com
            """.strip()
        }
    })

    # Valid claim 5: ProStyle Ionic
    purchase_date = TODAY - timedelta(days=15)
    claims.append({
        "id": "valid_005",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-010",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 15,
            "issue": "wont_turn_on",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "david.lee@email.com",
            "subject": "Brand new ProStyle Ionic won't turn on",
            "body": f"""
Hi,

I just bought a ProStyle Ionic hair dryer 2 weeks ago and it already won't turn on!
I've tried different outlets and nothing works. This is very frustrating for a
$80 product.

Serial: HD-010-394857
Purchased: {purchase_date.strftime('%m/%d/%Y')} from Best Buy
I have the receipt.

Please replace this defective unit ASAP.

David Lee
222 Elm Street
Seattle, WA 98101
            """.strip()
        }
    })

    # Valid claim 6: Cord defect
    purchase_date = TODAY - timedelta(days=75)
    claims.append({
        "id": "valid_006",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-008",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 75,
            "issue": "cord_defect",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "susan.taylor@icloud.com",
            "subject": "QuietBlow Deluxe - Cord Issue",
            "body": f"""
Hello,

The power cord on my QuietBlow Deluxe has started fraying at the base where it
connects to the dryer. I haven't done anything to damage it - it just started
happening. I'm worried it might be a fire hazard.

Model: QuietBlow Deluxe
Serial Number: HD-008-573920
Purchase Date: Around {purchase_date.strftime('%B %Y')}
I bought it at Bed Bath & Beyond and have my credit card statement as proof.

Please help!

Susan Taylor
888 Maple Lane
Denver, CO 80201
            """.strip()
        }
    })

    # Valid claim 7: TravelDry Mini
    purchase_date = TODAY - timedelta(days=50)
    claims.append({
        "id": "valid_007",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-003",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 50,
            "issue": "motor_failure",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "rachel.green@gmail.com",
            "subject": "TravelDry Mini defective",
            "body": f"""
Hi there,

My TravelDry Mini suddenly stopped working. The motor doesn't spin at all.
I've only used it a handful of times since I bought it about 7 weeks ago.

- Product: TravelDry Mini
- Serial #: HD-003-849201
- Purchased: {purchase_date.strftime('%B %d, %Y')}
- Where: Amazon.com

I have my Amazon order confirmation. This is definitely a defect.

Rachel Green
555 Central Park West, Apt 4B
New York, NY 10024
            """.strip()
        }
    })

    # Valid claim 8: KidSafe Dryer
    purchase_date = TODAY - timedelta(days=20)
    claims.append({
        "id": "valid_008",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-009",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 20,
            "issue": "auto_shutoff_malfunction",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "mom.jennifer@family.com",
            "subject": "KidSafe Dryer Warranty Issue",
            "body": f"""
Hello,

We bought the KidSafe Dryer for our daughter about 3 weeks ago. The auto shut-off
feature isn't working - it just keeps running without stopping. This defeats the
whole safety purpose of the product.

Details:
- KidSafe Dryer, model HD-009
- Serial: HD-009-102938
- Bought {purchase_date.strftime('%m/%d/%y')} from Target
- Have receipt

Please replace this as it's a safety concern.

Jennifer Adams
333 Family Drive
Austin, TX 78701
            """.strip()
        }
    })

    # Valid claim 9: SalonMaster Elite
    purchase_date = TODAY - timedelta(days=150)  # Within 180 day warranty
    claims.append({
        "id": "valid_009",
        "type": "valid_claim",
        "expected_classification": "claim",
        "expected_recommendation": "APPROVE",
        "ground_truth": {
            "product": "HD-006",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 150,
            "issue": "digital_control_failure",
            "within_warranty": True,
            "exclusion_triggered": None
        },
        "email": {
            "from": "mark.salon@hairpro.com",
            "subject": "SalonMaster Elite - Digital Display Not Working",
            "body": f"""
Hi,

The digital temperature display on my SalonMaster Elite has stopped working.
The dryer still functions but I can't see or adjust the temperature settings.

Product: SalonMaster Elite
Serial: HD-006-847362
Purchase Date: {purchase_date.strftime('%Y-%m-%d')}
Proof: Invoice attached

This is a professional unit used in my salon (which I know is allowed).
Should be within the 180-day warranty period.

Mark Johnson
Pro Hair Studio
444 Salon Row
Los Angeles, CA 90001
            """.strip()
        }
    })

    return claims


def generate_invalid_claims():
    """Generate 15 invalid claims (expired, exclusions, missing info)."""
    claims = []

    # ===== EXPIRED WARRANTY (5 claims) =====

    # Expired 1: Way past warranty
    purchase_date = TODAY - timedelta(days=200)
    claims.append({
        "id": "invalid_expired_001",
        "type": "invalid_expired",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-001",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 200,
            "issue": "motor_failure",
            "within_warranty": False,
            "rejection_reason": "warranty_expired"
        },
        "email": {
            "from": "sarah.jones@yahoo.com",
            "subject": "Warranty claim for AirFlow Basic",
            "body": f"""
Hi,

My AirFlow Basic hair dryer stopped heating. I bought it back in {purchase_date.strftime('%B %Y')}.
Serial number is HD-001-129384.

Can I get a replacement under warranty?

Sarah Jones
sarah.jones@yahoo.com
            """.strip()
        }
    })

    # Expired 2: Just past 90 days
    purchase_date = TODAY - timedelta(days=95)
    claims.append({
        "id": "invalid_expired_002",
        "type": "invalid_expired",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-002",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 95,
            "issue": "switch_defect",
            "within_warranty": False,
            "rejection_reason": "warranty_expired"
        },
        "email": {
            "from": "tom.baker@email.com",
            "subject": "AirFlow Pro switch broken",
            "body": f"""
Hello,

The heat switch on my AirFlow Pro is broken. I bought it on {purchase_date.strftime('%B %d, %Y')}.
Serial: HD-002-293847

Please process my warranty claim.

Tom Baker
777 Oak Street
Portland, OR 97201
            """.strip()
        }
    })

    # Expired 3: Over a year old
    purchase_date = TODAY - timedelta(days=400)
    claims.append({
        "id": "invalid_expired_003",
        "type": "invalid_expired",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-004",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 400,
            "issue": "motor_failure",
            "within_warranty": False,
            "rejection_reason": "warranty_expired"
        },
        "email": {
            "from": "old.customer@aol.com",
            "subject": "TravelDry Plus Warranty",
            "body": f"""
My TravelDry Plus from {purchase_date.strftime('%B %Y')} broke. The motor died.
Serial HD-004-384756. I need a replacement.

Thanks,
Old Customer
            """.strip()
        }
    })

    # Expired 4: 6 months for 90-day warranty product
    purchase_date = TODAY - timedelta(days=180)
    claims.append({
        "id": "invalid_expired_004",
        "type": "invalid_expired",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-007",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 180,
            "issue": "heating_element",
            "within_warranty": False,
            "rejection_reason": "warranty_expired"
        },
        "email": {
            "from": "kate.miller@gmail.com",
            "subject": "QuietBlow Compact not heating",
            "body": f"""
Hi there,

My QuietBlow Compact (serial HD-007-584930) stopped producing heat.
I purchased it approximately 6 months ago ({purchase_date.strftime('%B %Y')}).

Is this covered under warranty?

Kate Miller
            """.strip()
        }
    })

    # Expired 5: SalonMaster past 180 days
    purchase_date = TODAY - timedelta(days=200)
    claims.append({
        "id": "invalid_expired_005",
        "type": "invalid_expired",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-005",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 200,
            "issue": "motor_failure",
            "within_warranty": False,
            "rejection_reason": "warranty_expired"
        },
        "email": {
            "from": "salon.owner@beauty.com",
            "subject": "SalonMaster 3000 claim",
            "body": f"""
Our SalonMaster 3000 (HD-005-473829) has motor issues.
Purchased {purchase_date.strftime('%m/%d/%Y')}.

Please replace under warranty.

Salon Owner
            """.strip()
        }
    })

    # ===== EXCLUSION TRIGGERED (7 claims) =====

    # Exclusion 1: Water damage
    purchase_date = TODAY - timedelta(days=30)
    claims.append({
        "id": "invalid_water_001",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-003",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 30,
            "issue": "water_damage",
            "within_warranty": True,
            "exclusion_triggered": "water_damage",
            "rejection_reason": "exclusion_water_damage"
        },
        "email": {
            "from": "mike.brown@outlook.com",
            "subject": "TravelDry Mini not working after bathroom incident",
            "body": f"""
My TravelDry Mini fell into the sink while the water was running. Now it won't
turn on at all. I bought it on {purchase_date.strftime('%B %d')}, serial HD-003-847261.

Is this covered under warranty?

Mike Brown
mike.brown@outlook.com
            """.strip()
        }
    })

    # Exclusion 2: Commercial use for residential product
    purchase_date = TODAY - timedelta(days=45)
    claims.append({
        "id": "invalid_commercial_001",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-001",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 45,
            "issue": "motor_failure",
            "within_warranty": True,
            "exclusion_triggered": "commercial_use",
            "rejection_reason": "commercial_use_not_covered"
        },
        "email": {
            "from": "salon.beauty@business.com",
            "subject": "Bulk warranty claim - AirFlow Basic units failing",
            "body": f"""
We purchased 5 AirFlow Basic dryers for our salon on {purchase_date.strftime('%B %d')}.
After heavy daily use on clients, 3 of them have stopped working.

Serials: HD-001-483921, HD-001-483922, HD-001-483923

We need replacements ASAP for our business.

Beauty Salon Inc.
100 Salon Street
Miami, FL 33101
            """.strip()
        }
    })

    # Exclusion 3: Dropped/Physical damage
    purchase_date = TODAY - timedelta(days=20)
    claims.append({
        "id": "invalid_dropped_001",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-004",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 20,
            "issue": "physical_damage",
            "within_warranty": True,
            "exclusion_triggered": "physical_damage",
            "rejection_reason": "damage_from_dropping"
        },
        "email": {
            "from": "clumsy.user@gmail.com",
            "subject": "TravelDry Plus broke after falling",
            "body": f"""
Hi,

I accidentally dropped my TravelDry Plus from the bathroom counter onto the tile
floor. Now it makes a rattling noise and the heat doesn't work right.

Serial: HD-004-293847
Bought: {purchase_date.strftime('%m/%d/%Y')}

Can I get this fixed under warranty?

Thanks
            """.strip()
        }
    })

    # Exclusion 4: Continuous use exceeding limit
    purchase_date = TODAY - timedelta(days=40)
    claims.append({
        "id": "invalid_overuse_001",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-003",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 40,
            "issue": "overheating",
            "within_warranty": True,
            "exclusion_triggered": "continuous_use_exceeded",
            "rejection_reason": "exceeded_20_minute_limit"
        },
        "email": {
            "from": "long.hair@email.com",
            "subject": "TravelDry Mini overheated",
            "body": f"""
My TravelDry Mini overheated and stopped working. I was drying my very long hair
which takes about 45 minutes to an hour. After using it continuously for that time,
it just shut off and now won't turn back on.

Serial: HD-003-192837
Purchased {purchase_date.strftime('%B %d, %Y')}

Please help!
            """.strip()
        }
    })

    # Exclusion 5: Lint filter not cleaned
    purchase_date = TODAY - timedelta(days=60)
    claims.append({
        "id": "invalid_maintenance_001",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-007",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 60,
            "issue": "overheating",
            "within_warranty": True,
            "exclusion_triggered": "maintenance_neglect",
            "rejection_reason": "lint_filter_not_cleaned"
        },
        "email": {
            "from": "never.clean@email.com",
            "subject": "QuietBlow Compact keeps shutting off",
            "body": f"""
Hello,

My QuietBlow Compact keeps overheating and shutting off. I've had it for about
2 months (bought {purchase_date.strftime('%B %Y')}).

I'll be honest - I didn't know there was a lint filter and have never cleaned it.
When I looked, it was completely clogged with lint and dust.

Serial: HD-007-483920
Can I still get a replacement?
            """.strip()
        }
    })

    # Exclusion 6: Unauthorized repair
    purchase_date = TODAY - timedelta(days=50)
    claims.append({
        "id": "invalid_repair_001",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-002",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 50,
            "issue": "motor_failure",
            "within_warranty": True,
            "exclusion_triggered": "unauthorized_repair",
            "rejection_reason": "tampered_with"
        },
        "email": {
            "from": "diy.fixer@gmail.com",
            "subject": "AirFlow Pro still not working",
            "body": f"""
Hi,

My AirFlow Pro had a minor issue so I opened it up to try to fix it myself.
I watched some YouTube videos and tried to repair the motor. Unfortunately,
I made it worse and now it doesn't work at all.

Serial: HD-002-584930
Bought: {purchase_date.strftime('%m/%d/%Y')}

Can I get a replacement?
            """.strip()
        }
    })

    # Exclusion 7: Water damage (different product)
    purchase_date = TODAY - timedelta(days=25)
    claims.append({
        "id": "invalid_water_002",
        "type": "invalid_exclusion",
        "expected_classification": "claim",
        "expected_recommendation": "REJECT",
        "ground_truth": {
            "product": "HD-010",
            "purchase_date": purchase_date.isoformat(),
            "days_since_purchase": 25,
            "issue": "water_damage",
            "within_warranty": True,
            "exclusion_triggered": "water_damage",
            "rejection_reason": "exclusion_water_damage"
        },
        "email": {
            "from": "wet.bathroom@email.com",
            "subject": "ProStyle Ionic stopped working after getting wet",
            "body": f"""
My ProStyle Ionic got splashed with water while I was using it near the shower.
The bathroom was really steamy and some water dripped on it. Now it sparks
when I try to turn it on.

HD-010-394857
Bought {purchase_date.strftime('%B %d')}

Is this covered?
            """.strip()
        }
    })

    # ===== MISSING INFORMATION (3 claims) =====

    # Missing 1: No product info, no date, no serial
    claims.append({
        "id": "invalid_missing_001",
        "type": "invalid_missing_info",
        "expected_classification": "claim",
        "expected_recommendation": "NEED_INFO",
        "ground_truth": {
            "missing_fields": ["product_name", "purchase_date", "serial_number"],
            "action": "request_more_info"
        },
        "email": {
            "from": "vague.customer@email.com",
            "subject": "Hair dryer broken",
            "body": """
My hair dryer is broken. It just stopped working. Please send a new one.

Thanks
            """.strip()
        }
    })

    # Missing 2: Has product but no date or proof
    claims.append({
        "id": "invalid_missing_002",
        "type": "invalid_missing_info",
        "expected_classification": "claim",
        "expected_recommendation": "NEED_INFO",
        "ground_truth": {
            "missing_fields": ["purchase_date", "proof_of_purchase"],
            "product": "HD-002",
            "action": "request_more_info"
        },
        "email": {
            "from": "no.date@email.com",
            "subject": "AirFlow Pro warranty",
            "body": """
Hi,

I have an AirFlow Pro hair dryer (serial HD-002-938475) that stopped working.
The motor is dead.

I don't remember exactly when I bought it or where. I might have thrown away
the receipt. Can you still help?

Thanks
            """.strip()
        }
    })

    # Missing 3: No serial number, vague product
    claims.append({
        "id": "invalid_missing_003",
        "type": "invalid_missing_info",
        "expected_classification": "claim",
        "expected_recommendation": "NEED_INFO",
        "ground_truth": {
            "missing_fields": ["serial_number", "specific_product"],
            "action": "request_more_info"
        },
        "email": {
            "from": "partial.info@email.com",
            "subject": "Warranty for my hair dryer",
            "body": """
Hello,

I bought one of your hair dryers from Amazon about 2 months ago. I think it
was the travel one? The small pink one. Anyway, it stopped working.

I have my Amazon order somewhere but I can't find the serial number on the
product - where is it located?

Can you help?

Amy
            """.strip()
        }
    })

    return claims


def save_claims(claims: list, category: str):
    """Save claims to JSON files."""
    category_dir = TEST_CLAIMS_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    for claim in claims:
        filepath = category_dir / f"{claim['id']}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(claim, f, indent=2, ensure_ascii=False)
        print(f"Created: {filepath}")


def save_all_claims(all_claims: list):
    """Save all claims to a single dataset file."""
    filepath = TEST_CLAIMS_DIR / "test_dataset.json"
    dataset = {
        "generated_date": TODAY.isoformat(),
        "total_claims": len(all_claims),
        "distribution": {
            "spam": len([c for c in all_claims if c["type"] in ["spam", "inquiry"]]),
            "valid": len([c for c in all_claims if c["type"] == "valid_claim"]),
            "invalid": len([c for c in all_claims if c["type"].startswith("invalid")]),
        },
        "claims": all_claims
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nCreated master dataset: {filepath}")


def main():
    """Generate all test claims."""
    ensure_directories()

    print("Generating synthetic warranty claim emails...")
    print(f"Output directory: {TEST_CLAIMS_DIR}")
    print("-" * 50)

    # Generate claims
    spam_claims = generate_spam_claims()
    valid_claims = generate_valid_claims()
    invalid_claims = generate_invalid_claims()

    # Save to individual files
    save_claims(spam_claims, "spam")
    save_claims(valid_claims, "valid")
    save_claims(invalid_claims, "invalid")

    # Combine and save master dataset
    all_claims = spam_claims + valid_claims + invalid_claims
    save_all_claims(all_claims)

    print("-" * 50)
    print(f"Generated {len(all_claims)} test claims:")
    print(f"  - Spam/Irrelevant: {len(spam_claims)}")
    print(f"  - Valid Claims: {len(valid_claims)}")
    print(f"  - Invalid Claims: {len(invalid_claims)}")


if __name__ == "__main__":
    main()
