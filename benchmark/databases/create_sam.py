"""
Generate sam.db with synthetic Software Asset Management data.
Fixed random seed for reproducibility.
"""
import sqlite3
import random
from pathlib import Path
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

DB_PATH = Path(__file__).parent / "sam.db"

VENDORS = [
    "Adobe", "Microsoft", "Salesforce", "Oracle", "SAP",
    "ServiceNow", "Workday", "Atlassian", "GitHub", "Datadog",
    "Snowflake", "Zoom", "Slack", "AWS", "Google Cloud",
]

PRODUCTS = {
    "Adobe":       ["Creative Cloud", "Acrobat Pro", "Sign"],
    "Microsoft":   ["Office 365", "Azure DevOps", "Teams", "Defender"],
    "Salesforce":  ["Sales Cloud", "Service Cloud", "Marketing Cloud"],
    "Oracle":      ["Database EE", "Java SE", "Analytics Cloud"],
    "SAP":         ["S/4HANA", "SuccessFactors", "Ariba"],
    "ServiceNow":  ["ITSM", "HRSD", "CSM"],
    "Workday":     ["HCM", "Financial Management", "Recruiting"],
    "Atlassian":   ["Jira", "Confluence", "Bitbucket"],
    "GitHub":      ["Enterprise", "Copilot"],
    "Datadog":     ["Infrastructure", "APM", "Log Management"],
    "Snowflake":   ["Enterprise", "Business Critical"],
    "Zoom":        ["Business", "Phone", "Webinar"],
    "Slack":       ["Pro", "Business+", "Enterprise Grid"],
    "AWS":         ["Support Enterprise", "Marketplace"],
    "Google Cloud":["Workspace Business", "Workspace Enterprise"],
}

DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance",
    "HR", "Legal", "Operations", "Product", "Design", "IT"
]

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def create_sam_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE products (
            product_id    TEXT PRIMARY KEY,
            vendor_name   TEXT NOT NULL,
            product_name  TEXT NOT NULL,
            category      TEXT NOT NULL
        );
        CREATE TABLE contracts (
            contract_id   TEXT PRIMARY KEY,
            product_id    TEXT NOT NULL,
            department    TEXT NOT NULL,
            seats         INTEGER NOT NULL,
            annual_cost   REAL NOT NULL,
            start_date    TEXT NOT NULL,
            expiry_date   TEXT NOT NULL,
            renewal_flag  TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        CREATE TABLE entitlements (
            entitlement_id TEXT PRIMARY KEY,
            contract_id    TEXT NOT NULL,
            user_email     TEXT NOT NULL,
            last_used      TEXT,
            usage_days_90  INTEGER DEFAULT 0,
            FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
        );
    """)

    # Generate products
    products = []
    product_id = 1
    for vendor, prods in PRODUCTS.items():
        for prod in prods:
            category = (
                "Productivity" if any(k in prod for k in ["Office", "Teams", "Slack", "Zoom"]) else
                "Development"  if any(k in prod for k in ["DevOps", "GitHub", "Jira", "Bitbucket"]) else
                "Infrastructure" if any(k in prod for k in ["Azure", "AWS", "Cloud", "Datadog"]) else
                "Business"
            )
            products.append((f"PROD-{product_id:03d}", vendor, prod, category))
            product_id += 1

    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?)", products
    )

    # Generate contracts
    contracts = []
    today = date(2025, 1, 1)
    renewal_flags = ["auto", "manual", "none"]

    for i, (prod_id, vendor, prod_name, _) in enumerate(products):
        n_contracts = random.randint(1, 3)
        for j in range(n_contracts):
            start = random_date(date(2022, 1, 1), date(2024, 6, 30))
            duration_months = random.choice([12, 24, 36])
            expiry = date(
                start.year + (start.month + duration_months - 1) // 12,
                (start.month + duration_months - 1) % 12 + 1,
                1
            )
            seats = random.randint(5, 500)
            annual_cost = round(seats * random.uniform(50, 500), 2)
            dept = random.choice(DEPARTMENTS)
            renewal = random.choices(renewal_flags, weights=[50, 35, 15])[0]

            contracts.append((
                f"CON-{i*3+j+1:04d}", prod_id, dept,
                seats, annual_cost,
                start.isoformat(), expiry.isoformat(), renewal
            ))

    conn.executemany(
        "INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?)", contracts
    )

    # Generate entitlements
    entitlements = []
    ent_id = 1
    domains = ["cisco.com", "example.com", "corp.internal"]

    for con_id, prod_id, dept, seats, _, _, _, _ in contracts:
        # Assign 60-110% of seats (some over-provisioned, some under)
        actual_users = int(seats * random.uniform(0.60, 1.10))
        actual_users = max(1, min(actual_users, seats + 10))

        for u in range(actual_users):
            last_used = random_date(
                date(2024, 10, 1), date(2025, 1, 1)
            ).isoformat() if random.random() > 0.20 else None
            usage_days = random.randint(0, 90) if last_used else 0

            entitlements.append((
                f"ENT-{ent_id:06d}",
                con_id,
                f"user{ent_id}@{random.choice(domains)}",
                last_used,
                usage_days,
            ))
            ent_id += 1

    conn.executemany(
        "INSERT INTO entitlements VALUES (?,?,?,?,?)", entitlements
    )

    conn.commit()
    conn.close()

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    counts = {
        "products":     conn.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        "contracts":    conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0],
        "entitlements": conn.execute("SELECT COUNT(*) FROM entitlements").fetchone()[0],
    }

    # Interesting stats
    expiring_q1 = conn.execute("""
        SELECT COUNT(*) FROM contracts
        WHERE expiry_date BETWEEN '2025-01-01' AND '2025-03-31'
    """).fetchone()[0]
    unused = conn.execute("""
        SELECT COUNT(*) FROM entitlements
        WHERE usage_days_90 = 0
    """).fetchone()[0]
    conn.close()

    print(f"Created {DB_PATH}")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")
    print(f"  contracts expiring Q1 2025: {expiring_q1}")
    print(f"  entitlements with 0 usage:  {unused}")

if __name__ == "__main__":
    create_sam_db()
