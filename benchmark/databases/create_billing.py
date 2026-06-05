"""
Generate billing.db with synthetic enterprise data.
Fixed random seed for reproducibility across all experiment runs.
"""
import sqlite3
import random
from pathlib import Path
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

DB_PATH = Path(__file__).parent / "billing.db"

VENDORS = [
    "Adobe", "Microsoft", "Salesforce", "Oracle", "SAP",
    "ServiceNow", "Workday", "Slack", "Zoom", "Atlassian",
    "GitHub", "AWS", "Google Cloud", "Datadog", "Snowflake",
]

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def create_billing_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE invoices (
            invoice_id    TEXT PRIMARY KEY,
            vendor_name   TEXT NOT NULL,
            amount_usd    REAL NOT NULL,
            invoice_date  TEXT NOT NULL,
            quarter       TEXT NOT NULL,
            status        TEXT NOT NULL
        );
        CREATE TABLE purchase_orders (
            po_id         TEXT PRIMARY KEY,
            vendor_name   TEXT NOT NULL,
            amount_usd    REAL NOT NULL,
            po_date       TEXT NOT NULL,
            approved_by   TEXT NOT NULL
        );
        CREATE TABLE payments (
            payment_id    TEXT PRIMARY KEY,
            invoice_id    TEXT NOT NULL,
            amount_paid   REAL NOT NULL,
            payment_date  TEXT NOT NULL,
            dispute_flag  INTEGER DEFAULT 0,
            FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
        );
    """)

    # Generate invoices
    invoices = []
    quarters = {
        "Q1": (date(2024, 1, 1), date(2024, 3, 31)),
        "Q2": (date(2024, 4, 1), date(2024, 6, 30)),
        "Q3": (date(2024, 7, 1), date(2024, 9, 30)),
        "Q4": (date(2024, 10, 1), date(2024, 12, 31)),
    }
    statuses = ["paid", "pending", "disputed"]

    for i in range(1, 201):
        vendor = random.choice(VENDORS)
        quarter = random.choice(list(quarters.keys()))
        start, end = quarters[quarter]
        inv_date = random_date(start, end)
        amount = round(random.uniform(500, 50000), 2)
        status = random.choices(statuses, weights=[70, 20, 10])[0]
        invoices.append((
            f"INV-2024-{i:04d}", vendor, amount,
            inv_date.isoformat(), quarter, status
        ))

    conn.executemany(
        "INSERT INTO invoices VALUES (?,?,?,?,?,?)", invoices
    )

    # Generate purchase orders (most invoices have matching POs)
    approvers = ["john.smith", "sarah.jones", "mike.chen", "lisa.wang"]
    pos = []
    for i, (inv_id, vendor, amount, inv_date, quarter, _) in enumerate(invoices):
        if random.random() < 0.85:  # 85% have matching PO
            # Introduce discrepancies for 15% of matched POs
            po_amount = amount if random.random() > 0.15 else round(amount * random.uniform(0.85, 1.15), 2)
            pos.append((
                f"PO-2024-{i+1:04d}", vendor, po_amount,
                inv_date, random.choice(approvers)
            ))

    conn.executemany(
        "INSERT INTO purchase_orders VALUES (?,?,?,?,?)", pos
    )

    # Generate payments
    payments = []
    for i, (inv_id, _, amount, inv_date, _, status) in enumerate(invoices):
        if status == "paid":
            pay_date = date.fromisoformat(inv_date) + timedelta(days=random.randint(15, 45))
            dispute = 0
            payments.append((
                f"PAY-2024-{i+1:04d}", inv_id, amount,
                pay_date.isoformat(), dispute
            ))
        elif status == "disputed":
            pay_date = date.fromisoformat(inv_date) + timedelta(days=random.randint(30, 90))
            payments.append((
                f"PAY-2024-{i+1:04d}", inv_id, amount * 0.5,
                pay_date.isoformat(), 1
            ))

    conn.executemany(
        "INSERT INTO payments VALUES (?,?,?,?,?)", payments
    )

    conn.commit()
    conn.close()

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    counts = {
        "invoices": conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0],
        "purchase_orders": conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0],
        "payments": conn.execute("SELECT COUNT(*) FROM payments").fetchone()[0],
    }
    conn.close()
    print(f"Created {DB_PATH}")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

if __name__ == "__main__":
    create_billing_db()
