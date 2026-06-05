"""
Generate iam.db with synthetic Identity and Access Management data.
Fixed random seed for reproducibility.
"""
import sqlite3
import random
from pathlib import Path
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

DB_PATH = Path(__file__).parent / "iam.db"

DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance",
    "HR", "Legal", "Operations", "Product", "Design", "IT"
]

RESOURCES = [
    "prod-database", "staging-database", "dev-database",
    "finance-reports", "hr-records", "sales-crm",
    "source-code-repo", "ci-cd-pipeline", "cloud-console",
    "billing-system", "payroll-system", "legal-vault",
    "marketing-platform", "analytics-dashboard", "api-gateway",
]

ACCESS_LEVELS = ["read", "write", "admin"]
USER_ROLES = ["engineer", "manager", "analyst", "admin", "contractor", "auditor"]
USER_STATUSES = ["active", "inactive", "contractor"]

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def create_iam_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE users (
            user_id       TEXT PRIMARY KEY,
            email         TEXT NOT NULL,
            department    TEXT NOT NULL,
            role          TEXT NOT NULL,
            status        TEXT NOT NULL,
            hire_date     TEXT NOT NULL,
            manager_id    TEXT
        );
        CREATE TABLE permissions (
            permission_id TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            resource      TEXT NOT NULL,
            access_level  TEXT NOT NULL,
            granted_date  TEXT NOT NULL,
            granted_by    TEXT NOT NULL,
            expiry_date   TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE TABLE access_logs (
            log_id        TEXT PRIMARY KEY,
            user_id       TEXT NOT NULL,
            resource      TEXT NOT NULL,
            action        TEXT NOT NULL,
            timestamp     TEXT NOT NULL,
            success       INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
    """)

    # Generate users
    users = []
    domains = ["cisco.com", "example.com"]
    managers = []

    for i in range(1, 201):
        dept = random.choice(DEPARTMENTS)
        role = random.choices(
            USER_ROLES,
            weights=[35, 15, 20, 5, 15, 10]
        )[0]
        status = random.choices(
            USER_STATUSES,
            weights=[75, 10, 15]
        )[0]
        hire_date = random_date(date(2018, 1, 1), date(2024, 6, 30))
        domain = random.choice(domains)
        user_id = f"USR-{i:04d}"

        if role == "manager" and len(managers) < 20:
            managers.append(user_id)

        users.append((
            user_id,
            f"user{i}@{domain}",
            dept, role, status,
            hire_date.isoformat(),
            random.choice(managers) if managers and role != "manager" else None,
        ))

    conn.executemany(
        "INSERT INTO users VALUES (?,?,?,?,?,?,?)", users
    )

    # Generate permissions
    permissions = []
    perm_id = 1
    admin_users = [u[0] for u in users if u[3] == "admin"]
    granters = admin_users + [u[0] for u in users if u[3] == "manager"]

    for user_id, email, dept, role, status, _, _ in users:
        # Number of permissions depends on role
        n_perms = {
            "engineer": random.randint(3, 8),
            "manager":  random.randint(4, 10),
            "analyst":  random.randint(2, 6),
            "admin":    random.randint(8, 15),
            "contractor": random.randint(1, 4),
            "auditor":  random.randint(2, 5),
        }.get(role, 3)

        resources = random.sample(RESOURCES, min(n_perms, len(RESOURCES)))

        for resource in resources:
            # Access level depends on role
            if role == "admin":
                level = random.choices(ACCESS_LEVELS, weights=[20, 30, 50])[0]
            elif role in ("engineer", "manager"):
                level = random.choices(ACCESS_LEVELS, weights=[30, 50, 20])[0]
            else:
                level = random.choices(ACCESS_LEVELS, weights=[60, 35, 5])[0]

            granted = random_date(date(2022, 1, 1), date(2024, 12, 1))
            expiry = None
            if role == "contractor":
                expiry = (granted + timedelta(days=random.randint(90, 365))).isoformat()

            granter = random.choice(granters) if granters else "USR-0001"

            permissions.append((
                f"PERM-{perm_id:06d}",
                user_id, resource, level,
                granted.isoformat(), granter, expiry,
            ))
            perm_id += 1

    conn.executemany(
        "INSERT INTO permissions VALUES (?,?,?,?,?,?,?)", permissions
    )

    # Generate access logs
    logs = []
    actions = ["login", "read", "write", "delete", "export", "admin_action"]
    log_id = 1

    for user_id, _, _, role, status, _, _ in users:
        if status == "inactive":
            n_logs = random.randint(0, 3)
        elif role == "admin":
            n_logs = random.randint(20, 50)
        else:
            n_logs = random.randint(5, 25)

        user_perms = [
            p for p in permissions if p[1] == user_id
        ]
        if not user_perms:
            continue

        for _ in range(n_logs):
            perm = random.choice(user_perms)
            resource = perm[2]
            action = random.choice(actions)
            ts = random_date(date(2024, 10, 1), date(2025, 1, 1))
            success = 1 if random.random() > 0.05 else 0

            logs.append((
                f"LOG-{log_id:07d}",
                user_id, resource, action,
                f"{ts.isoformat()}T{random.randint(8,18):02d}:{random.randint(0,59):02d}:00",
                success,
            ))
            log_id += 1

    conn.executemany(
        "INSERT INTO access_logs VALUES (?,?,?,?,?,?)", logs
    )

    conn.commit()
    conn.close()

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    counts = {
        "users":        conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "permissions":  conn.execute("SELECT COUNT(*) FROM permissions").fetchone()[0],
        "access_logs":  conn.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0],
    }
    admin_perms = conn.execute("""
        SELECT COUNT(*) FROM permissions
        WHERE access_level = 'admin'
    """).fetchone()[0]
    inactive_with_access = conn.execute("""
        SELECT COUNT(DISTINCT u.user_id) FROM users u
        JOIN permissions p ON u.user_id = p.user_id
        WHERE u.status = 'inactive'
    """).fetchone()[0]
    conn.close()

    print(f"Created {DB_PATH}")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")
    print(f"  admin-level permissions: {admin_perms}")
    print(f"  inactive users with active permissions: {inactive_with_access}")

if __name__ == "__main__":
    create_iam_db()
