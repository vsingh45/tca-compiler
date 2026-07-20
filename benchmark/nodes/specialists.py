"""
Specialist node implementations for TCA-Compiler benchmark.

Each node inherits from BaseNode and implements:
- build_prompt(): constructs the LLM prompt with DB schema + query
- parse_result(): extracts structured output from LLM response

Nodes:
- ExtractNode:      entity extraction from natural language query
- SQLGenNode:       SQL generation + execution against SQLite
- BillingReconNode: invoice vs PO reconciliation
- IAMAuditNode:     permission analysis
- PolicyCheckNode:  compliance assessment
- CrossReconNode:   cross-system reconciliation
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from .base import BaseNode


# ── Schema descriptions for LLM context ──────────────────────────────────────

BILLING_SCHEMA = """
Tables:
- invoices(invoice_id, vendor_name, amount_usd, invoice_date, quarter, status)
  status: 'paid' | 'pending' | 'disputed'
- purchase_orders(po_id, vendor_name, amount_usd, po_date, approved_by)
- payments(payment_id, invoice_id, amount_paid, payment_date, dispute_flag)
"""

SAM_SCHEMA = """
Tables:
- products(product_id, vendor_name, product_name, category)
- contracts(contract_id, product_id, department, seats, annual_cost,
            start_date, expiry_date, renewal_flag)
  renewal_flag: 'auto' | 'manual' | 'none'
- entitlements(entitlement_id, contract_id, user_email,
               last_used, usage_days_90)
"""

IAM_SCHEMA = """
Tables:
- users(user_id, email, department, role, status, hire_date, manager_id)
  status: 'active' | 'inactive' | 'contractor'
  role: 'engineer' | 'manager' | 'analyst' | 'admin' | 'contractor' | 'auditor'
- permissions(permission_id, user_id, resource, access_level,
              granted_date, granted_by, expiry_date)
  access_level: 'read' | 'write' | 'admin'
- access_logs(log_id, user_id, resource, action, timestamp, success)
"""


def _get_schema(db_name: str) -> str:
    return {
        "billing": BILLING_SCHEMA,
        "sam":     SAM_SCHEMA,
        "iam":     IAM_SCHEMA,
    }.get(db_name, BILLING_SCHEMA)


def _safe_execute(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Execute SQL safely, return list of row dicts."""
    try:
        # Only allow SELECT statements
        clean = sql.strip().rstrip(";")
        if not clean.upper().startswith("SELECT"):
            return []
        cursor = conn.execute(clean)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchmany(20)]
    except Exception as e:
        return [{"error": str(e)}]


# ── Specialist Nodes ──────────────────────────────────────────────────────────

class ExtractNode(BaseNode):
    """
    Extracts entities and key concepts from the user query.
    Runs first in every workflow — provides context for downstream nodes.
    """
    node_class = "extract"

    def build_prompt(self, query, context, db_conn):
        return (
            f"Extract the key entities, time periods, vendors, and data "
            f"requirements from this enterprise query. Be concise.\n\n"
            f"Query: {query}\n\n"
            f"Respond with: entities found, time period, primary data source needed, "
            f"key metrics or fields required."
        )

    def parse_result(self, response_text, query, db_conn):
        return {
            "extracted": response_text,
            "query": query,
        }


class SQLGenNode(BaseNode):
    """
    Generates and executes SQL against the benchmark database.
    Returns SQL query + executed results.
    """
    node_class = "sql-gen"

    def build_prompt(self, query, context, db_conn):
        # Get the DB name from connection path
        db_path = db_conn.execute("PRAGMA database_list").fetchone()[2]
        db_name = "sam" if "sam" in db_path else "iam" if "iam" in db_path else "billing"
        schema = _get_schema(db_name)

        context_block = ""
        if context:
            context_block = "\nPrior context:\n" + "\n".join(
                f"- {c[:200]}" for c in context[:3]
            )

        return (
            f"Write a SQLite SELECT query to answer this question.\n\n"
            f"Schema:\n{schema}\n"
            f"{context_block}\n"
            f"Question: {query}\n\n"
            f"Return ONLY the SQL query, nothing else. "
            f"Use SQLite syntax. Limit results to 20 rows."
        )

    def parse_result(self, response_text, query, db_conn):
        # Extract SQL from response
        sql_match = re.search(
            r"```sql\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE
        )
        if sql_match:
            sql = sql_match.group(1).strip()
        else:
            # Try to find raw SQL
            lines = [
                l for l in response_text.split("\n")
                if l.strip().upper().startswith("SELECT")
            ]
            sql = lines[0].strip() if lines else response_text.strip()

        results = _safe_execute(db_conn, sql)
        return {
            "sql": sql,
            "results": results,
            "row_count": len(results),
        }

    def evaluate(self, structured, answer_text, task):
        """SQL node: evaluate based on topics + whether SQL executed."""
        required = task.get("required_topics", [])

        # Check if SQL ran successfully
        results = structured.get("results", [])
        sql_executed = len(results) > 0 and "error" not in results[0]

        # Check topic recall
        answer_lower = answer_text.lower()
        sql_lower = structured.get("sql", "").lower()
        combined = answer_lower + " " + sql_lower

        hits = sum(
            1 for topic in required
            if topic.lower().replace("-", " ") in combined
            or topic.lower() in combined
        )
        recall = hits / len(required) if required else 1.0

        # Both SQL execution and topic recall required
        correct = sql_executed and recall >= 0.5
        return correct, recall


class BillingReconNode(BaseNode):
    """
    Reconciles invoices against purchase orders.
    Identifies discrepancies and flags issues.
    """
    node_class = "billing-recon"

    def build_prompt(self, query, context, db_conn):
        # Get summary stats from DB for context
        try:
            discrepancies = db_conn.execute("""
                SELECT i.vendor_name,
                       i.amount_usd as invoice_amt,
                       p.amount_usd as po_amt,
                       ABS(i.amount_usd - p.amount_usd) as diff
                FROM invoices i
                LEFT JOIN purchase_orders p
                  ON i.vendor_name = p.vendor_name
                WHERE p.amount_usd IS NOT NULL
                  AND ABS(i.amount_usd - p.amount_usd) > 1000
                LIMIT 10
            """).fetchall()
            disc_text = "\n".join(
                f"  {r[0]}: invoice=${r[1]:,.0f} PO=${r[2]:,.0f} diff=${r[3]:,.0f}"
                for r in discrepancies
            ) if discrepancies else "  None found above $1,000"
        except Exception:
            disc_text = "  Unable to query discrepancies"

        context_block = ""
        if context:
            context_block = "\nPrior agent results:\n" + "\n".join(
                f"- {c[:300]}" for c in context[:3]
            )

        return (
            f"Reconcile billing records and identify discrepancies.\n\n"
            f"Current discrepancies (invoice vs PO, diff > $1,000):\n{disc_text}\n"
            f"{context_block}\n"
            f"Task: {query}\n\n"
            f"Analyze the discrepancies, identify patterns, and summarize findings. "
            f"Be specific about vendors and amounts."
        )

    def parse_result(self, response_text, query, db_conn):
        return {
            "reconciliation": response_text,
            "query": query,
        }


class IAMAuditNode(BaseNode):
    """
    Analyzes IAM permissions for policy violations and risks.
    """
    node_class = "iam-audit"

    def build_prompt(self, query, context, db_conn):
        # Get relevant IAM stats
        try:
            admin_count = db_conn.execute(
                "SELECT COUNT(*) FROM permissions WHERE access_level='admin'"
            ).fetchone()[0]
            inactive_with_access = db_conn.execute("""
                SELECT COUNT(DISTINCT u.user_id)
                FROM users u JOIN permissions p ON u.user_id = p.user_id
                WHERE u.status = 'inactive'
            """).fetchone()[0]
            contractor_count = db_conn.execute(
                "SELECT COUNT(*) FROM users WHERE status='contractor'"
            ).fetchone()[0]
            stats = (
                f"  Admin permissions: {admin_count}\n"
                f"  Inactive users with active permissions: {inactive_with_access}\n"
                f"  Active contractors: {contractor_count}"
            )
        except Exception:
            stats = "  Unable to query IAM stats"

        context_block = ""
        if context:
            context_block = "\nPrior agent results:\n" + "\n".join(
                f"- {c[:300]}" for c in context[:3]
            )

        return (
            f"Analyze IAM permissions and identify access risks.\n\n"
            f"Current IAM stats:\n{stats}\n"
            f"{context_block}\n"
            f"Task: {query}\n\n"
            f"Identify specific users, permissions, or patterns that violate "
            f"least-privilege principles or pose security risks."
        )

    def parse_result(self, response_text, query, db_conn):
        return {
            "audit_findings": response_text,
            "query": query,
        }


class PolicyCheckNode(BaseNode):
    """
    Checks findings against enterprise policy rules.
    Flags violations and recommends actions.
    """
    node_class = "policy-check"

    # Simplified policy rules
    POLICIES = """
    Policy Rules:
    1. Invoices over $50,000 require VP-level approval
    2. Admin permissions require dual approval
    3. Contractor access must expire within 90 days
    4. Production write access restricted to Engineering and IT
    5. Software purchases must use preferred vendor list
    6. Invoices without matching POs require CFO exception
    7. License renewals need 90-day advance notice
    8. Financial system access requires segregation of duties
    """

    def build_prompt(self, query, context, db_conn):
        context_block = ""
        if context:
            context_block = "\nFindings from prior agents:\n" + "\n".join(
                f"- {c[:400]}" for c in context[:4]
            )

        return (
            f"Check the following findings against enterprise policy.\n\n"
            f"{self.POLICIES}\n"
            f"{context_block}\n"
            f"Task: {query}\n\n"
            f"For each policy violation found: name the policy, describe the "
            f"violation, assess severity (low/medium/high/critical), and "
            f"recommend specific remediation action."
        )

    def parse_result(self, response_text, query, db_conn):
        # Extract severity mentions
        severities = re.findall(
            r"\b(low|medium|high|critical)\b", response_text.lower()
        )
        return {
            "policy_findings": response_text,
            "severities_found": list(set(severities)),
            "violation_count": len(severities),
        }


class CrossReconNode(BaseNode):
    """
    Reconciles data across multiple systems (billing + SAM + IAM).
    Identifies cross-system inconsistencies.
    """
    node_class = "cross-recon"

    def build_prompt(self, query, context, db_conn):
        context_block = ""
        if context:
            context_block = "\nResults from all prior agents:\n" + "\n".join(
                f"[Agent {i+1}]: {c[:400]}"
                for i, c in enumerate(context[:5])
            )

        return (
            f"Perform cross-system reconciliation and identify inconsistencies.\n\n"
            f"{context_block}\n"
            f"Task: {query}\n\n"
            f"Compare findings across billing, SAM, and IAM systems. "
            f"Identify: (1) data present in one system but missing in another, "
            f"(2) values that conflict between systems, "
            f"(3) users or vendors that appear anomalous across systems. "
            f"Provide specific examples with system references."
        )

    def parse_result(self, response_text, query, db_conn):
        return {
            "cross_system_findings": response_text,
            "query": query,
        }


# ── Node factory ─────────────────────────────────────────────────────────────

NODE_CLASSES = {
    "extract":       ExtractNode,
    "sql-gen":       SQLGenNode,
    "billing-recon": BillingReconNode,
    "iam-audit":     IAMAuditNode,
    "policy-check":  PolicyCheckNode,
    "cross-recon":   CrossReconNode,
}

def make_node(
    node_class: str,
    client,
    profiler,
    workflow_id: str,
    node_id: str,
    tier: str,
    memory_strategy: str,
    condition: str,
    shared_capacity: int = 32,
) -> BaseNode:
    """Factory: create the right node instance for a given node_class."""
    cls = NODE_CLASSES.get(node_class)
    if cls is None:
        raise ValueError(
            f"Unknown node_class: {node_class}. "
            f"Available: {list(NODE_CLASSES.keys())}"
        )
    return cls(
        client=client,
        profiler=profiler,
        workflow_id=workflow_id,
        node_id=node_id,
        tier=tier,
        memory_strategy=memory_strategy,
        condition=condition,
        shared_capacity=shared_capacity,
    )
