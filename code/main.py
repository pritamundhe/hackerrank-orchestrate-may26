"""
main.py -- Entry point for the Multi-Domain Support Triage Agent

Usage:
    python main.py [--tickets PATH] [--output PATH] [--verbose]

Defaults:
    --tickets  ../support_tickets/support_tickets.csv
    --output   ../support_tickets/output.csv
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars print cleanly
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from agent import CorpusLoader, BM25Retriever, SafetyRouter, TriageAgent

REPO_ROOT       = Path(__file__).parent.parent.resolve()
DEFAULT_TICKETS = REPO_ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT  = REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_FIELDS = [
    "Issue", "Subject", "Company",
    "status", "product_area", "response", "justification", "request_type",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    p.add_argument("--tickets", default=str(DEFAULT_TICKETS),
                   help="Path to input CSV (default: support_tickets/support_tickets.csv)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Path to output CSV (default: support_tickets/output.csv)")
    p.add_argument("--verbose", action="store_true", help="Print per-ticket results to stdout")
    return p.parse_args()


def build_agent() -> TriageAgent:
    print("Loading corpus ...", flush=True)
    loader = CorpusLoader(REPO_ROOT / "data")
    chunks = loader.load()
    print(f"  Loaded {len(chunks)} chunks from data/", flush=True)
    print("  Building BM25 index ...", flush=True)
    retriever = BM25Retriever(chunks)
    router = SafetyRouter()
    print("  Ready.\n", flush=True)
    return TriageAgent(retriever, router)


def process_csv(agent: TriageAgent, tickets_path: Path, output_path: Path, verbose: bool) -> None:
    with open(tickets_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nProcessing {len(rows)} tickets ...\n" + "-" * 60, flush=True)

    results = []
    for i, row in enumerate(rows, 1):
        issue   = (row.get("Issue")   or "").strip()
        subject = (row.get("Subject") or "").strip()
        company = (row.get("Company") or "").strip()

        label = (subject or issue)[:55]
        print(f"[{i:02d}/{len(rows)}] {label!r}", end=" ... ", flush=True)
        t0 = time.time()
        result = agent.process(issue, subject, company)
        elapsed = time.time() - t0
        print(f"{result.status.upper()} ({elapsed:.2f}s)", flush=True)

        if verbose:
            print(f"       area={result.product_area}  type={result.request_type}")
            preview = result.response[:110].replace("\n", " ")
            print(f"       {preview}")
            print()

        results.append({
            "Issue":         issue,
            "Subject":       subject,
            "Company":       company,
            "status":        result.status,
            "product_area":  result.product_area,
            "response":      result.response,
            "justification": result.justification,
            "request_type":  result.request_type,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "-" * 60)
    print(f"Done. Output written to: {output_path}")

    replied   = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"  replied={replied}  escalated={escalated}  total={len(results)}")


def main() -> None:
    args = parse_args()
    tickets_path = Path(args.tickets)
    output_path  = Path(args.output)

    if not tickets_path.exists():
        print(f"ERROR: Tickets file not found: {tickets_path}")
        sys.exit(1)

    agent = build_agent()
    process_csv(agent, tickets_path, output_path, args.verbose)


if __name__ == "__main__":
    main()
