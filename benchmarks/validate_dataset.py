"""Validate the controlled QA benchmark dataset.

Usage (from the repository root, with .venv activated):

    python3 benchmarks/validate_dataset.py
    python3 benchmarks/validate_dataset.py path/to/other.jsonl
    python3 benchmarks/validate_dataset.py --strict   # warnings fail too

Exit code is 0 if there are no errors, 1 otherwise.

Errors  = the case would give wrong metrics (must fix or delete the case).
Warnings = the case is probably weak or off-spec (review it).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "datasets" / "controlled_qa.jsonl"

REQUIRED_FIELDS = {
    "case_id",
    "category",
    "history",
    "query",
    "gold_answer",
    "evidence_turn_ids",
}

# category -> (min_turns, max_turns), taken from the generation prompt
CATEGORY_LENGTHS = {
    "lexical": (20, 40),
    "semantic": (20, 40),
    "buried": (80, 150),
    "duplicates": (30, 60),
    "long_evidence": (15, 30),
    "multi_evidence": (30, 60),
    "recency": (20, 40),
}

STOPWORDS = {
    "about", "after", "again", "also", "because", "been", "before", "being",
    "could", "does", "from", "have", "into", "just", "like", "make", "more",
    "need", "only", "other", "over", "please", "really", "should", "some",
    "that", "their", "them", "then", "there", "these", "they", "this",
    "very", "want", "were", "what", "when", "where", "which", "while",
    "will", "with", "would", "your", "you", "the", "and", "for", "are",
}


def normalize_category(raw: object) -> str:
    """Accept 'buried', 'C) buried', 'Buried' etc."""
    text = str(raw).strip().lower()
    return re.sub(r"^[a-g]\)\s*", "", text)


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_case(case: object) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(case, dict):
        return ["line is not a JSON object"], warnings

    missing = REQUIRED_FIELDS - case.keys()
    if missing:
        return [f"missing fields: {sorted(missing)}"], warnings

    case_id = case["case_id"]
    category = normalize_category(case["category"])
    history = case["history"]
    query = case["query"]
    gold = case["gold_answer"]
    evidence_ids = case["evidence_turn_ids"]

    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id must be a non-empty string")
    if category not in CATEGORY_LENGTHS:
        errors.append(
            f"unknown category {case['category']!r}; expected one of "
            f"{sorted(CATEGORY_LENGTHS)}"
        )
    if not isinstance(query, str) or not query.strip():
        errors.append("query must be a non-empty string")
    if not isinstance(gold, str) or not gold.strip():
        errors.append("gold_answer must be a non-empty string")

    # History structure
    if not isinstance(history, list) or not history:
        errors.append("history must be a non-empty list")
        return errors, warnings

    for i, turn in enumerate(history):
        if not isinstance(turn, dict):
            errors.append(f"turn {i} is not an object")
            continue
        role = turn.get("role")
        content = turn.get("content")
        expected = "user" if i % 2 == 0 else "assistant"
        if role not in ("user", "assistant"):
            errors.append(f"turn {i}: invalid role {role!r}")
        elif role != expected:
            errors.append(f"turn {i}: role {role!r}, expected {expected!r} (roles must alternate from user)")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"turn {i}: content must be a non-empty string")

    if errors:
        # Later checks assume well-formed turns and strings.
        return errors, warnings

    contents = [t["content"] for t in history]
    n = len(history)

    # Evidence ids
    if not isinstance(evidence_ids, list) or not evidence_ids:
        errors.append("evidence_turn_ids must be a non-empty list")
        return errors, warnings
    if not all(is_int(i) for i in evidence_ids):
        errors.append("evidence_turn_ids must contain only integers")
        return errors, warnings
    bad = [i for i in evidence_ids if i < 0 or i >= n]
    if bad:
        errors.append(f"evidence_turn_ids out of range 0..{n - 1}: {bad}")
        return errors, warnings
    if len(set(evidence_ids)) != len(evidence_ids):
        errors.append("evidence_turn_ids contains duplicates")

    evidence_set = set(evidence_ids)

    # Gold answer parts ("A | B" for multi_evidence)
    parts = [p.strip() for p in gold.split(" | ")]
    if any(not p for p in parts):
        errors.append("gold_answer has an empty part around ' | '")
        return errors, warnings
    if category == "multi_evidence":
        if len(parts) < 2:
            errors.append("multi_evidence gold_answer should have parts joined by ' | '")
        if len(evidence_ids) < 2:
            errors.append("multi_evidence needs at least 2 evidence turns")
    elif len(parts) > 1:
        warnings.append("gold_answer contains ' | ' but category is not multi_evidence")

    for part in parts:
        if len(part.split()) > 6:
            warnings.append(f"gold part {part!r} is longer than 6 words")

    # Presence: verbatim, case-sensitive, in evidence turns
    if len(parts) == 1:
        for i in evidence_ids:
            if parts[0] not in contents[i]:
                errors.append(f"gold_answer {parts[0]!r} not found verbatim in evidence turn {i}")
    else:
        for part in parts:
            if not any(part in contents[i] for i in evidence_ids):
                errors.append(f"gold part {part!r} not found verbatim in any evidence turn")
        for i in evidence_ids:
            if not any(part in contents[i] for part in parts):
                errors.append(f"evidence turn {i} contains none of the gold parts")

    # Leakage: no gold part (even case-insensitively) in non-evidence turns
    for i, text in enumerate(contents):
        if i in evidence_set:
            continue
        lowered = text.lower()
        for part in parts:
            if part.lower() in lowered:
                errors.append(f"gold part {part!r} leaks into non-evidence turn {i}")

    # Query sanity
    if isinstance(query, str):
        q = query.lower()
        for part in parts:
            if part.lower() in q:
                warnings.append(f"query contains gold part {part!r} (answer is given away)")

    # Category-specific soft checks
    if category in CATEGORY_LENGTHS:
        lo, hi = CATEGORY_LENGTHS[category]
        if not lo <= n <= hi:
            warnings.append(f"history has {n} turns; {category} expects {lo}-{hi}")

    if category == "semantic":
        evidence_words: set[str] = set()
        for i in evidence_ids:
            evidence_words |= content_words(contents[i])
        shared = content_words(query) & evidence_words
        if shared:
            warnings.append(
                f"semantic query shares content words with evidence: {sorted(shared)}"
            )

    if category == "long_evidence":
        for i in evidence_ids:
            words = len(contents[i].split())
            if not 150 <= words <= 300:
                warnings.append(f"evidence turn {i} has {words} words; long_evidence expects 150-300")

    if category == "buried":
        for i in evidence_ids:
            if not n / 3 <= i <= 2 * n / 3:
                warnings.append(f"evidence turn {i} is not in the middle third of {n} turns")

    if category == "recency":
        for i in evidence_ids:
            if i < n - 2:
                warnings.append(f"recency evidence turn {i} is not in the last 2 turns")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH, help="JSONL file to validate")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()

    path: Path = args.path
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    total_errors = 0
    total_warnings = 0
    n_cases = 0
    n_clean = 0
    seen_ids: dict[str, int] = {}
    by_category: Counter[str] = Counter()
    lengths: list[int] = []

    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("```"):
                print(f"line {line_no}: [warn] markdown fence ignored")
                total_warnings += 1
                continue

            n_cases += 1
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"line {line_no}: [ERROR] invalid JSON: {exc}")
                total_errors += 1
                continue

            errors, warnings = validate_case(case)

            label = case.get("case_id", "?") if isinstance(case, dict) else "?"
            if isinstance(case, dict) and isinstance(case.get("case_id"), str):
                cid = case["case_id"]
                if cid in seen_ids:
                    errors.append(f"duplicate case_id (first seen on line {seen_ids[cid]})")
                else:
                    seen_ids[cid] = line_no
                by_category[normalize_category(case.get("category", "?"))] += 1
                if isinstance(case.get("history"), list):
                    lengths.append(len(case["history"]))

            for msg in errors:
                print(f"line {line_no} [{label}] ERROR: {msg}")
            for msg in warnings:
                print(f"line {line_no} [{label}] warn:  {msg}")

            total_errors += len(errors)
            total_warnings += len(warnings)
            if not errors and not (warnings and args.strict):
                n_clean += 1

    print()
    print(f"File:     {path}")
    print(f"Cases:    {n_cases}  (usable: {n_clean})")
    print(f"Errors:   {total_errors}")
    print(f"Warnings: {total_warnings}")
    if by_category:
        print("By category:")
        for name, count in sorted(by_category.items()):
            print(f"  {name:<16} {count}")
    if lengths:
        print(f"History length: min {min(lengths)}, max {max(lengths)}, mean {sum(lengths) / len(lengths):.1f}")

    if total_errors or (args.strict and total_warnings):
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())