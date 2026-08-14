#!/usr/bin/env python3
"""Build Qwen2.5 chat-format JSONL from the Kilimo seed corpus."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Kilimo, an offline agricultural advisor for East African smallholder "
    "farmers and extension officers. Give practical, stepwise answers grounded in "
    "rainfed smallholder conditions. Prefer low-cost interventions first. When "
    "uncertain, say so and recommend consulting a local extension officer. Do not "
    "invent pesticide product names or illegal advice."
)

QUESTION_PARAPHRASES = [
    "Please advise: {q}",
    "Extension question from my farm: {q}",
    "I need practical help: {q}",
    "{q} Keep the answer actionable for a smallholder.",
]

ANSWER_PREFIXES = [
    "",
    "Here is practical guidance. ",
    "Short answer for your situation: ",
]


def load_seed(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def to_messages(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def augment_row(row: dict, rng: random.Random) -> list[dict]:
    q = row["question"].strip()
    a = row["answer"].strip()
    out = [to_messages(q, a)]
    template = rng.choice(QUESTION_PARAPHRASES)
    prefix = rng.choice(ANSWER_PREFIXES)
    out.append(to_messages(template.format(q=q), f"{prefix}{a}".strip()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("data/seed"),
        help="Directory of seed JSONL shards (all *.jsonl are merged)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/build/train.jsonl"),
        help="Output chat JSONL",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Emit one paraphrased copy per seed row",
    )
    parser.add_argument("--seed-rng", type=int, default=42, help="RNG seed for augment")
    args = parser.parse_args()

    if not args.seed_dir.is_dir():
        raise SystemExit(f"Seed directory not found: {args.seed_dir}")

    shards = sorted(args.seed_dir.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"No *.jsonl seed shards in {args.seed_dir}")

    rows: list[dict] = []
    for shard in shards:
        rows.extend(load_seed(shard))

    seen: set[str] = set()
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            raise SystemExit(f"Duplicate seed id across shards: {rid}")
        seen.add(rid)

    rng = random.Random(args.seed_rng)
    examples: list[dict] = []
    for row in rows:
        if "question" not in row or "answer" not in row:
            raise SystemExit(f"Seed row missing question/answer: {row.get('id')}")
        if args.augment:
            examples.extend(augment_row(row, rng))
        else:
            examples.append(to_messages(row["question"].strip(), row["answer"].strip()))

    rng.shuffle(examples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Merged {len(shards)} shards / {len(rows)} rows")
    print(f"Wrote {len(examples)} examples -> {args.out}")


if __name__ == "__main__":
    main()
