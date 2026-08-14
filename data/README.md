# Kilimo training data

## Seed corpus

[`seed/ag_qa.jsonl`](seed/ag_qa.jsonl) holds curated East African smallholder Q&A
(maize, cassava, beans, coffee, banana, plus a few general extension topics).
Rows include `id`, `crop`, `topic`, `region`, `question`, `answer`. Optional
`language` (`sw`) marks Swahili examples.

This is an original teaching corpus written for the ADTC scaffold. It is **not** a
verbatim dump of any single government manual. Before a serious training run,
expand it with:

- Locally licensed extension leaflets you have permission to use
- Paraphrases of public FAQ content (rewrite; do not paste copyrighted manuals)
- Farmer-group workshop notes you own

## Build

```bash
python scripts/build_dataset.py --out data/build/train.jsonl
python scripts/build_dataset.py --out data/build/train.jsonl --augment
```

`data/build/` is gitignored. Output is Qwen-style chat JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Adding rows

Append one JSON object per line to `seed/ag_qa.jsonl`. Keep answers practical,
stepwise, and honest about uncertainty. Prefer low-cost interventions first.
Do not invent unregistered pesticide brand claims.

## Eval separation

Held-out prompts for manual quality checks live in [`../eval/prompts.jsonl`](../eval/prompts.jsonl).
Do not copy those prompts into the seed file.
