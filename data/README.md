# Kilimo training data

## Seed corpus

Every `*.jsonl` shard in `seed/` is merged by the builder:

- [`seed/ag_qa.jsonl`](seed/ag_qa.jsonl) — staple crops: maize, cassava, beans, coffee, banana
- [`seed/ag_qa_extended.jsonl`](seed/ag_qa_extended.jsonl) — livestock, poultry, vegetables,
  soil, irrigation, markets, weather, aquaculture, agroforestry, pesticide safety

Rows include `id`, `crop`, `topic`, `region`, `question`, `answer`. Optional
`language` (`sw`) marks Swahili examples. Duplicate `id`s across shards are a
build error.

The extended shard exists because the challenge defines agriculture as "crop,
livestock, weather, and market advisory". The two hidden judge prompts can land
anywhere in that space, so a corpus covering only staple crops would overfit.

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

`--augment` adds one paraphrased copy per row. Output is shuffled.

`data/build/` is gitignored. Output is Qwen-style chat JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Adding rows

Append one JSON object per line to any shard in `seed/`, or add a new shard.
Keep answers practical, stepwise, and honest about uncertainty. Prefer low-cost
interventions first. Do not invent unregistered pesticide brand claims, and do
not state specific dose rates that vary by district — point to the label and the
local extension officer instead.

## Eval separation

Held-out prompts for manual quality checks live in [`../eval/prompts.jsonl`](../eval/prompts.jsonl).
Do not copy those prompts into the seed file.
