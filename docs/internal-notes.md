# Internal notes

Working notes for the README we will write together later. Facts and decisions
only, not prose. Every number here was verified against the shipped dataset.

## Run commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY
python run.py                 # or: uvicorn app.api.main:app
```

UI and API both on http://127.0.0.1:8000. OpenAPI docs at `/docs`.
Tests: `pytest` (142 tests, offline, no key needed, uses `LLM_PROVIDER=stub`).

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/api/query` | NL question -> plan -> result |
| POST | `/api/anomalies` | `{types, window}`; works with no LLM |
| GET | `/api/anomalies` | all detectors, all time |
| GET | `/api/health` | dataset, ingest report, LLM reachability, active thresholds |
| GET | `/api/schema` | columns, enums, reference clock, example questions |
| GET | `/` | minimal UI |

## Model choice

Primary: **`openai/gpt-oss-20b` on Groq free tier**. Fallback: `openai/gpt-oss-120b`.
Offline: Ollama `qwen3:4b` (~2.6GB) or `gpt-oss:20b` (needs ~16GB RAM).

Reasons:
- Groq free tier needs no credit card; 30 req/min, 14,400/day, org-level.
- Strict Structured Outputs (token-level constrained decoding, guaranteed
  schema-valid JSON) is limited to `openai/gpt-oss-20b`, `openai/gpt-oss-120b`,
  `qwen/qwen3.8-27b`. gpt-oss-20b is the fastest of those.
- Same model family runs locally under Ollama, which compiles a JSON schema into
  a GBNF grammar. One prompt + one schema covers hosted and offline.
- Apache 2.0, 131k context, 3.6B active params (MoE).

**`llama-3.3-70b-versatile` and `llama-3.1-8b-instant` were deprecated by Groq on
2026-06-17.** Most tutorials still reference them. `GroqProvider` raises a clear
error naming replacements if either is configured.

Groq Structured Outputs cannot be combined with tool calling or streaming, so the
plan goes through `response_format`, not function calls.

No fine-tuning: constrained decoding against a fixed schema is a better fit and
the brief gives no reason to train.

## Architecture

```
question -> LLM (schema-constrained) -> QueryPlan JSON -> Pydantic validation
         -> deterministic pandas executor -> numbers -> LLM phrases the answer
```

The model emits a closed vocabulary (known fields, fixed operators, enum values).
It never emits SQL or code. Anything outside the vocabulary fails validation and
never reaches the executor, so injection is structurally impossible rather than
filtered. Every number comes from `app/query/executor.py`, which is unit-tested
without a model. The narrator receives computed results only.

Module layout: `domain/` schema, `data/` ingest + store, `query/` plan + executor
+ timewindow, `anomaly/` detectors, `llm/` providers, `api/` FastAPI, `ui/` page.

Deliberately not used: vector DB (500 rows, no semantic search need), agent
framework (one constrained call), DuckDB (plan DSL already covers the surface).

## Dataset facts

500 rows, 0 rejected, 0 duplicate IDs, 0 parse failures. 2024-01-01 to 2024-03-30.
Columns match brief section 3.3 spelling; loader also accepts section 3.2 spelling
(`resp_time_hrs`, `resol_time_hrs`, `cust_rating`) via alias map.

- Status: Resolved 327, Open 111, Escalated 62
- Priority: Medium 169, Low 142, High 134, Critical 55
- Category: General 189, Billing 159, Technical 152
- 12 agents, 26 distinct issue summaries
- 173 nulls in `resolution_time_hrs` and `customer_rating`, aligning **exactly**
  with the 173 non-Resolved tickets
- `response_time_hrs` is uniform on [0.2, 5.0]; `resolution_time_hrs` is
  right-skewed, median 12.0, max 119.7
- **28 tickets have `resolution_time_hrs < response_time_hrs`** (chronologically
  impossible). Kept in the data, surfaced by the integrity detector, because
  dropping them would silently shift every aggregate.

## Reference clock

Relative dates resolve against the newest ticket (2024-03-30 18:06), not wall
clock. Otherwise "this month" returns zero rows and reads as a broken system.
Configurable via `REFERENCE_CLOCK=system`. Every response echoes the resolved
window label.

## Anomaly thresholds (calibrated, not guessed)

| Detector | Rule | Flags |
|---|---|---|
| resolution_outlier | IQR fence **per priority** | 18 |
| aging_unresolved | High/Critical unresolved > 24h | 80 |
| response_sla_breach | Critical 2h, High 4h, Medium 6h, Low 12h | 61 |
| low_rating | rating <= 2 | 47 |
| data_integrity | resolution < response, status/field disagreement | 28 |

Two calibration findings worth explaining in the walkthrough:

1. **Grouped by priority, not category.** Category medians are 11.4 / 12.1 / 13.2h
   (no signal). Priority medians are 3.7 / 6.9 / 14.5 / 26.6h. A global fence
   flags 21 but is dominated by Low-priority tickets normal for their class and
   misses Critical tickets at 5x their class median. Per-priority flags 18.
2. **Response time uses fixed SLA, not statistics.** Response time is uniform on
   [0.2, 5.0] so its IQR fence sits at 7.65 and would never fire. Fixed targets
   flag 61, all Critical/High. Priority and response time are uncorrelated in
   this data, which is itself a finding.

Groups smaller than 8 are skipped rather than given a meaningless fence.

## Verified example queries

| Question | Answer |
|---|---|
| How many tickets are currently open? | 111 |
| Which agent resolved the most tickets this month? | AGT-01, 16 (Mar 2024) |
| Show me all Critical tickets not resolved within 12 hours. | 34 |
| What is the average customer rating for Technical category tickets? | 3.74 (104 of 152 rated) |
| Are there any anomalies in resolution times this week? | 1 outlier, 41 in scope |
| Which agent has the lowest average customer rating? | AGT-08, 3.4800 |
| How many unresolved billing tickets are there? | 58 |

AGT-08 (3.4800) vs AGT-11 (3.4828) look tied at 2dp but are not. The executor
detects genuine ties and reports all tied groups.

## Known limitations (for the README's limitations section)

- The aging rule flags 80 of 80 unresolved High/Critical tickets because the data
  spans three months with no backfilled resolutions. Correct per the brief, noisy
  by nature. `AGING_HOURS` is configurable; the detector says so in its note.
- Query DSL has no nested boolean logic beyond `filters` AND `filters_any` OR. No
  arbitrary parenthesised expressions.
- No multi-hop or comparative questions ("compare Q1 vs Q2 for agents who...").
- Single-process, in-memory. Right for 500 rows; would need a real store beyond
  roughly a million.
- No conversation memory; each question is independent.
- Free-tier rate limit is 30 req/min. Two LLM calls per question (plan +
  narration); set `NARRATE_ANSWERS=false` to halve it.
- `issue_summary` is only matched with `contains`, no semantic search.
- Timestamps are naive; a multi-timezone deployment would need tz handling.

## Scaling notes (walkthrough question: "how would you scale this?")

Swap the pandas executor for the same plan compiled to parameterised SQL against
DuckDB or Postgres; the plan DSL is the stable interface, so nothing above it
changes. Cache plans by normalised question text. Move detectors to a scheduled
job writing to a flags table. Add per-user rate limiting in front of the LLM.
