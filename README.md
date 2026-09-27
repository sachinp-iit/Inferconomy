# Inferconomy

> **Optimize inference. Maximize token economy.**

Inferconomy is an LLM inference optimization library. It decides **how** a model
should infer and **how much** computation each request deserves, instead of
applying one fixed strategy and one fixed budget to every request.

It is a **library**, not a framework. It sits underneath whatever you already
use — an agent, a router, an IDE, a coding assistant, a plain HTTP service — and
wraps the LLM client you already have.

```
pip install inferconomy        # status: not yet released
```

The package builds and installs today (`US-001` is done) but has no public
release. What exists so far is the data contracts, the client boundary, token and
price accounting, and deterministic test doubles — see the
[progress log](#progress-log) for exactly where it stands.

Work proceeds one user story at a time, tracked in [BACKLOG.md](BACKLOG.md) and
recorded in the [progress log](#progress-log).

---

## Status

🚧 **Pre-alpha.** No release yet. No code is published at this time — the
architecture below is the design we are building to, and the [roadmap](#roadmap)
and [progress log](#progress-log) track what actually exists.

We are publishing progress rather than staying quiet, so that the design can be
reviewed before it hardens. Nothing below the roadmap is implemented.

---

## Why

LLM applications run on fixed inference behaviour:

- fixed reasoning depth
- fixed output budgets
- fixed prompting strategies
- fixed verification patterns

But the computation a request needs varies by orders of magnitude. A factual
lookup and a hard debugging session do not deserve the same budget. Allocating a
uniform maximum is wasteful; allocating a uniform minimum is wrong.

Inferconomy makes that allocation automatic, and — critically — it starts small
and escalates rather than truncating:

```
 Every request              Request
      ↓                         ↓
 Same strategy            Analyze task
      ↓                         ↓
 Same budget              Choose strategy
      ↓                         ↓
                    Allocate initial budget
                              ↓
                        LLM inference
                              ↓
                        Is it sufficient?
                         ↙          ↘
                       Yes           No
                        ↓             ↓
                     Return     Escalate / switch strategy
                                     ↓
                              Continue inference
```

---

## Core idea

Inference is an adaptive resource-allocation problem. For a task `q`, find the
inference procedure `I` that is cheap but sufficient:

```
minimize    Cost(I, q)

subject to  Quality(I, q) >= target
```

The objective is **not** "generate fewer tokens." It is **minimum sufficient
computation**. That computation spans strategy, reasoning depth, reasoning
budget, output budget, verification, refinement, escalation, and early
termination.

A budget cut that degrades task quality is not a successful optimization. Token
savings are only meaningful *while task success is preserved*, and that is the
number we report.

---

## How it works

```
┌──────────────────────────────────────────────────────────┐
│                    Host application                       │
│      agent · router · IDE · assistant · custom service   │
└───────────────────────────┬──────────────────────────────┘
                            │  Request  +  LLM client
                            ▼
┌──────────────────────────────────────────────────────────┐
│                      Inferconomy                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Capability probe — what can this model actually    │  │
│  │  expose? reasoning tokens · effort control ·       │  │
│  │  logprobs · separable chain-of-thought              │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Decision engine (pluggable)                        │  │
│  │  is reasoning needed · task shape · uncertainty    │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Strategy planner                                   │  │
│  │  direct · reason · decompose · reason+verify · …    │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Policy + budget controller (pluggable)             │  │
│  │  initial budget · escalation · early termination    │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Runtime loop — generate · inspect · continue ·     │  │
│  │  stop · adapt                                        │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Telemetry — tokens · cost · latency · decisions    │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           ▼
                    Any LLM provider
```

### Module map

| Module | Responsibility |
|---|---|
| `inferconomy.api` | `optimize()` — the single public entry point |
| `inferconomy.contracts` | `Request`, `Response`, `Usage`, `OptimizationReport`, and the enums |
| `inferconomy.client` | `Client` protocol and `CompletionOptions` — the only outside-world dependency |
| `inferconomy.testing` | `FakeClient` and helpers, shipped so users can test without spending tokens |
| `inferconomy.capabilities` | Probes what the target model supports |
| `inferconomy.decision` | `DecisionEngine` protocol + adapters |
| `inferconomy.strategy` | Strategy catalogue and selection |
| `inferconomy.policy` | `Policy` protocol — budget allocation, escalation, stopping |
| `inferconomy.runtime` | The generate/inspect/continue/stop loop |
| `inferconomy.providers` | Provider clients and usage extraction |
| `inferconomy.telemetry` | Token, cost, and latency accounting |
| `inferconomy.tokens` | Pluggable `TokenEstimator` for providers that do not report usage |
| `inferconomy.costs` | Versioned price table, loadable from bundled JSON or an operator's file |

### Design principles

1. **Quality before savings.** Never trade meaningful task quality for tokens.
2. **Minimum sufficient computation.** Not every task deserves the same budget.
3. **Escalate, never truncate.** A budget is a starting point, not a cap.
4. **Model-agnostic interface, not uniform efficiency.** See below.
5. **Framework-agnostic.** We wrap a client; we do not own your architecture.
6. **Every decision is measurable.** Quality, tokens, latency, cost — or it did
   not happen.
7. **Policy is pluggable.** Allocation is a swappable component, not a hardcoded
   heuristic.

---

## Model support and capability probing

We are model-agnostic in the sense that matters: **one interface, any provider.**

We are *not* claiming uniform efficiency across models, because that would be
false. Adaptive control needs levers, and models expose different ones:

| Capability | What it enables | Fallback if absent |
|---|---|---|
| Separate reasoning-token budget | Reasoning/output budget split | Single output budget |
| Reasoning-effort control | Direct mode switching | Prompt-level instruction |
| Chain-of-thought visible to caller | Checkpointed sufficiency checks | End-of-response check only |
| Logprobs / confidence | Probabilistic early exit | Deterministic heuristics |
| Token usage in response | Exact cost accounting | Estimated accounting, flagged as such |

Inferconomy probes for these at runtime, uses what is available, and **degrades
to a documented baseline otherwise**. Results that depend on an estimated rather
than exact cost are marked as such in the report. A model with no reasoning
control and no inspectable trace offers a policy very little to work with, and we
would rather say so in the report than imply a saving we did not achieve.

Planned adapters: OpenAI-compatible (which also covers OpenRouter, Groq, Together,
Fireworks, DeepSeek, Mistral, and local vLLM / Ollama servers), Anthropic, and
Google. Anthropic and Google are next, not done.

---

## Example API

> Provisional. Expected to change during Phase 1.

```python
from inferconomy import optimize

result = optimize(
    client=llm,
    request="Explain eventual consistency with an example.",
)

print(result.text)
print(result.usage)  # exact token + cost accounting
print(result.report)  # why this strategy, this budget, this stop
```

A report, roughly:

```python
result.report = OptimizationReport(
    strategy="direct",
    capabilities_used=(Capability.USAGE_REPORTING,),
    initial_budget=180,
    additional_budget=0,
    stopped_on=StopReason.SUFFICIENCY,
    decisions=("classified as direct-response task", "allocated 180 tokens"),
)

result.report.total_budget  # 180  (derived, cannot contradict the fields)
result.report.escalated  # False
result.usage.cost_exact  # whether cost is measured or estimated
```

The report is a typed object, not a dict, and it is intended to be
machine-readable. Every optimization should be auditable after the fact, and
`policy=<your policy>` is the seam for supplying your own allocation behaviour.

Two properties of these types are worth knowing before you rely on them:

- **One request is not one call.** Adaptive allocation means several LLM calls
  can serve a single user request, so `Usage` aggregates across calls and counts
  them in `llm_calls`. Code that assumes one call per request will compute cost
  wrongly.
- **Not every number is a measurement.** `Usage` carries `cost_exact`, which is
  `False` by default. A library should under-claim precision rather than let you
  read an estimate as a fact.

---

## Evaluation methodology

We report against fixed-budget and non-adaptive baselines. The comparison that
matters is not *tokens saved* but **cost saved while task success is preserved**.

| Metric | Purpose |
|---|---|
| Task success | Does optimization preserve quality? |
| Quality delta vs. baseline | Quality change, not just a success rate |
| Output / reasoning / total tokens | Where the compute went |
| Cost | Actual monetary effect |
| Latency | Runtime impact, including our own overhead |
| Decision overhead | What the optimization layer itself cost |
| Escalation rate | How often the initial budget was insufficient |
| Failure rate | How often optimization harmed the result |
| Cost-quality frontier | Efficiency across quality levels |

### Four controls we hold ourselves to

Self-reported efficiency numbers are easy to manufacture. Four controls keep ours
honest, and we consider a result without them unusable:

1. **Oracle budgeter.** How much could *perfect* allocation have saved? Without
   this upper bound you cannot tell a good result from a mediocre one.
2. **Cost-matched baseline.** The fixed-budget baseline is allowed the *same*
   token count. Otherwise any saving is a strawman.
3. **Null condition and calibrated judging.** Judge noise is measured and
   subtracted; a null configuration must measure as a difference of zero.
4. **Per-domain reporting.** Gains on mathematical and code reasoning do not
   transfer to summarization, extraction, or open-ended explanation. Results are
   reported per category, never aggregated into a single favourable number.

Benchmark coverage spans factual QA, extraction, classification, translation,
summarization, explanation, comparison, mathematical reasoning, coding,
debugging, planning, analysis, and structured generation.

---

## Roadmap

Sequenced so that each phase is defensible before the next is attempted. The
harness comes first, on purpose: a measurement is worth more than a controller
built on unmeasured assumptions.

### Phase 0 — Measurement foundation

- [x] `pyproject.toml`, package skeleton, installable and importable
- [x] Core request / response contracts, typed and serializable
- [x] `Client` protocol and deterministic fake, suite runs fully offline
- [x] Token and cost telemetry with exact-vs-estimated accounting
- [ ] Fixed-budget baseline harness
- [ ] Oracle budgeter as an upper bound
- [ ] Null condition and calibrated judge harness

### Phase 1 — One working path

- [ ] Capability probe
- [ ] OpenAI-compatible provider adapter
- [ ] `DecisionEngine` and `Policy` protocols
- [ ] `direct` and `reason` strategies
- [ ] Sufficiency detection and escalation loop
- [ ] First published cost-quality frontier on one benchmark

### Phase 2 — Adaptation and breadth

- [ ] `decompose`, `reason+verify`, `reason+refine`
- [ ] Strategy switching mid-request
- [ ] Decision-engine adapters beyond the built-in heuristic
- [ ] Anthropic and Google adapters
- [ ] Streaming support

### Phase 3 — Optimization

- [ ] Budget calibration from observed outcomes
- [ ] Quality-aware allocation
- [ ] Benchmark suite across the full category list
- [ ] Runtime metrics and regression gates in CI

### Phase 4 — Research

- [ ] Dynamic strategy composition
- [ ] Inference-state-aware continuation instead of restart
- [ ] Value-of-computation estimation
- [ ] Learned allocation policies

Phases 3 and 4 are research, not commitments. We would rather publish one
measured result than four aspirations. Notably, a *negative* result — that
strategy composition does not beat the best single strategy — is a valid and
useful outcome, and we will publish it as one.

---

## Progress log

We publish progress here rather than only in release notes, so the trajectory is
visible while the design is still open to review. Convention:

- Newest entry first, `YYYY-MM-DD` heading.
- Added in the same pull request as the change it describes.
- States what works, not what is planned. Plans belong in the roadmap.
- Honest about regressions. A number that got worse goes in the log, not hidden.

### 2026-09-26

Project created. README restructured to separate the design from the plan.
Public architecture, design principles, evaluation methodology, and a
measurement-first roadmap are now defined. No implementation yet.

### 2026-09-26 — US-001, installable package skeleton

Inferconomy is now a real package rather than a directory of notes.

- `pyproject.toml` with hatchling, src layout, dynamic versioning
- **Zero runtime dependencies**, and that is a design commitment rather than an
  accident: a library that wraps a client you already have should not force a
  provider SDK or a framework on you
- `import inferconomy` and `__version__` work from a clean environment
- Tests assert the installed distribution version matches `__init__`, so the
  dynamic version cannot silently drift
- CI on Python 3.10–3.13: lint, format, strict `mypy`, tests with coverage, plus
  a build-and-install check and a README link check
- `Typing :: Typed` is declared and the `py.typed` marker actually ships — a
  test enforces the second half

`optimize()` is not exported yet. It will be when it exists.

### 2026-09-26 — US-002, core request and response contracts

`Request`, `Response`, `Usage`, and `OptimizationReport` now exist as frozen,
validated, serializable types. 73 tests, 100% statement coverage of the module.

Decisions worth recording, because they are not obvious from the signatures:

- **`inferconomy.contracts`, not `inferconomy.types`.** Avoids shadowing the
  standard library `types` module. The module map above was updated to match.
- **One request is not one call.** `Usage` aggregates across every LLM call and
  counts them in `llm_calls`. Anyone computing cost per request needs this.
- **`Usage.cost_exact` defaults to `False`.** A library should under-claim
  precision rather than let a caller read an estimate as a measurement. Summing
  usage is conjunctive on exactness for the same reason: one estimated component
  makes the total an estimate.
- **`total_budget` and `escalated` are derived, not stored.** A report cannot
  claim a total that contradicts its own fields, because the inconsistency is
  not representable.
- **Our `StopReason` is separate from the provider's `finish_reason`.** One says
  whether the optimizer was satisfied; the other says what the model did. Both
  are reported, neither is normalized into the other.
- **Subclass invariants are enforced.** `reasoning_tokens` cannot exceed
  `output_tokens`, `cached_input_tokens` cannot exceed `input_tokens`, and
  `cost_exact=True` requires a cost. These are the mistakes that produce
  confidently wrong savings claims.
- **Serialization is versioned and strict.** Reports carry `schema_version`, and
  a newer one is rejected rather than silently loaded as this version. Unknown
  enum values raise with the valid options listed, rather than being dropped.
- **Sequence fields are normalized to tuples** on construction, so a caller
  passing a list cannot mutate a frozen object after the fact.

`Request` deliberately carries no sampling parameters and no quality target. Those
belong to the strategy and the policy; accepting them here would put the
optimization decision back in the caller's hands, which is the thing this project
exists to remove.

### 2026-09-26 — US-003, client protocol and deterministic fake

`inferconomy.client` defines the whole of the library's dependency on the outside
world. A provider adapter implements `Client`; nothing else in the library knows
which provider or SDK is underneath. 111 tests, 100% coverage.

- **The protocol has two members** — `capabilities` and `complete` — and a test
  enforces that. A protocol that grows to accommodate a provider's full feature
  set stops being honest about what it supports.
- **`CompletionOptions` is separate from `Request`.** The request says *what* to
  solve; the options say *how* to call the API. Widening `Request` to carry the
  allocated budget would collapse the distinction that the budget controller
  exists to exploit.
- **The protocol is async.** Adaptive allocation makes a sequence of *dependent*
  calls — classify, generate, judge sufficiency, escalate — and every major
  provider SDK is async-first. A sync core would cost a thread per call or a
  nested event loop, and latency is one of the metrics we report.
- **Adapters ignore options they cannot honour** rather than raising, which is
  what lets one adapter span endpoints with different feature sets. Genuine
  failures still raise, so a real error is never reported as a successful
  truncated answer.
- **`FakeClient` ships inside the package**, not just in our test suite. Anyone
  integrating Inferconomy needs to test their own code without spending tokens,
  and that need does not stop at our repository.
- **The fake raises when its script runs out** rather than repeating the last
  response. Silently repeating hides a test that made more calls than it
  intended, which in a library whose purpose is counting calls is exactly the bug
  worth catching. `repeat_last=True` opts in.
- **The suite is offline by construction.** An autouse fixture in `conftest.py`
  fails any attempt to reach a non-loopback host, with tests asserting that the
  guard both blocks external hosts and permits loopback — a guard nobody
  verifies is not a guard.

### 2026-09-26 — US-004, token accounting with explicit provenance

`inferconomy.tokens` estimates what a provider declined to report, and every
figure it produces is labelled as an estimate. 150 tests, 100% coverage.

- **Token and cost provenance are separate flags** — `Usage.tokens_exact` and
  `Usage.cost_exact`. They fail independently: a provider can report exact token
  counts while the price of its cheapest tier is unknown, which is real data that
  one combined flag would force us to describe as wholly estimated.
- **Both default to `False`.** A library should under-claim precision rather than
  let a caller read an estimate as a measurement. A missing number is honest; a
  wrong one is not.
- **An estimator can never confer exactness.** `estimate_usage` hard-codes
  `tokens_exact=False`, so no future tokenizer or adapter can quietly upgrade an
  estimate into a measurement. `TokenEstimator` is a protocol precisely so that
  a real tokenizer can improve accuracy without touching that guarantee.
- **`Usage.exact` and `Usage.basis` are derived, never stored.** A record cannot
  claim a provenance its numbers do not support, and `OptimizationReport`
  exposes `usage_basis` and `citable` from the same source.
- **Aggregation degrades exactness.** `Usage.__add__` conjoins the flags: a sum
  containing one estimated component is an estimate. Escalation means several
  calls, and a total that hid one estimated call would be the most flattering
  possible way to be wrong.
- **The fake client now delegates to the shipped estimator.** It used to carry a
  private copy, which is a divergence waiting to happen.

### 2026-09-26 — US-005, versioned cost table

`inferconomy.costs` loads provider pricing from JSON that versions itself, ships
inside the wheel, and can be replaced by a file you supply. 215 tests, 100%
coverage.

- **Prices live in data, not code.** A vendor changing a price should not require
  a release from us. The file carries its own `version` and `schema_version`,
  independent of the package version, so a correction can ship without touching
  code.
- **Every price carries its provenance** — `source`, `as_of`, and `verified` — and
  a test asserts each one is present. A number nobody can trace is not a data
  point, it is folklore.
- **The bundled snapshot claims no verification.** Inferconomy cannot read your
  invoice, so every bundled price ships with `verified: false` and
  `price_usage` will not report a cost as exact on the strength of it. Real token
  counts priced at an unverified rate are reported as `priced_unverified` — a
  number is produced, and it is never citable. Supply a table you have checked
  and the same tokens become citable.
- **A user file replaces the bundled table entirely; the two are never merged.**
  Merging would silently resurrect bundled prices for models an operator meant
  to exclude, which is how a cost model acquires prices nobody chose.
- **An unknown model raises.** Returning zero would turn "we do not know what
  this costs" into "this is free", which is the most expensive kind of wrong. The
  error names the table, its version, and what it does know.
- **Cache reads and reasoning tokens are re-priced, not added.** Both are subsets
  of their parent count, so a million cached input tokens is a million input
  tokens, not two million.
- **Money is rounded to twelve places.** Unrounded per-million division
  accumulates float error in the low digits, and two runs that should agree stop
  agreeing.
- **Pricing never compounds a previous estimate.** A record that already carries a
  cost is priced from its token counts alone.

---

## Documentation

Full documentation will live in the [project wiki](https://github.com/sachinp-iit/Inferconomy/wiki):
core concepts, getting started, providers, models and capabilities, strategies,
policies and budgets, custom decision engines, telemetry, evaluation
methodology, benchmarks, and architecture.

This README announces and orients. The wiki explains. Sections are not
duplicated between them.

---

## What Inferconomy is not

Not an agent framework, a RAG framework, a tool orchestration framework, a
memory framework, a workflow engine, or a model router. Those can use
Inferconomy as an inference optimization layer.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Larger architectural or strategy changes
should open an issue first so the trade-offs can be discussed. Changes that
affect policy behaviour are expected to come with evaluation results.

## Security

See [SECURITY.md](SECURITY.md). Please report vulnerabilities privately rather
than through public issues.

## License

MIT. See [LICENSE](LICENSE).
