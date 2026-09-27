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
| `inferconomy.providers` | `OpenAICompatibleClient`, the `Transport` seam, and the provider error taxonomy |
| `inferconomy.telemetry` | Token, cost, and latency accounting |
| `inferconomy.tokens` | Pluggable `TokenEstimator` for providers that do not report usage |
| `inferconomy.costs` | Versioned price table, loadable from bundled JSON or an operator's file |
| `inferconomy.benchmark` | Fixed-budget baseline runs, reproducible from a config file, plus the oracle bound |
| `inferconomy.judge` | `Judge` protocol, the null condition, and per-benchmark judge noise |
| `inferconomy.frontier` | Cost-quality curve assembly, and the gate that decides whether one may be published |

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

```python
report = await probe_capabilities(client)  # one tiny call, per model
options = report.adapt(CompletionOptions(effort="high", reasoning_budget=4096))
# -> effort and reasoning_budget are gone, each with a recorded reason
```

`inferconomy.capabilities` reports **three** outcomes per capability, not two:

| Outcome | Meaning |
|---|---|
| `supported` | Confirmed by a positive observation. |
| `unsupported` | Confirmed by a *definitive* answer from the endpoint. |
| `unknown` | Not established. The lever may well work. |

The third value is the point. A capability can usually be *confirmed* by a
positive observation but rarely *refuted* by a null one: a model asked a trivial
question may emit no reasoning trace on a model that reasons perfectly well.
Reading that as "unsupported" would attribute a fact about the model to a
limitation of our probe, then select a fallback as though it were established. An
unconfirmed lever is therefore **not pulled**, and the run records the
degradation and its reason, because a call whose effect cannot be attributed is
indistinguishable from one where the lever worked.

An adapter that cannot test a capability reports `unknown` rather than guessing,
and a claim the probe does not confirm is surfaced as a disagreement instead of
being quietly overwritten.

Planned adapters: OpenAI-compatible (which also covers OpenRouter, Groq, Together,
Fireworks, DeepSeek, Mistral, and local vLLM / Ollama servers), Anthropic, and
Google. The OpenAI-compatible one is done. Anthropic and Google are next, not done.

### Using an OpenAI-compatible endpoint

```python
from inferconomy.providers import OpenAICompatibleClient

api = OpenAICompatibleClient(
    base_url="https://api.deepseek.com",  # or api.openai.com/v1, or localhost:8000/v1
    api_key=os.environ["DEEPSEEK_API_KEY"],
)
```

The base URL you paste is the one from that provider's docs; the `/v1` and
`/chat/completions` parts are filled in for you, and a base that already has them
is left alone.

`httpx` is needed for the default transport and comes from the `openai` extra.
It is an extra rather than a dependency because **the library itself has none** -
the adapter accepts any object with a `post_json` method, so if you already have
an HTTP client with your proxy, CA bundle, and connection pool configured, pass
it in instead:

```python
OpenAICompatibleClient(base_url=..., transport=HttpxTransport(client=my_client))
```

Things worth knowing, because they are where "one adapter" does real work:

- **`max_tokens` or `max_completion_tokens` is configurable.** OpenAI rejects the
  first for its reasoning models; the local servers accept only the first. There
  is no value that works everywhere, so it is a setting
  (`max_tokens_field=`) rather than a guess.
- **There is no standard reasoning-token budget field** in this format, so
  Inferconomy does not invent one. It claims no such capability and omits the
  value unless you name the endpoint's dialect. A made-up field name would either
  be ignored - silent pretence - or draw a 400 on every call.
- **A lever you did not confirm is dropped, and one you set is sent.** Set
  `effort="high"` and this adapter sends `reasoning_effort`; on an endpoint
  without that field the 400 names it, which is better than ignoring the value
  and leaving you to believe the model was told to think harder. Probe first and
  the problem disappears:

  ```python
  report = await probe_capabilities(api)
  options = report.adapt(CompletionOptions(effort="high", reasoning_budget=4096))
  response = await api.complete(request, options.options)
  ```

- **Usage is the provider's, or an estimate, and always says which.** A response
  with a usage block gives `tokens_exact=True`; one without falls back to
  `estimate_usage`, which is `tokens_exact=False`. Cost is never invented, because
  cost needs a rate the endpoint did not send.
- **Failures are classified by what you can do about them** - `AuthenticationError`
  and `RequestRejectedError` are not retryable, `RateLimitError`, `ServerError` and
  `TransportError` are - and the provider's own message always survives on
  `error.provider_message`. `ContextLengthExceeded` is separated out because
  shrinking the request is a valid response to it.

**Known limitation, stated rather than hidden:** endpoints fold prompt cache hits
into `prompt_tokens` and `Usage` has nowhere to put a cached-token count, so a
flat price table *over-states* the cost of a cached prompt. It errs upward rather
than losing money, and fixing it properly means a cached-token field on `Usage`
and tiered rates in the price table.

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

`inferconomy.frontier` enforces all four as a publication gate rather than a
promise. `Frontier.publish()` raises `UnpublishableFrontier` unless every control
passes, naming each one that did not, and a frontier starts out `published=False`
so an assembled curve cannot be passed off as a vetted one. Two of the checks are
numeric because they can be faked by accident:

- **The cost-matched check compares token counts,** and fails if any point used
  more tokens than the reference arm. The reference arm defaults to the
  *largest* budget for exactly this reason: a saving has to be measured against
  the arm that was given the most compute, and on a budget sweep the cheapest arm
  is by definition the one holding the fewest tokens.
- **The judging check requires a passed null and a measured noise floor,** and
  fails if the judge was a proxy. It also refuses an oracle bound computed
  against a different baseline than the curve's, since dividing a saving by a
  bound taken from another reference arm is a ratio of two unrelated numbers.
- **A quality delta inside the noise floor is not a finding.** The point stays on
  the curve, because hiding it would misreport the shape, but it is marked
  `resolvable=False` and the frontier reports which points those were.

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
- [x] Fixed-budget baseline harness
- [x] Oracle budgeter as an upper bound
- [x] Null condition and calibrated judge harness

### Phase 1 — One working path

- [x] Capability probe
- [x] OpenAI-compatible provider adapter
- [ ] `DecisionEngine` and `Policy` protocols
- [ ] `direct` and `reason` strategies
- [ ] Sufficiency detection and escalation loop
- [ ] First published cost-quality frontier on one benchmark *(harness, publish
      gate, capability probe, and provider adapter all shipped; needs
      credentials and a real run)*

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

### 2026-09-26 — US-006, fixed-budget baseline runner

`inferconomy.benchmark` runs a benchmark at a fixed budget and emits the full
metric row. 287 tests, 100% coverage.

- **A run is reproducible from its config alone, and that is checkable.**
  `BaselineConfig` serialises to JSON and `fingerprint` hashes the canonical
  form; the fingerprint is recorded in the result. Reproducibility is a comparison
  of two hashes rather than a promise.
- **No result contains a wall-clock timestamp.** A field that changes on every run
  would make that comparison vacuous. Latency comes from the provider's own
  reported `latency_ms`.
- **The baseline is prevented from getting clever.** One call per task, no
  inspection of the output, no retry, no escalation. `escalated` is always False
  and `decision_overhead_ms` always 0.0 — recorded as constants precisely so an
  adaptive run has a zero point to be measured against.
- **Quality is `None`, not zero.** No judge exists yet, and a zero would read as a
  task that scored nothing rather than one that was never scored.
- **There is no grand mean.** `by_category()` is the only aggregation, because a
  single number across heterogeneous categories measures the category mix rather
  than the optimiser. This is per-domain reporting enforced by the API rather
  than promised in a footnote.
- **Mean cost is taken over priced rows only,** and the count of priced rows is
  reported beside it. Averaging over unpriced rows would report an unknown cost
  as a free one.
- **Provenance survives the run.** Each row carries `usage_basis` and `citable`,
  and the run records the price table version and whether any of its rates were
  verified — so "reported usage, unverified rate" stays visible as exactly that.
- **A provider error aborts the run.** A partial run that looks complete is worse
  than a crash, because a crash cannot be mistaken for a result.

---

### 2026-09-26 — US-007, oracle budgeter

`compute_oracle` turns a baseline plus every other budget into an upper bound on
what perfect per-task allocation would have saved. 328 tests, 100% coverage.

```python
from inferconomy.benchmark import compute_oracle

result = compute_oracle([baseline_run, low_budget_run, high_budget_run])
print(result.max_savings_fraction)  # the bound
print(result.attainable)  # always False
print(result.citable)  # False if any cost was estimated
```

- **The bound is per task, not a single budget.** For each task the oracle takes
  the cheapest cost it observed and reports the difference. This is the whole
  point: a real allocator would give easy tasks a small budget and hard ones a
  large one, and only a per-task comparison can bound that.
- **`attainable` is a property that always returns `False`,** not a stored field.
  Someone editing a results file to claim the bound was achieved still gets
  `False` back, so the label cannot be laundered through serialization.
- **The caveats travel inside the payload.** `ORACLE_CAVEATS` is serialised
  alongside the number, because a caveat living only in a docstring is a caveat
  nobody reads once the result has been passed around as JSON.
- **The dominant caveat is that quality is assumed, not measured.** No judge
  exists yet, so the oracle can say how much better allocation might have saved
  and cannot say whether the answer would still have been right. A real judge may
  find the cheapest run for a task is not the best one. The bound is only
  meaningful once that assumption is replaced.
- **The baseline is its own candidate,** so a task whose baseline was already its
  cheapest reports zero saving rather than a negative one. The bound cannot dip
  below zero, and `max_savings_fraction` is capped at 1.0.
- **Estimated costs produce a bound that is not citable.** The saving is still
  reported — it is the best estimate available — but `citable` is `False`, since a
  bound built on estimates is an estimate of an estimate.
- **Uncomparable tasks are excluded and counted,** never priced at zero. A bound
  computed while quietly dropping tasks is not a bound, so `incomparable_rows` is
  reported and blocks citability.
- **Runs must share tasks, strategy, and price table,** or the comparison is
  refused. Two price tables would make the difference measure the tables, and
  comparing two strategies reports the strategy gap as though it were a budget
  gap.

---

### 2026-09-27 — US-011, OpenAI-compatible provider adapter

`inferconomy.providers.OpenAICompatibleClient` speaks the one wire format that
OpenAI, OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, vLLM, and Ollama
all adopted, and absorbs the dialects underneath it. 702 tests, 100% coverage.

```python
api = OpenAICompatibleClient(base_url="https://api.deepseek.com", api_key=...)
report = await probe_capabilities(api)  # US-010
options = report.adapt(CompletionOptions(effort="high"))  # drops it if unconfirmed
response = await api.complete(request, options.options)
```

- **The `Transport` seam is what keeps the library dependency-free.** The stdlib
  has no async HTTP, and there are no runtime dependencies, so the HTTP client is
  either optional or injected. It is injected: `Transport` is a `post_json` method,
  the default wraps `httpx` behind an extra, and a caller with their own client
  passes it in. This is also why the suite can prove normalization against
  recorded payloads offline, which the repository's network guard requires anyway.
  A test now asserts the *installed distribution metadata* carries no runtime
  dependency, so the commitment is checked rather than repeated in a comment.
- **The interesting work is the disagreements, not the envelope.** These endpoints
  agree on chat completions and differ on nearly everything inside it:
  `max_tokens` versus `max_completion_tokens`, `reasoning` versus
  `reasoning_content`, usage blocks that some gateways strip entirely. All of it
  is settled inside the adapter, so a caller reads one `Response` and never learns
  which provider they are on.
- **There is no standard field for a separate reasoning-token budget** in this
  format, and that is a finding rather than a gap: OpenAI folds reasoning into
  `max_completion_tokens`, OpenRouter wants a nested object, DeepSeek has neither.
  So the adapter refuses to invent a field name, claims no such capability, and
  omits the value unless a dialect is named. A guessed field would be ignored -
  which is the silent pretence this library exists to prevent - or draw a 400 on
  every call.
- **It claims only what the format guarantees.** Usage reporting is standardized;
  reasoning content and an effort knob are vendor extensions, so claiming them
  would be an overclaim that US-010's probe then flags on every Mistral request.
  Declared capabilities default to `USAGE_REPORTING` alone, and the runtime probe
  supplies the rest.
- **A lever you set is sent, and refused loudly if the endpoint lacks it.** Set
  `effort="high"` on Mistral and you get a 400 naming `reasoning_effort`, not a
  silent success. The tempting alternative - drop the value and carry on - leaves
  a caller believing the model was told to think harder when it was not. The
  supported order is probe, adapt, send, and there is a test that runs all three.
- **Usage is the provider's or an estimate, and never better than it is.** A
  response with a usage block gives `tokens_exact=True`; without one,
  `estimate_usage` gives `tokens_exact=False` and cost stays `None`, because cost
  needs a rate the endpoint did not send. Two calls that differ only in whether
  the provider reported usage land on opposite sides of the provenance boundary,
  which is a test.
- **Errors are classified by what a caller can do about them.** `retryable` is the
  field to branch on, and it defaults to `False`, because retrying an
  unclassifiable error forever is how a bug becomes an outage. A rejected key is
  permanent; a 429, a 5xx, and a reset connection are not. The provider's own
  message always survives on `error.provider_message`, including the nested,
  bare-string, and top-level spellings these endpoints use, and response bodies are
  clipped before they reach an exception because error text ends up in logs.
  `ContextLengthExceeded` is split out because shrinking the request is a valid
  response to it - at the cost of classifying on message text, which is a guess
  and labelled as one.
- **A success that is not a success is a failure.** A 200 carrying an HTML error
  page, a 200 with no choices, a choice with no message - all raise, because
  returning a blank answer for them would let a misconfigured gateway look like a
  model with nothing to say. A null `content` is *not* one of those: reasoning
  models emit it legitimately, and that is an empty answer.
- **Base URLs normalize from what the docs actually say.** `https://api.deepseek.com`
  becomes `.../v1/chat/completions`; a base already ending in the full path is left
  alone; a local server with no version segment gets one, because Ollama's
  compatible route lives under `/v1` and the bare path is a 404 that looks like a
  broken server. All eight are covered by table tests.

**Known limitation, stated rather than hidden:** cache-hit prompt tokens are folded
into `prompt_tokens` and `Usage` has no field for them, so a flat price table
over-states the cost of a cached prompt. It errs upward instead of losing the
difference, and doing it properly means a cached-token field on `Usage` plus tiered
rates in the price table - a change to those contracts, not something to smuggle
into an adapter.

### 2026-09-27 — US-010, capability probe

`inferconomy.capabilities` finds out what a target model actually does instead of
what an adapter hopes it does. 554 tests, 100% coverage.

```python
report = await probe_capabilities(client)  # one tiny call, ~16 output tokens
assert report.supports(Capability.LOGPROBS)  # False unless actually confirmed

adapted = report.adapt(CompletionOptions(effort="high", reasoning_budget=4096))
for drop in adapted.degradations:
    print(drop.capability.value, "->", drop.fallback)
```

- **The outcome is three-valued, and that is the whole design.** A capability can
  usually be *confirmed* by a positive observation and rarely *refuted* by a null
  one. A model asked `17 * 23` may emit no reasoning trace on a model that
  reasons perfectly well, so a missing trace is silence, not refusal. Reporting
  `unsupported` there would attribute a fact about the model to a limitation of
  our probe, and then pick a fallback as though it had been established. So
  `unknown` exists, and it is never quietly downgraded.
- **Unconfirmed means not pulled.** `adapt()` strips a lever unless it is
  confirmed, and records what was dropped and why. Sending an unverified lever
  produces a call whose effect cannot be attributed, and the caller cannot tell
  that from a lever that worked, so the conservative reading is the only one that
  keeps a run record truthful. Unknown is treated the same as unsupported here,
  deliberately.
- **Every capability has a documented fallback as data, not prose.** The table
  lives in `FALLBACKS` so a caller can print the reason next to a result, and a
  test fails if a capability is added without one. A degradation nobody can
  explain is indistinguishable from a bug.
- **Absence and declaration are not the same kind of evidence.** `tokens_exact =
  False` is the adapter *declaring* its counts are estimates, which settles the
  question and is reported `unsupported`; a null `reasoning_text` is silence and
  is reported `unknown`. Both look like "nothing in the response", and collapsing
  them would either overstate a known-degraded cost figure or invent a fact about
  reasoning.
- **The three capabilities a response cannot reveal are delegated, not guessed.**
  Reasoning budget, effort, and logprobs look identical in a successful response
  whether or not the lever did anything, so they need the adapter to test the
  endpoint, through an optional `CapabilityProbeProvider`. That method returns
  `True`/`False`/`None`, where `None` means "no definitive answer". Adapters
  without it get `unknown`, the honest outcome: the lever was not tested, not
  shown to be missing. The extension is separate from the `Client` protocol on
  purpose, so an adapter that cannot answer stays a usable client instead of
  carrying a method whose honest answer is "I do not know".
- **Only overclaims count as disagreements.** An adapter claiming a lever the
  probe could not confirm is a bug, and the dangerous direction, since a policy
  reading the declaration would pull a dead lever. The reverse, a lever the model
  has and the adapter never mentioned, is a discovery rather than a conflict, and
  is on `report.discovered` instead. My first cut reported both directions and
  produced a disagreement for *every* confirmed capability on any adapter with a
  short declaration, which buried the one signal worth having.
- **Probing costs money and says so.** `report.calls` is in the payload, the
  prompt is one a reasoning model will actually reason about, and `cache_key`
  exists so a report gets reused instead of re-probed per request. A failed probe
  refuses to claim the model that was asked for, since no call reached the
  endpoint to confirm it.
- **A probe never raises.** Its job is to describe a degraded environment, so a
  provider failure becomes an `unknown` with the error text attached. Probing at
  startup should not be the thing that stops the process. This is load bearing: a
  malformed test double here surfaced as a tidy report of "nothing supported
  anywhere" with the `TypeError` sitting in the detail, which is precisely how a
  real adapter bug would look.

### 2026-09-26 — US-008, null condition and calibrated judge

`inferconomy.judge` defines the `Judge` protocol, proves the evaluation harness is
sound, and measures how much a judge wobbles. 395 tests, 100% coverage.

```python
from inferconomy.judge import JudgeRequest, judge_noise, null_condition

items = [JudgeRequest(task_id="t1", category="code", prompt="...", response="...")]

null = await null_condition(my_judge, items)
assert null.passed  # the provable zero held
noise = await judge_noise(my_judge, items, repeats=3)
print(noise.min_detectable_delta)  # smallest delta worth claiming
```

- **A null is only provable when both arms are the same object.** The first check
  compares a score set against itself, which is zero by arithmetic identity no
  matter how noisy the judge is. A nonzero result can only mean this library is
  miscounting, so it is a hard assertion. Anything else is only *empirically*
  zero, and the size of its failure is noise rather than a defect — so it is
  reported, not asserted.
- **The second check asks whether the judge depends on batch position.** Judging
  the same items in reverse and comparing is not trivially zero, so it measures
  something real: a model judge whose verdict moves with where an item sat is
  broken in a way a single pass would never reveal. It is asserted only when the
  judge declares itself deterministic, and reported either way.
- **`min_detectable_delta` is a maximum across categories, not a mean,** so a
  quiet benchmark cannot hide a noisy one. A frontier reporting a 0.4% quality
  gain against a 1.2% floor has reported the judge, not the system.
- **Three different noise measures, because they fail differently.**
  `mean_stdev` is the typical wobble, `max_range` is the worst single item that a
  mean would hide, and `unanimous_fraction` is the difference between an
  instrument and a coin.
- **Scores outside `[0, 1]`, `NaN`, and infinities are rejected at the boundary**
  rather than averaged. A judge returning `1.5` is broken, and averaging hides
  that instead of surfacing it.
- **A `PROXY` basis makes the report `citable=False`.** Output length is
  correlated with quality rather than measuring it, and a pipeline mixing a model
  judge with a length proxy has not measured quality at all.
- **Aggregation is canonical, so input order cannot move the answer.** Summation
  order changes the last bits of a float mean; a harness whose output depends on
  arrival order would be reporting its own noise.
- **Judging runs sequentially on purpose,** since the order check assumes the only
  thing varying between passes is the order.
- **Deltas are `later - earlier`,** and the direction is stated rather than
  implied. For cost the good direction is negative and for quality it is positive,
  and a metric whose good direction depends on the quantity is one that will be
  quoted backwards.

---

### 2026-09-26 — US-009, frontier assembly and the publication gate *(partial)*

`inferconomy.frontier` builds a cost-quality curve from measured runs and judged
quality, and refuses to publish one that is missing evidence. 466 tests, 100%
coverage. **The curve itself is not published yet — see below.**

```python
from inferconomy.frontier import FrontierConfig, build_frontier

config = FrontierConfig.from_file("benchmarks/frontier.json")
frontier = build_frontier(config, runs, quality, oracle=oracle, null=null, noise=noise)
payload = frontier.publish()  # raises UnpublishableFrontier if a control failed
```

- **`publish()` raises rather than warning.** A frontier missing its oracle, its
  null condition, or a cost-matched baseline is not a weak result, it is an
  unusable one, so the error names every failing control and tells the caller
  which check to go and satisfy.
- **A frontier starts `published=False`**, and only `publish()` sets it, so an
  assembled curve cannot be mistaken for a vetted one by inspecting the object.
- **The reference arm defaults to the largest budget.** This was a bug the tests
  found. Savings have to be measured against the arm given the most compute; on a
  budget sweep the cheapest arm holds the fewest tokens by construction, so
  defaulting to the smallest made this project's own cost-matched-baseline
  control impossible to satisfy.
- **An oracle bound from a different baseline is rejected.** The original code
  divided a saving measured against one arm by a bound computed from another and
  called the result a captured fraction. That is a ratio of two unrelated numbers,
  so a mismatch now fails the oracle control and yields no fraction at all.
- **The token comparison is numeric, not a claim.** A point that used more tokens
  than the reference arm fails the control, with a 5% tolerance for the difference
  between two runs of the same budget.
- **A delta inside the noise floor is reported, not claimed.** Such a point keeps
  its place on the curve — removing it would misreport the shape — but is marked
  `resolvable=False`, and `Frontier.unresolvable()` lists them.
- **The controls travel in the payload,** with a reason string each, so a reader
  of a results file meets the objections alongside the curve.

**What is not done, and why.** The published curve needs real data, and there is
none: an adapter now exists (US-011) but there are no credentials to call one
with. A frontier built on the shipped fake would be a real cost-quality curve of a
test double, and publishing it as evidence would be precisely the failure this
project exists to prevent. The committed `benchmarks/frontier.json` is a template
with placeholder model and judge, fingerprinted so the reproducibility claim
becomes a comparison of two hashes once a real run exists. The backlog records
this story as `blocked` rather than done.

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
