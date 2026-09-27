# Backlog

Work proceeds **one user story at a time**. Each story is implemented, tested,
committed, and pushed as a single change. The README [progress log](README.md#progress-log)
is updated in the same commit as the story it describes.

Stories are ordered so that each one produces something observable and testable,
and so that measurement exists before it is used to justify optimization.

Status legend: `todo` · `in progress` · `done`

---

## Phase 0 — Measurement foundation

> Nothing here optimizes anything. This phase exists so that later claims can be
> checked rather than believed.

### US-001 · Installable package skeleton `done`

As a developer, I want to install the package and import it, so that
Inferconomy is a real library rather than a directory of notes.

**Done when** `pip install -e .` succeeds, `import inferconomy` works from a clean
environment, the package version is importable, and CI runs on push.

### US-002 · Core request and response types `done`

As a developer, I want typed `Request`, `Response`, `Usage`, and
`OptimizationReport` objects, so that results are self-describing and not
stringly-typed dictionaries.

**Done when** all four types exist, are immutable where appropriate, and are
covered by round-trip serialization tests.

### US-003 · Client protocol and fake client `done`

As a developer, I want to inject any LLM client behind a narrow protocol, so
that the library never depends on a specific provider SDK.

**Done when** a `Client` protocol exists, a deterministic fake client is
included, and every test in the suite runs with no network access.

### US-004 · Token accounting, exact versus estimated `done`

As an operator, I want to know whether reported token counts are exact or
estimated, so that I never act on a cost number of unknown provenance.

**Done when** usage carries an explicit `exact: bool`, the estimate path is
covered by tests, and estimated results are flagged in the report.

Delivered as `Usage.tokens_exact` and `Usage.cost_exact` tracked separately, with
`Usage.exact` and `Usage.basis` as derived summaries, a pluggable
`TokenEstimator` in `inferconomy.tokens` whose estimates can never be marked
exact, and an optional `OptimizationReport.usage` field exposing
`usage_basis` and `citable`.

### US-005 · Versioned cost table `done`

As an operator, I want provider pricing kept in a separate, versioned data file
that I can override, so that price changes do not require a code release.

**Done when** costs load from a bundled JSON file, can be replaced by a user file,
and are versioned independently of the package.

Delivered as `inferconomy.costs` with a bundled `data/prices.json` that ships in
the wheel, a table `version` and `schema_version` independent of the package, and
`load_price_table(path)` for a full override. Every price carries `source`,
`as_of`, and `verified`; no bundled price is verified, so no cost derived from
the snapshot is ever reported as exact.

### US-006 · Fixed-budget baseline runner `done`

As a researcher, I want to run a benchmark with a fixed strategy and budget, so
that I have a reference point to compare against.

**Done when** a baseline run is reproducible from a config alone and emits the
full metric row.

Delivered as `inferconomy.benchmark`: `BaselineConfig` serialises to JSON and
carries a `fingerprint` hash of its canonical form, `run_baseline` emits a
`MetricRow` per task and repetition, and `RunResult.by_category` is the only
aggregation offered. Rows carry `usage_basis`, `citable`, zero decision overhead,
`escalated=False`, and `quality=None` until a judge exists.

### US-007 · Oracle budgeter `done`

As a researcher, I want to know how much *perfect* allocation would have saved,
so that I can tell a good result from a mediocre one.

**Done when** the oracle reports an upper bound on achievable savings and is
clearly labelled as unattainable in practice.

> Shipped as `inferconomy.benchmark.compute_oracle`. Takes the baseline plus every
> other budget, picks the cheapest cost per task, and reports the difference as
> `max_savings_usd`/`max_savings_fraction`. Three things make it un-misreadable:
> `attainable` is a property that always returns False, so a deserialized result
> cannot arrive claiming otherwise; `ORACLE_CAVEATS` travels inside the serialized
> payload; and a bound built on estimated costs reports `citable=False`. The
> baseline is its own candidate, so savings are never negative. Runs must share
> tasks, strategy, and price table, or the comparison is refused.

### US-008 · Null condition and calibrated judge `done`

As a researcher, I want judge noise measured and a provably-null configuration
included, so that reported quality deltas are not artifacts of judging.

**Done when** the null configuration measures as a difference of zero and judge
noise is quantified per benchmark.

> Shipped as `inferconomy.judge`. `null_condition` separates two checks that are
> usually conflated: a score set compared against *itself*, which is zero by
> arithmetic and so can only mean the harness is broken, and the same items judged
> in reverse order, which is not guaranteed and measures positional stability. Only
> the first is asserted. `judge_noise` reports `mean_stdev`, `max_range`,
> `unanimous_fraction`, and `position_bias` per category, and
> `min_detectable_delta` is the largest of those as a floor on claimable deltas.
> Scores outside `[0, 1]` and non-finite scores are rejected at the boundary rather
> than averaged, `PROXY` bases make a report `citable=False`, and aggregation is
> canonical so input order cannot move the answer.

### US-009 · First published cost-quality frontier `blocked`

As a user, I want to see a real cost-quality curve, so that I can judge the
project on evidence.

**Done when** the frontier is published with all four evaluation controls in
place and the data is reproducible from a committed config.

> **Not done, and the reason is that producing the data would be dishonest.**
> The machinery and the publish gate ship; the curve does not, because there is
> no provider adapter (US-010/US-011) and no credentials to call one. A frontier
> built on the shipped fake would be a real cost-quality curve of a test double,
> and publishing it as evidence would be the exact failure this project exists to
> prevent.
>
> What ships in `inferconomy.frontier`: `build_frontier` assembles a curve from
> measured runs and judged quality, and `Frontier.publish()` raises
> `UnpublishableFrontier` unless all four controls pass, naming each failing one.
> A frontier defaults to `published=False` so an assembled one cannot be mistaken
> for a vetted one. `FrontierConfig` is fingerprinted and a template config is
> committed at `benchmarks/frontier.json`, so the reproducibility claim is a
> comparison of two hashes once real data exists.
>
> Two things this story turned up and fixed: the reference arm now defaults to the
> *largest* budget, because savings must be measured against the arm that was
> given the most compute — defaulting to the smallest made the project's own
> cost-matched-baseline control unsatisfiable. And an oracle bound computed from a
> different baseline is now rejected, because dividing a saving by a bound taken
> against another reference arm is a ratio of two unrelated numbers.
>
> Remaining to finish this story: US-010 and US-011, then a real run against a real
> judge, then publication.

---

## Phase 1 — One working path

### US-010 · Capability probe `todo`

As a developer, I want Inferconomy to detect what my model actually supports, so
that it uses real levers and does not pretend to.

**Done when** the probe reports reasoning-token control, effort control, visible
chain-of-thought, logprobs, and usage reporting, with a documented fallback per
capability.

### US-011 · OpenAI-compatible provider adapter `todo`

As a developer, I want to pass any OpenAI-compatible endpoint, so that one
adapter covers OpenAI, OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral,
and local vLLM / Ollama servers.

**Done when** the adapter normalizes requests, responses, and usage across those
endpoints without provider-specific logic leaking upward.

### US-012 · Decision engine protocol and heuristic adapter `todo`

As a developer, I want a pluggable decision engine, so that task analysis is
replaceable and testable.

**Done when** the protocol is implemented by a zero-cost heuristic and by a fake
returning fixed values, with neither requiring network access.

### US-013 · Policy protocol and reference policy `todo`

As a developer, I want allocation to be a swappable policy, so that the strategy
is not hardcoded and can be supplied or replaced independently.

**Done when** a documented, deliberately conservative reference policy ships as
the default, and a user-supplied policy fully replaces it.

### US-014 · `direct` strategy `todo`

As a user, I want simple requests answered without wasted reasoning, so that I
pay for the task and not for the worst case.

**Done when** `direct` completes a trivial request well within the baseline cost
and with no quality loss.

### US-015 · `reason` strategy `todo`

As a user, I want reasoning to be available as a distinct strategy, so that
allocation is a real choice rather than a single path.

**Done when** `reason` is selectable and measurably differs from `direct` in both
cost and quality on tasks that warrant it.

### US-016 · Sufficiency detection `todo`

As a user, I want inference to stop when enough has been produced, so that I am
not billed for reasoning past the answer.

**Done when** stopping is driven by a defined signal, and the report records
which signal fired.

### US-017 · Escalation `todo`

As a user, I want more computation allocated when the first budget was
insufficient, so that a tight budget never corrupts a hard answer.

**Done when** an insufficient first attempt is detected and escalated, and the
report shows initial and additional budget separately.

### US-018 · Auditable report `todo`

As an operator, I want every decision explained, so that I can trust and debug
the layer.

**Done when** the report covers strategy, capabilities used, budget decisions,
stop reason, and whether cost figures were exact.

---

## Phase 2 — Adaptation and breadth

### US-019 · `decompose` strategy `todo`
### US-020 · `reason` + `verify` strategy `todo`
### US-021 · `reason` + `refine` strategy `todo`
### US-022 · Mid-request strategy switching `todo`
### US-023 · External decision engine adapter `todo`
### US-024 · Anthropic adapter `todo`
### US-025 · Google adapter `todo`
### US-026 · Streaming support `todo`

---

## Phase 3 — Optimization

Budget calibration from observed outcomes · quality-aware allocation · full
benchmark suite · runtime metrics and regression gates.

## Phase 4 — Research

Dynamic strategy composition · inference-state-aware continuation ·
value-of-computation estimation · learned allocation policies. These are
research directions, not commitments. A negative result is publishable.
