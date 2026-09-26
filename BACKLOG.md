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

### US-002 · Core request and response types `todo`

As a developer, I want typed `Request`, `Response`, `Usage`, and
`OptimizationReport` objects, so that results are self-describing and not
stringly-typed dictionaries.

**Done when** all four types exist, are immutable where appropriate, and are
covered by round-trip serialization tests.

### US-003 · Client protocol and fake client `todo`

As a developer, I want to inject any LLM client behind a narrow protocol, so
that the library never depends on a specific provider SDK.

**Done when** a `Client` protocol exists, a deterministic fake client is
included, and every test in the suite runs with no network access.

### US-004 · Token accounting, exact versus estimated `todo`

As an operator, I want to know whether reported token counts are exact or
estimated, so that I never act on a cost number of unknown provenance.

**Done when** usage carries an explicit `exact: bool`, the estimate path is
covered by tests, and estimated results are flagged in the report.

### US-005 · Versioned cost table `todo`

As an operator, I want provider pricing kept in a separate, versioned data file
that I can override, so that price changes do not require a code release.

**Done when** costs load from a bundled JSON file, can be replaced by a user file,
and are versioned independently of the package.

### US-006 · Fixed-budget baseline runner `todo`

As a researcher, I want to run a benchmark with a fixed strategy and budget, so
that I have a reference point to compare against.

**Done when** a baseline run is reproducible from a config alone and emits the
full metric row.

### US-007 · Oracle budgeter `todo`

As a researcher, I want to know how much *perfect* allocation would have saved,
so that I can tell a good result from a mediocre one.

**Done when** the oracle reports an upper bound on achievable savings and is
clearly labelled as unattainable in practice.

### US-008 · Null condition and calibrated judge `todo`

As a researcher, I want judge noise measured and a provably-null configuration
included, so that reported quality deltas are not artifacts of judging.

**Done when** the null configuration measures as a difference of zero and judge
noise is quantified per benchmark.

### US-009 · First published cost-quality frontier `todo`

As a user, I want to see a real cost-quality curve, so that I can judge the
project on evidence.

**Done when** the frontier is published with all four evaluation controls in
place and the data is reproducible from a committed config.

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
