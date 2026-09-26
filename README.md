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
| `inferconomy.types` | `Request`, `Response`, `Usage`, `OptimizationReport` |
| `inferconomy.capabilities` | Probes what the target model supports |
| `inferconomy.decision` | `DecisionEngine` protocol + adapters |
| `inferconomy.strategy` | Strategy catalogue and selection |
| `inferconomy.policy` | `Policy` protocol — budget allocation, escalation, stopping |
| `inferconomy.runtime` | The generate/inspect/continue/stop loop |
| `inferconomy.providers` | Provider clients and usage extraction |
| `inferconomy.telemetry` | Token, cost, and latency accounting |

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
print(result.usage)        # exact token + cost accounting
print(result.report)       # why this strategy, this budget, this stop
```

A report, roughly:

```python
result.report = OptimizationReport(
    strategy="direct",
    capabilities={"reasoning_budget": False, "usage_reporting": True},
    initial_budget=180,
    additional_budget=0,
    total_budget=180,
    escalated=False,
    stopped_on="sufficiency",
    cost_exact=True,
)
```

The report is a typed object, not a dict, and it is intended to be
machine-readable. Every optimization should be auditable after the fact, and
`policy=<your policy>` is the seam for supplying your own allocation behaviour.

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

- [ ] `pyproject.toml`, package skeleton, installable and importable
- [ ] Token and cost telemetry with exact-vs-estimated accounting
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
