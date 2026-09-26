# Contributing to Inferconomy

Thank you for contributing to Inferconomy.

Inferconomy is an open-source LLM inference optimization library focused on
dynamically selecting inference strategies, allocating computation, and
optimizing token usage while preserving task quality.

## Getting Started

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Add or update tests where appropriate.
5. Run the project's validation and test suite.
6. Open a pull request with a clear description of the change.

## Pull Requests

A good pull request should:

* Explain what changed and why.
* Keep the scope focused.
* Include tests for new behavior where practical.
* Avoid unrelated formatting or refactoring.
* Document configuration or behavioral changes.
* Include benchmarks or evaluation results when the change affects inference
  efficiency, latency, token usage, or output quality.
* Clearly identify security-sensitive changes.

For larger architectural or inference-strategy changes, open an issue first so
the design and trade-offs can be discussed before substantial implementation
work begins.

## Code Quality

Contributions should prioritize:

* Correctness
* Inference quality
* Token efficiency
* Low latency and runtime overhead
* Security
* Maintainability
* Clear interfaces
* Good error handling
* Testability
* Minimal unnecessary complexity

## Inference Strategies

When adding or modifying an inference strategy:

* Clearly document its purpose and expected behavior.
* Explain when the strategy should be selected.
* Define its computational and token requirements where applicable.
* Avoid assumptions that unnecessarily restrict supported LLM providers.
* Include evaluation results when practical.
* Consider both quality preservation and inference cost.
* Ensure the strategy can fail safely when its assumptions are not satisfied.

Inference optimization should not prioritize token reduction at the expense of
meaningful task quality.

## Budget Allocation

Changes involving reasoning or output budgets should:

* Clearly document the budgeting mechanism.
* Avoid arbitrary fixed limits when adaptive allocation is appropriate.
* Consider quality, latency, and token cost together.
* Handle insufficient budgets gracefully.
* Support adaptive escalation when additional computation is necessary.
* Avoid unnecessary inference restarts where continuation is possible.

The goal is minimum-sufficient inference rather than simply minimizing token
count.

## Model and Provider Compatibility

Inferconomy is designed to operate as a model- and provider-independent
inference optimization layer.

Contributions should:

* Avoid unnecessary coupling to a specific LLM provider.
* Keep provider-specific behavior isolated behind appropriate interfaces.
* Clearly document provider-specific assumptions when unavoidable.
* Preserve compatibility with different host applications and inference
  environments where practical.

## Decision and Adaptation Logic

Changes affecting inference decisions or runtime adaptation should include
enough information to understand:

* What task characteristics are considered.
* How an inference strategy is selected.
* How the initial computation budget is determined.
* What signals indicate that additional computation is required.
* How escalation or strategy changes occur.
* What conditions terminate inference.
* How quality and efficiency are evaluated.

Avoid designs that introduce unnecessary computation, unpredictable behavior,
or hidden external side effects.

## Tests and Evaluation

Run the relevant tests and validation checks before opening a pull request.

Changes affecting inference behavior should include appropriate evaluation
where practical.

Depending on the change, this may include:

* Unit tests
* Integration tests
* Token-usage measurements
* Latency measurements
* Quality evaluations
* Budget-allocation evaluations
* Regression benchmarks
* Provider compatibility tests

If a test or benchmark cannot be run locally, explain why in the pull request.

## Research Contributions

Research-oriented contributions are welcome.

When introducing a new inference optimization technique, strategy, or
algorithm, contributors should provide relevant background, references to
prior work where applicable, and a clear explanation of what is different or
improved.

Claims about efficiency or quality improvements should be supported by
reproducible experiments whenever practical.

## Licensing

By contributing to Inferconomy, you agree that your contributions are
provided under the project's MIT License, subject to the terms of that
license.

You retain copyright in your contributions unless you explicitly transfer it
under a separate agreement.

## Community

Please keep discussions constructive, respectful, and technical.

Contributions should focus on improving Inferconomy, its inference
optimization capabilities, and its ecosystem.
