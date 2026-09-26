# Security Policy

## Supported Versions

Inferconomy is currently under active development. Security fixes are
prioritized for the latest version available on the default branch.

| Version                 | Supported   |
| ----------------------- | ----------- |
| Latest / default branch | Yes         |
| Older releases          | Best effort |

## Reporting a Vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

If you discover a security vulnerability in Inferconomy, please report it
privately to the project maintainer.

Include, where possible:

* A description of the vulnerability
* Steps to reproduce the issue
* The affected component or version
* Potential impact
* A proof of concept, if available
* Any suggested mitigation

Please avoid including secrets, credentials, API keys, production data, model
credentials, or other sensitive information in the report.

The maintainer will acknowledge receipt and investigate the report. Once a fix
is available, the vulnerability may be disclosed publicly after reasonable
coordination with affected users and contributors.

## Security-Sensitive Areas

Inferconomy may operate within applications that interact with LLM providers,
inference APIs, model endpoints, application data, prompts, generated outputs,
telemetry systems, and other infrastructure.

Contributors should treat the following as potentially sensitive:

* API keys and authentication tokens
* Model-provider credentials
* Prompts and user inputs
* Generated model outputs
* Inference traces and telemetry
* Application data
* Model configuration
* Usage and cost information
* Internal system metadata

Inferconomy should not expose, persist, or transmit sensitive information
unless explicitly required by the host application or configured by the user.

Never commit secrets, API keys, access tokens, credentials, private keys,
production data, private prompts, or other sensitive information to the
repository.

## Security Best Practices for Contributors

Contributors should:

* Validate and sanitize untrusted inputs where appropriate.
* Avoid logging credentials, secrets, prompts, or sensitive model outputs.
* Minimize the collection and retention of inference telemetry.
* Treat provider responses and model-generated content as untrusted data.
* Avoid introducing unnecessary network access or external dependencies.
* Review changes affecting provider integrations, telemetry, persistence, and
  runtime execution carefully.
* Keep dependencies reasonably up to date and address known security issues.
* Never hard-code credentials or authentication material in source code or tests.

## Scope

Security reports are encouraged for vulnerabilities that could result in:

* Unauthorized access to credential
