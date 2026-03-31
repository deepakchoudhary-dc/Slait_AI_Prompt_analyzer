# Security Policy

## Scope
This project evaluates AI coding transcripts.

- Core evaluation engine is implemented with Python standard library modules only.
- Optional web UI uses a pinned Streamlit dependency documented in `requirements-ui.txt`.

## Dependency Policy
- No runtime third-party packages are allowed by default.
- New packages require explicit approval and a security review before use.
- Version pinning is mandatory when dependencies are introduced.
- Lockfiles must be committed for reproducible builds when package managers are used.
- Known advisories and mitigation status must be recorded in `docs/dependency_vetting.md`.

## Package Vetting Checklist (Before Install)
1. Verify maintainer identity and project ownership history.
2. Review recent release activity for suspicious changes.
3. Check known vulnerabilities (CVE/GHSA/OSV/advisories).
4. Confirm integrity expectations (hash pinning and reproducible versions).
5. Validate whether a standard-library alternative exists.

## Operational Guardrails
- Prefer least privilege for credentials and tokens.
- Never hard-code secrets.
- Redact sensitive values in transcript processing when possible (`--strip-sensitive`).
- Fail safely on malformed inputs and preserve auditability through logs.
- Run frontend services on trusted networks only; default to localhost during development.
- On Windows hosts, apply SMB/NTLM hardening controls when running web apps.

## Reporting Security Issues
Open a private security report to the maintainers with:
- Affected component and version
- Reproduction steps
- Impact assessment
- Suggested fix or mitigation
