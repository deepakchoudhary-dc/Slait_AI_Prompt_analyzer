# Dependency Vetting Notes

## Scope
This project keeps runtime dependencies minimal. Core evaluator is standard-library-only. Streamlit is optional for web UI.

## Streamlit Vetting Snapshot (March 31, 2026)

Reviewed sources:
- PyPI package metadata for Streamlit (`streamlit`)
- GitHub Advisory Database entries for `streamlit` (pip ecosystem)
- OSV index (ecosystem/package listing)

Notable advisories observed:
- GHSA-7p48-42j8-8846 / CVE-2026-33682 (Windows SSRF/NTLM exposure)
  - Affected: `< 1.54.0`
  - Patched: `1.54.0`
- GHSA-rxff-vr5r-8cj5 / CVE-2024-42474 (Windows path traversal)
  - Affected: `< 1.37.0`
  - Patched: `1.37.0`
- GHSA-9c6g-qpgj-rvxw / CVE-2023-27494 (legacy XSS range)
  - Affected: `>= 0.63.0, < 0.81.0`
  - Patched: `0.81.0`

## Pinning Decision
- Pinned optional UI dependency to `streamlit==1.55.0`.
- This version is newer than published patched baselines above.

## Local Hardening Recommendations
- Prefer localhost-only execution during development.
- Keep Streamlit updated with patch releases.
- Run periodic advisory checks in CI/CD.
- On Windows, apply SMB/NTLM hardening controls for defense-in-depth.

## Future Dependency Intake Policy
Before adding any new package:
1. Verify maintainer trust and release provenance.
2. Review GHSA/CVE/OSV advisories and affected ranges.
3. Pin exact version and justify necessity.
4. Record decision and risk tradeoffs in this document.
