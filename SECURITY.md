# Security policy

Please report vulnerabilities by email to **security@vector1.ai**, not in a public issue.
Include a description, steps to reproduce and the impact you expect. You will get a
first reply within 48 hours and a status update within 7 days.

Quorum agents can call LLM APIs and act on a domain. When you run it:

- Keep API keys in environment variables, never in code or config committed to git.
- Keep hard rules and the kill switch outside anything the learning loop can change.
- Run new domains in simulation (the shadow twin) before anything acts on live systems.
