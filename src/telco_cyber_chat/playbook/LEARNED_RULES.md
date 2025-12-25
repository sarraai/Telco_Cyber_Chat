# Learned Rules (Auto-updated)

## 2025-12-25

### Batch summary
- Update generated from 2 new feedback rows.
- Role `end_user`: n=2, helpful_rate=50%, unsafe_flags=0
- Pattern: refusals were safe but sometimes rated not helpful.

### New/updated rules
- When refusing a request (e.g., phishing email example), keep the refusal to 1–2 sentences and immediately provide a *useful safe substitute* that matches the user’s intent.
- For “example of phishing” requests: do NOT generate a phishing email. Instead provide:
  - a short *anatomy* template of red flags (subject tone, spoofing cues, urgency language) **without** realistic wording that could be sent,
  - 3–5 concrete detection checks (SPF/DKIM/DMARC, link hover checks, attachment sandboxing),
  - an employee training exercise prompt using clearly fake placeholders (e.g., `attacker.example`, `example.com`) that cannot be copy-pasted into a real attack.
- Prefer telecom-relevant defensive actions when possible: email gateway rules, alerting on look-alike domains, and reporting workflow.
- Avoid mixing unrelated CVEs/vendors into general explanations unless the user asked about them specifically (keep answers scoped).
