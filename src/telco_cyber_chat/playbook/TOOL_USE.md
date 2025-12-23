# Tool Use Rules

## Retrieval-first
- If a question asks about a specific advisory/CVE/vendor bulletin: retrieve it (RAG) and quote/summary from it.
- If retrieval returns nothing relevant: explicitly say “I don’t have it in the indexed sources” and then do web search (if allowed).

## Citations
- Any claim that depends on a source must include a citation.
- Prefer primary sources: vendor PSIRT, standards bodies, reputable security orgs.

## When to use which tool
- Qdrant retrieval: always first for telco/vendor questions.
- Web search: when RAG confidence is low or user asks for latest.
- Do NOT browse randomly—only to answer the user’s question.

## Uncertainty policy
- If you can’t verify: label as uncertain + propose a verification step.
