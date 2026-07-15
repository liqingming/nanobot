Summarize an archived conversation segment into two separate products.

Output exactly these two XML-like sections, with concise bullets only and no preamble:

<continuation>
- Current objective, completed work, verified facts, relevant changed files, unresolved work or blockers, user constraints, and the next safe action.
</continuation>
<memory-candidates>
- [permanent|durable|ephemeral|correction] only facts useful to long-term memory or later Dream consolidation.
</memory-candidates>

Continuation rules:
- It is for the next 1–10 turns of this same session, not a permanent memory.
- If an existing continuation summary is supplied, replace stale state and merge still-active facts; never merely append it.
- Preserve decisions and negative constraints that prevent repeated investigation or unauthorized changes.
- Keep only conclusions with necessary paths, commands, or evidence; do not copy raw logs or verbose tool output.
- If there is no active work state, output `(nothing)` inside this section.

Memory-candidate rules:
- Include only SNIP facts: Signal (the user would repeat it if forgotten), Novel, Important, and Persistent (relevant after two weeks).
- Prefer user corrections and preferences over solutions, decisions, events, and environment facts.
- Do not include repository-derived code facts, conversational filler, audit breadcrumbs, or skip-tagged entries.
- If there are no candidates, output `(nothing)` inside this section.

Tool records identify their tool name and arguments. Treat successful tests and reads as evidence, but report only their useful conclusion.
