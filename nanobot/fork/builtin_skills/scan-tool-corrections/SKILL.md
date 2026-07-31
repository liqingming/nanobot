---
name: scan-tool-corrections
description: Scan a nanobot runtime session for recurring tool execution mistakes, deduplicate them, and add safe prevention or automatic correction rules to the fork-only tool correction corpus. Use when the user provides a nanobot data/cache directory plus a session ID or session directory and asks to scan errors, update the tool correction corpus, or prevent repeated wasted tool rounds.
---

# Scan tool corrections

Require exactly two inputs:

- `directory`: nanobot data/cache root, its `sessions` directory, or a session directory.
- `session`: session ID/directory name; use `.` when `directory` is already the session directory.

## Workflow

1. Run the bundled scanner:

   ```text
   python <this-skill-dir>/scripts/scan_session_errors.py --directory <directory> --session <session>
   ```

2. Read the JSON report in chronological order. Never copy raw command arguments, environment values, tokens, headers, or credentials into the correction corpus.
3. Compare fingerprints and meanings with `nanobot/fork/agent/tools/corrections.py`:
   - Increment `observed_count` for a repeated root cause.
   - Add one catalog entry for a new root cause, with the next `sequence` value.
   - Use an automatic correction only when it deterministically preserves intent and security boundaries.
   - Otherwise add a short `preventive_hint` to the affected tool description.
   - Do not weaken general deny rules, raise arbitrary limits, retry destructive commands, or rewrite an ambiguous command.
4. Keep implementation in `nanobot/fork/`; modify only the fork correction catalog/wrappers and `tests/fork/` unless the user explicitly broadens scope.
5. Add or update tests for ordering, observed counts, safe behavior, and sensitive-data exclusion.
6. Run focused tests, Ruff, and `git diff --check`. Report unrelated failures separately. Commit or push only when requested.

## Scanner interpretation

- `tool_error`: explicit preparation/execution failure; normally a strong candidate.
- `nonzero_exit`: execution returned a failing exit code even if the audit status was `ok`.
- `diagnostic_error`: traceback or syntax failure visible in tool output.
- Repeated events with the same call ID are one occurrence.
- Treat timeouts as distinct only when the command pattern/root cause differs.

If the session cannot be resolved uniquely, list candidate session directories and ask for the exact ID. Do not guess.
