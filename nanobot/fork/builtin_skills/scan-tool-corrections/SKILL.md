---
name: scan-tool-corrections
description: Locate one named nanobot topic from a supplied work directory using nanobot's workspace-cache and session-metadata rules, scan that topic for recurring tool execution mistakes, and add safe corrections to the fork-only corpus. Use when the user supplies a work directory plus a topic name and asks to scan errors, update the tool correction corpus, or prevent repeated wasted tool rounds; never require a cache path, session ID, or session directory.
---

# Scan tool corrections

Require exactly two user inputs:

- `directory`: the nanobot work directory used when the topic was created.
- `topic`: the displayed topic name (`metadata.cli_title`).

Never ask the user for a cache directory, session ID, or session directory.

## Workflow

1. Run the bundled scanner:

   ```text
   python <this-skill-dir>/scripts/scan_session_errors.py --directory <directory> --topic <topic>
   ```

   The scanner applies nanobot's storage rules: map the work directory to its workspace cache, read the first metadata record from `sessions/*.jsonl`, match `metadata.cli_title`, derive the safe session directory from its key, then open that topic's `runtime.log`.

2. Read the JSON report in chronological order. Never copy raw command arguments, environment values, tokens, headers, or credentials into the correction corpus.
3. Compare fingerprints and meanings with `nanobot/fork/agent/tools/corrections.py`:
   - Increment `observed_count` for a repeated root cause.
   - Add one catalog entry for a new root cause, with the next `sequence` value.
   - Use an automatic correction only when it deterministically preserves intent and security boundaries.
   - Otherwise add a short `preventive_hint` to the affected tool description.
   - Do not weaken general deny rules, raise arbitrary limits, retry destructive commands, or rewrite an ambiguous command.
4. Keep implementation in `nanobot/fork/`; modify only the fork correction catalog/wrappers and `tests/fork/` unless the user explicitly broadens scope.
5. Add or update tests for topic resolution, ordering, observed counts, safe behavior, and sensitive-data exclusion.
6. Run focused tests, Ruff, and `git diff --check`. Report unrelated failures separately. Commit or push only when requested.

## Scanner interpretation

- `tool_error`: explicit preparation/execution failure; normally a strong candidate.
- `nonzero_exit`: execution returned a failing exit code even if the audit status was `ok`.
- `diagnostic_error`: traceback or syntax failure visible in tool output.
- Repeated events with the same call ID are one occurrence.
- Treat timeouts as distinct only when the command pattern/root cause differs.

Require an exact topic-title match after trimming surrounding whitespace. If zero or multiple sessions match, return available/candidate topics and stop; do not select by recency.