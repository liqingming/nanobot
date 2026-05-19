# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## write_file vs exec — Writing to User-Specified Paths

When the user asks to write content to a specific file path:

- **Use `write_file`**, not `exec` (e.g., Python/shell script). `exec` runs in the workspace directory, so any relative path in the script resolves to the workspace, NOT the user's intended location.
- Even for absolute paths, prefer `write_file` for clarity and reliability.
- Example: user says "过滤后写回 C:\Users\liqm\Desktop\新建文本文档.txt" → call `write_file` with that exact path. Do not write to `_out.txt` first.

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
