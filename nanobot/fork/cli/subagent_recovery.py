"""核验旧执行者停止后补录持久回执的管理命令。"""

import argparse
import json
from pathlib import Path

from nanobot.fork.agent.subagent_receipts import SubagentReceiptStore


def main() -> int:
    parser = argparse.ArgumentParser(description="核验旧执行者停止后，补录可跨重启查询的宿主回执")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--confirm-stopped", action="store_true",
                        help="已核实旧执行者及其外部进程停止；不能仅凭 unknown 使用")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = SubagentReceiptStore(args.data_dir).reconcile(
            args.task_id, args.session, args.workspace,
            confirmed_stopped=args.confirm_stopped, reason=args.reason,
            evidence_file=args.evidence_file,
        )
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
