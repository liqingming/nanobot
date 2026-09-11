"""持久化宿主子任务回执；进程失联与确认停止分开，恢复必须留存核验依据。"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout

from nanobot.utils.atomic_write import replace_file_with_retry


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _workspace(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


class SubagentReceiptStore:
    """每个执行者持有独立 OS 文件锁，回执落盘后才释放；不按 PID 或超时猜测停止。"""

    def __init__(self, data_dir: Path):
        self.root = Path(data_dir).resolve() / "subagent-receipts"
        self.instance = uuid.uuid4().hex
        self._leases: dict[str, FileLock] = {}

    def _path(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("任务 ID 不能为空")
        digest = hashlib.sha256(task_id.encode()).hexdigest()
        return self.root / (digest + ".json")

    def _lock(self, task_id: str) -> FileLock:
        return FileLock(str(self._path(task_id).with_suffix(".lock")), timeout=0)

    def read(self, task_id: str) -> dict | None:
        try:
            row = json.loads(self._path(task_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if (not isinstance(row, dict) or row.get("version") != 1
                or row.get("task_id") != task_id or not isinstance(row.get("session"), str)
                or not isinstance(row.get("workspace"), str)
                or not isinstance(row.get("receipt"), dict)
                or row["receipt"].get("task_id") != task_id):
            raise ValueError("子任务回执损坏，不能据此确认停止")
        return row

    @staticmethod
    def matches(row: dict, session: str, workspace: Path) -> bool:
        return (row["session"], row["workspace"]) == (session, _workspace(workspace))

    def _write(self, row: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(row["task_id"])
        fd, name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=self.root)
        temp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(row, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            replace_file_with_retry(temp, path)
            if os.name != "nt":
                fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            temp.unlink(missing_ok=True)

    def register(self, task_id: str, session: str, workspace: Path, receipt: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        lease = self._lock(task_id)
        lease.acquire()
        try:
            if self.read(task_id) is not None:
                raise ValueError("任务 ID 已登记，不能覆盖或重复启动")
            self._write({
                "version": 1, "task_id": task_id, "session": session,
                "workspace": _workspace(workspace), "created_at": _now(),
                "host": {"instance": self.instance, "pid": os.getpid(),
                         "machine": socket.gethostname()},
                "receipt": receipt,
            })
            self._leases[task_id] = lease
        except BaseException:
            lease.release()
            raise

    def finish(self, task_id: str, receipt: dict) -> dict:
        row = self.read(task_id)
        if row is None or task_id not in self._leases:
            raise ValueError("缺少当前执行者的持久化归属，不能提交终态")
        final = {**receipt, "confirmation_source": "host_cleanup"}
        if not final.get("worker_stopped"):
            final["confirmation_source"] = "cleanup_unverified"
        row.update(receipt=final, finished_at=_now())
        self._write(row)
        self._leases.pop(task_id).release()
        return final

    def inspect(self, task_id: str, session: str, workspace: Path) -> dict | None:
        row = self.read(task_id)
        if row is None or not self.matches(row, session, workspace):
            return None
        receipt = row["receipt"]
        if receipt.get("worker_stopped") is True:
            return dict(receipt)
        try:
            with self._lock(task_id):
                # 旧进程退出仍不证明它创建的外部进程已清理。
                return {**receipt, "state": "unknown", "worker_stopped": False,
                        "recovery_required": True, "owner_available": False,
                        "detail": "执行者已失联或清理未确认；核实外部进程后使用恢复命令。"}
        except Timeout:
            return {**receipt, "state": "running", "worker_stopped": False,
                    "owner_available": True,
                    "detail": "执行者仍由其他宿主管理；请在原宿主等待或取消。"}

    def list(self, session: str, workspace: Path) -> list[dict]:
        rows = []
        for path in self.root.glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
                raise ValueError("子任务回执损坏")
            if self.matches(row, session, workspace):
                receipt = self.inspect(row["task_id"], session, workspace)
                if receipt is not None:
                    rows.append(receipt)
        return rows

    def reconcile(self, task_id: str, session: str, workspace: Path, *,
                  confirmed_stopped: bool, reason: str, evidence_file: Path) -> dict:
        """管理员核验后的显式补录；不是由模型把 unknown 推断成 stopped。"""
        if confirmed_stopped is not True or not reason.strip():
            raise ValueError("必须确认执行者及外部进程已停止，并提供核验说明")
        evidence_file = evidence_file.resolve(strict=True)
        evidence_hash = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock(task_id):
                row = self.read(task_id)
                if row is not None and not self.matches(row, session, workspace):
                    raise ValueError("任务属于其他话题或工作区，拒绝恢复")
                if row is not None and row["receipt"].get("worker_stopped") is True:
                    return dict(row["receipt"])
                prior = row
                receipt = {
                    "task_id": task_id, "state": "stopped", "worker_stopped": True,
                    "stop_reason": "reconciled", "business_success": "unverified",
                    "confirmation_source": "operator_verified",
                }
                self._write({
                    "version": 1, "task_id": task_id, "session": session,
                    "workspace": _workspace(workspace), "receipt": receipt,
                    "finished_at": _now(), "previous_record": prior,
                    "recovery": {"operator": getpass.getuser(), "reason": reason,
                                 "evidence_file": str(evidence_file),
                                 "evidence_sha256": evidence_hash},
                })
                return receipt
        except Timeout as exc:
            raise ValueError("执行者仍持有宿主锁，不能解除占用或覆盖状态") from exc
