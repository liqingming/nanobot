"""编译真实 Java 客户端并与 Python 回环服务联调；不启动 Rider 或模型。"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from nanobot.fork.cli.ide_bridge import IDEBridge
from nanobot.fork.cli.ide_context import PendingIDEContext


async def smoke(rider_home: Path) -> None:
    root = Path(__file__).resolve().parent
    jdk_bin = rider_home / "jbr/bin"
    suffix = ".exe" if os.name == "nt" else ""
    artifact = root / "build/nanobot-context-0.1.1.zip"
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=root / "build") as temporary:
        work = Path(temporary)
        (work / "other").mkdir()
        with ZipFile(artifact) as archive:
            jar = work / "plugin.jar"
            jar.write_bytes(archive.read("nanobot-context/lib/nanobot-context.jar"))
        classpath = os.pathsep.join([
            str(jar), str(rider_home / "lib/module-intellij.libraries.gson.jar"), str(work),
        ])
        subprocess.run([
            str(jdk_bin / f"javac{suffix}"), "-encoding", "UTF-8", "--release", "21",
            "-cp", classpath, "-d", str(work),
            str(root / "src/test/java/dev/nanobot/rider/BridgeSmoke.java"),
        ], check=True)
        pending = PendingIDEContext(work, lambda labels: None)
        pending.switch("cli:java-smoke")
        bridge = IDEBridge(pending, lambda: "联调话题", work / "registry")
        await bridge.start()
        try:
            process = await asyncio.create_subprocess_exec(
                str(jdk_bin / f"java{suffix}"), "-Dfile.encoding=UTF-8",
                "-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8", "-cp", classpath,
                "dev.nanobot.rider.BridgeSmoke", str(bridge.registry), str(work),
            )
            try:
                code = await asyncio.wait_for(process.wait(), 30)
            except TimeoutError:
                process.kill()
                await process.wait()
                raise
            if code:
                raise RuntimeError(f"Java 联调失败：{code}")
            assert len(pending.items) == 1
            assert pending.items[0].text == "未保存的中文代码\nConsole.WriteLine(1);"
            assert "未保存的中文代码" in pending.consume("请解释")
            assert not pending.items
            print("Python 端已验证：仅加入待发送上下文，提交时消费快照")
        finally:
            await bridge.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rider-home", required=True, type=Path)
    asyncio.run(smoke(parser.parse_args().rider_home))
