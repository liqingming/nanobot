"""使用已安装的 Rider SDK 离线构建插件，无需下载 Gradle/完整 IDE。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile


def build(rider_home: Path, jdk_home: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent
    rider_home = rider_home.resolve(strict=True)
    java_home = jdk_home or rider_home / "jbr"
    javac = java_home / "bin" / ("javac.exe" if os.name == "nt" else "javac")
    if not javac.is_file():
        raise SystemExit("未找到 javac，请使用 --jdk-home 指定 JDK 21 或更新版本。")
    jars = sorted((rider_home / "lib").glob("*.jar"))
    if not jars:
        raise SystemExit("Rider SDK 的 lib 目录无 JAR 文件。")
    sources = sorted((root / "src/main/java").rglob("*.java"))
    output = root / "build"
    output.mkdir(exist_ok=True)
    # 每次全新编译，避免旧 class 混入成品；临时目录仅位于插件 build 下。
    with tempfile.TemporaryDirectory(prefix="compile-", dir=output) as temporary:
        work = Path(temporary)
        classes = work / "classes"
        classes.mkdir()
        args = ["-encoding", "UTF-8", "--release", "21", "-classpath",
                os.pathsep.join(str(jar) for jar in jars), "-d", str(classes)]
        args.extend(str(source) for source in sources)
        argfile = work / "javac.args"
        argfile.write_text("\n".join('"' + arg.replace("\\", "/") + '"' for arg in args),
                           encoding="utf-8")
        subprocess.run([str(javac), "@" + str(argfile)], check=True)
        shutil.copytree(root / "src/main/resources", classes, dirs_exist_ok=True)
        jar_path = work / "nanobot-context.jar"
        with ZipFile(jar_path, "w") as jar:
            for file in sorted(classes.rglob("*")):
                if file.is_file():
                    jar.write(file, file.relative_to(classes).as_posix())
        artifact = output / "nanobot-context-0.1.1.zip"
        with ZipFile(artifact, "w") as package:
            package.write(jar_path, "nanobot-context/lib/nanobot-context.jar")
    print(artifact)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rider-home", type=Path, required=True)
    parser.add_argument("--jdk-home", type=Path)
    options = parser.parse_args()
    build(options.rider_home, options.jdk_home)
