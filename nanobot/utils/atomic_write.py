"""Atomic file replacement helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike[str]]

DEFAULT_REPLACE_RETRIES = 3
DEFAULT_REPLACE_RETRY_DELAY_SECONDS = 0.05


def replace_file_with_retry(
    tmp_path: PathLike,
    path: PathLike,
    *,
    retries: int = DEFAULT_REPLACE_RETRIES,
    delay_seconds: float = DEFAULT_REPLACE_RETRY_DELAY_SECONDS,
) -> None:
    """Replace a file, tolerating short Windows file-lock races."""
    tmp = Path(tmp_path)
    target = Path(path)
    for attempt in range(retries + 1):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt >= retries:
                raise
            time.sleep(delay_seconds)
