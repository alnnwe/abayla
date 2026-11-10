"""Описание оборудования и версий для воспроизводимости."""
from __future__ import annotations

import os
import platform


def hardware_info() -> str:
    mem = ""
    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
            mem = f", RAM {kb / 1024 / 1024:.1f} GB"
    except OSError:
        pass
    cpu = platform.processor() or platform.machine()
    gpu = "нет GPU"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"{platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} CPU{mem}, {gpu}, Python {platform.python_version()}"


def package_versions() -> dict:
    out = {}
    for name in ("numpy", "pandas", "sklearn", "torch", "transformers", "aiogram"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return out
