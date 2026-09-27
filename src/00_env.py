"""00 - Environment gate and provenance record.

Two jobs:
  1. HARD GATE: refuse to continue unless CUDA is actually available. A
     CPU-only torch wheel has silently installed itself on this machine
     before, which would turn script 04 into a multi-hour CPU job.
  2. Record the exact hardware and library versions into results/env.json so
     every latency number in the paper is attributable to a known machine.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def sh(cmd):
    """Run a command, return stripped stdout or None."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def cpu_info():
    info = {"platform_processor": platform.processor()}
    name = sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"')
    if name:
        info["model"] = name.splitlines()[0].strip()
    cores = sh(
        'powershell -NoProfile -Command '
        '"(Get-CimInstance Win32_Processor).NumberOfCores"'
    )
    logical = sh(
        'powershell -NoProfile -Command '
        '"(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"'
    )
    if cores:
        info["physical_cores"] = int(cores.splitlines()[0])
    if logical:
        info["logical_cores"] = int(logical.splitlines()[0])
    import os

    info["os_cpu_count"] = os.cpu_count()
    return info


def ram_info():
    total = sh(
        'powershell -NoProfile -Command '
        '"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"'
    )
    if total:
        try:
            b = int(total.splitlines()[0])
            return {"total_bytes": b, "total_gib": round(b / 1024 ** 3, 2)}
        except ValueError:
            pass
    return {}


def gpu_info():
    out = sh(
        "nvidia-smi --query-gpu=name,memory.total,driver_version "
        "--format=csv,noheader"
    )
    info = {"nvidia_smi": out}
    if out:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 3:
            info.update(
                {"name": parts[0], "memory_total": parts[1], "driver_version": parts[2]}
            )
    return info


def versions():
    v = {"python": sys.version.split()[0], "python_full": sys.version}
    for mod in (
        "torch",
        "transformers",
        "sklearn",
        "statsmodels",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "nltk",
    ):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            v[mod] = "NOT INSTALLED ({})".format(type(exc).__name__)
    return v


def main():
    C.banner("00 - ENVIRONMENT GATE")

    import torch

    avail = torch.cuda.is_available()
    print("torch                 : {}".format(torch.__version__))
    print("torch.version.cuda    : {}".format(torch.version.cuda))
    print("cuda.is_available()   : {}".format(avail))

    if not avail:
        print("\n" + "!" * 78)
        print("FATAL: torch.cuda.is_available() is False.")
        print("Refusing to fall back to CPU training, as instructed.")
        print("Most likely cause: a CPU-only torch wheel got installed.")
        print("  torch.version.cuda == None  =>  CPU-only wheel; reinstall with:")
        print("    pip uninstall -y torch")
        print("    pip install torch --index-url "
              "https://download.pytorch.org/whl/cu124")
        print("!" * 78)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    print("GPU name              : {}".format(gpu_name))
    print("compute capability    : {}.{}".format(*cap))
    print("GPU total memory      : {:.2f} GiB".format(props.total_memory / 1024 ** 3))
    print("device count          : {}".format(torch.cuda.device_count()))
    print("cudnn enabled         : {}".format(torch.backends.cudnn.enabled))
    print("torch.get_num_threads : {}".format(torch.get_num_threads()))

    # prove the GPU really computes, not just that it is enumerated
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(a @ b).all().item())
    print("matmul smoke test     : {}".format("PASS" if ok else "FAIL"))
    if not ok:
        sys.exit(1)

    env = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cpu": cpu_info(),
        "ram": ram_info(),
        "gpu": gpu_info(),
        "gpu_torch": {
            "name": gpu_name,
            "compute_capability": "{}.{}".format(*cap),
            "total_memory_gib": round(props.total_memory / 1024 ** 3, 2),
            "multi_processor_count": props.multi_processor_count,
            "cuda_available": True,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "versions": versions(),
        "seed": C.SEED,
    }
    C.save_json(env, C.RESULTS / "env.json", "env")

    print("\nCPU : {}".format(env["cpu"].get("model")))
    print("      {} physical / {} logical cores".format(
        env["cpu"].get("physical_cores"), env["cpu"].get("logical_cores")))
    print("RAM : {} GiB".format(env["ram"].get("total_gib")))
    print("GPU : {}".format(env["gpu"].get("name")))
    print("\nGATE PASSED - GPU training is available.")


if __name__ == "__main__":
    main()
