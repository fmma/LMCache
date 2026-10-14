#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one leg of the GDS L1 backend comparison.

A leg is a storage configuration for the KV cache: ``none`` (no LMCache at
all, prefill is always recomputed), ``cufile``, ``opends``, or ``ugds``. The
``none`` leg is the reference every storage leg has to beat -- without it a
cache-hit TTFT number says nothing about whether loading KV from storage is
cheaper than recomputing it.

All legs share the request driver, prompt generator and test shapes with
:mod:`bench_e2e`; only service startup differs. Results go to JSON with the
same schema for every leg so one plotter handles all of them.

Usage:
    python bench_legs.py --leg none   --model /root/models/Qwen3-0.6B
    python bench_legs.py --leg cufile --slab-dir /mnt/datasets/lmcache_gds
    python bench_legs.py --leg opends --slab-dir /mnt/datasets/lmcache_gds \
        --io-threads 4 --queue-depth 32
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_e2e as be

LEGS = ("none", "cufile", "opends", "ugds")


def start_vllm(model: str, env: dict, log_path: Path, use_lmcache: bool):
    """Start vLLM, with the LMCache KV connector only when a leg needs it."""
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = [
        vllm_bin,
        "serve",
        model,
        "--port",
        str(be.VLLM_PORT),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
    ]
    if use_lmcache:
        cmd += [
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_role": "kv_both",
                    "kv_load_failure_policy": "recompute",
                    "kv_connector_extra_config": {
                        "lmcache.mp.port": be.LMCACHE_PORT
                    },
                }
            ),
        ]
    env = {
        **env,
        "PYTHONUNBUFFERED": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    print(f"Starting vLLM (lmcache={use_lmcache}); log: {log_path}", flush=True)
    proc = be._start_logged_process(cmd, env, log_path)
    try:
        be._wait_for_service(
            f"http://127.0.0.1:{be.VLLM_PORT}/health",
            "vLLM",
            proc,
            log_path,
            timeout=600,
        )
    except Exception:
        be.kill_proc(proc)
        raise
    return proc


def wait_for_port(port: int, label: str, proc: subprocess.Popen,
                  log_path: Path, timeout: int = 600) -> None:
    """Wait until the ZMQ socket accepts connections.

    The server has no HTTP health endpoint, so readiness is a TCP connect on
    the ZMQ port. The timeout is generous because --gds-l1-prewrite zero-fills
    the whole slab before the socket ever binds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"{label} exited with status {rc} during startup\n"
                f"--- {log_path} (tail) ---\n{be._log_tail(log_path)}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                print(f"  {label} ready", flush=True)
                return
        time.sleep(2)
    raise TimeoutError(
        f"{label} not ready after {timeout}s\n"
        f"--- {log_path} (tail) ---\n{be._log_tail(log_path)}"
    )


def start_lmcache_server(args, env: dict, log_path: Path):
    """Start the lmcache server for a storage leg."""
    path = args.device if args.leg == "ugds" else args.slab_dir
    lmcache_bin = os.path.join(os.path.dirname(sys.executable), "lmcache")
    cmd = [
        lmcache_bin,
        "server",
        "--host",
        "localhost",
        "--port",
        str(be.LMCACHE_PORT),
        "--chunk-size",
        str(args.chunk_size),
        "--max-gpu-workers",
        "4",
        "--l1-size-gb",
        str(args.l1_size_gb),
        "--eviction-policy",
        "LRU",
        "--gds-l1-backend",
        args.leg,
        "--gds-l1-path",
        path,
    ]
    if args.prewrite:
        cmd.append("--gds-l1-prewrite")
    print(f"Starting lmcache server ({args.leg}, {path}); log: {log_path}",
          flush=True)
    proc = be._start_logged_process(cmd, env, log_path)
    try:
        wait_for_port(be.LMCACHE_PORT, "lmcache server", proc, log_path)
    except Exception:
        be.kill_proc(proc)
        raise
    return proc


def leg_env(args) -> dict:
    env = {**os.environ}
    if args.leg == "opends":
        env["LMCACHE_OPENDS_LIB"] = args.opends_lib
        if args.io_threads:
            env["OPENDS_AISIO_IO_THREADS"] = str(args.io_threads)
        if args.queue_depth:
            env["OPENDS_AISIO_QUEUE_DEPTH"] = str(args.queue_depth)
        if args.cpu_mask:
            env["OPENDS_AISIO_CPU_MASK"] = args.cpu_mask
    return env


def provenance(args, env: dict) -> dict:
    """Record what actually ran, so a number can be traced back to a stack."""
    info = {"leg": args.leg, "model": args.model}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        info["torch"] = f"unavailable: {exc}"
    try:
        import vllm

        info["vllm"] = vllm.__version__
    except Exception as exc:
        info["vllm"] = f"unavailable: {exc}"
    for key in (
        "LMCACHE_OPENDS_LIB",
        "OPENDS_AISIO_IO_THREADS",
        "OPENDS_AISIO_QUEUE_DEPTH",
        "OPENDS_AISIO_CPU_MASK",
    ):
        if key in env:
            info[key] = env[key]
    if args.leg in ("cufile", "opends"):
        info["nvidia_fs_stats_before"] = read_nvidia_fs_stats()
    return info


def read_nvidia_fs_stats():
    """nvidia-fs counters distinguish real GDS DMA from a POSIX fallback."""
    try:
        return Path("/proc/driver/nvidia-fs/stats").read_text()
    except OSError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leg", required=True, choices=LEGS)
    p.add_argument("--model", default="/root/models/Qwen3-0.6B")
    p.add_argument("--slab-dir", default="/mnt/datasets/lmcache_gds")
    p.add_argument("--device", default="/dev/ugds_drv0",
                   help="raw device for the ugds leg")
    p.add_argument("--opends-lib", default="libopends_aisio.so")
    p.add_argument("--io-threads", type=int, default=0)
    p.add_argument("--queue-depth", type=int, default=0)
    p.add_argument("--cpu-mask", default="")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--l1-size-gb", type=int, default=8)
    p.add_argument("--prewrite", action="store_true", default=True,
                   help="zero-fill the slab so extent conversion is not "
                        "inside the measured path (default on)")
    p.add_argument("--no-prewrite", dest="prewrite", action="store_false")
    p.add_argument("--output", default="")
    p.add_argument("--log-dir", default="/root/lmcache-bench-logs")
    args = p.parse_args()

    be.MODEL = args.model
    if not args.output:
        args.output = f"bench_leg_{args.leg}.json"

    env = leg_env(args)
    log_dir = Path(args.log_dir)
    if args.leg in ("cufile", "opends"):
        Path(args.slab_dir).mkdir(parents=True, exist_ok=True)

    meta = provenance(args, env)
    procs = []
    try:
        if args.leg != "none":
            procs.append(start_lmcache_server(
                args, env, log_dir / f"lmcache-{args.leg}.log"))
        procs.append(start_vllm(
            args.model, env, log_dir / f"vllm-{args.leg}.log",
            use_lmcache=args.leg != "none"))

        print("\nWarmup request...", flush=True)
        be._request(be._unique_prompt(64, 1))
        time.sleep(1)

        seq = be.run_sequential()
        conc = be.run_concurrent()

        meta["nvidia_fs_stats_after"] = read_nvidia_fs_stats()
        out = {**meta, **seq, **conc}
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}", flush=True)
    finally:
        print("\nShutting down services...", flush=True)
        for proc in reversed(procs):
            be.kill_proc(proc)
        subprocess.run(["pkill", "-f", "VLLM::"], check=False)
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
