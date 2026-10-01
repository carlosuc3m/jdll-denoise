"""Reproducible Noise2Fast benchmark matrix including process-cold wall time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="256,512,1024,2048")
    parser.add_argument("--efforts", default="quick,balanced,thorough")
    parser.add_argument("--devices", default="cpu,cuda,mps")
    parser.add_argument("--kinds", default="grayscale,rgb,stack")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stack-depth", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("noise2fast-matrix.json"))
    args = parser.parse_args()

    records = []
    benchmark = Path(__file__).with_name("benchmark.py")
    for size in map(int, args.sizes.split(",")):
        for effort in args.efforts.split(","):
            for device in args.devices.split(","):
                for kind in args.kinds.split(","):
                    command = [
                        sys.executable,
                        str(benchmark),
                        "--method",
                        "noise2fast",
                        "--effort",
                        effort,
                        "--device",
                        device,
                        "--size",
                        str(size),
                        "--kind",
                        kind,
                        "--stack-depth",
                        str(args.stack_depth),
                    ]
                    if args.steps is not None:
                        command.extend(("--steps", str(args.steps)))
                    started = time.perf_counter()
                    completed = subprocess.run(command, capture_output=True, text=True)
                    process_cold_seconds = time.perf_counter() - started
                    if completed.returncode:
                        records.append(
                            {
                                "size": size,
                                "effort": effort,
                                "device": device,
                                "kind": kind,
                                "available": False,
                                "error": completed.stderr.strip()[-1000:],
                            }
                        )
                        continue
                    record = json.loads(completed.stdout)
                    record["process_wall_seconds_including_cold_and_warm"] = process_cold_seconds
                    records.append(record)
                    args.output.write_text(json.dumps(records, indent=2) + "\n")
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
