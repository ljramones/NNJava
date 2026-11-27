#!/usr/bin/env python3
"""Run packaged JPMS consumers after mvn install and consumer compilation."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-metal", action="store_true")
parser.add_argument("--repository", type=Path, default=Path.home() / ".m2/repository")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
repository = args.repository / "org/dynamisengine"

def artifact(name, version):
    path = repository / name / version / f"{name}-{version}.jar"
    if not path.is_file():
        raise SystemExit(f"Missing installed artifact: {path}")
    return path

def run(module, paths, *arguments, native=False):
    command = ["java"]
    if native:
        command += ["--enable-native-access=org.dynamisengine.gpu.metal"]
        # Published shader packages are encapsulated; KernelLibrary reads them
        # from dynamis-gpu-api. Open only these resource packages to that module.
        for shader_module, package in (
                ("org.dynamisengine.vectrix.compute", "org.dynamisengine.vectrix.compute.kernels"),
                ("org.dynamisengine.gpu.primitives", "org.dynamisengine.gpu.primitives.kernels"),
                ("org.dynamisengine.gpu.primitives", "org.dynamisengine.gpu.primitives.metal")):
            command += ["--add-opens", f"{shader_module}/{package}=org.dynamisengine.gpu.api"]
    command += ["--module-path", os.pathsep.join(map(str, paths)), "-m", module, *arguments]
    subprocess.run(command, check=True, cwd=root)

numerics = [artifact(name, "4.1.0") for name in ("vectrix-core", "vectrix-ml")]
training = [artifact(name, "0.1.0-SNAPSHOT") for name in
            ("neuralnetworks", "nnjava", "nnjava-core", "nnjava-nn", "nnjava-transformer",
             "nnjava-data", "nnjava-io", "nnjava-training", "nnjava-backend-vectrix")]
run("example.nnjava.training/example.LibraryConsumer",
    [root / "examples/consumer/target/classes", *training, *numerics])
inference = [artifact(name, "0.1.0-SNAPSHOT") for name in
             ("nnjava-core", "nnjava-nn", "nnjava-transformer", "nnjava-io", "nnjava-backend-vectrix")]
run("example.nnjava.inference/example.InferenceConsumer",
    [root / "examples/inference-consumer/target/classes", *inference, *numerics])
if args.gpu_metal:
    gpu = [artifact(name, "0.1.0-SNAPSHOT") for name in ("nnjava-core", "nnjava-backend-vectrix-gpu")]
    gpu += [artifact("vectrix-gpu", "4.1.0"), *numerics]
    gpu += [artifact(name, "1.14.1") for name in ("dynamis-gpu-api", "dynamis-gpu-primitives", "dynamis-gpu-metal")]
    for provider in ("vectrix-gpu-exact", "dynamis-gpu-throughput"):
        run("example.nnjava.gpu/example.GpuConsumer",
            [root / "examples/gpu-consumer/target/classes", *gpu], provider, native=True)
