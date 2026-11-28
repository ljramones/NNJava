#!/usr/bin/env python3
"""Run packaged JPMS consumers after mvn install and consumer compilation."""
import argparse
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-metal", action="store_true")
parser.add_argument("--repository", type=Path, default=Path.home() / ".m2/repository")
parser.add_argument("--pretrained-dir", type=Path, help="Opt-in pinned SmolLM2 assets plus independently generated reference")
parser.add_argument("--pretrained-report", type=Path)
parser.add_argument("--simd", action="store_true", help="Use installed SIMD Vectrix artifacts and enable the vector module")
args = parser.parse_args()
if args.pretrained_report and args.pretrained_dir is None:
    parser.error("--pretrained-report requires --pretrained-dir")
root = Path(__file__).resolve().parents[1]
repository = args.repository / "org/dynamisengine"

def artifact(name, version, classifier=""):
    suffix = f"-{classifier}" if classifier else ""
    path = repository / name / version / f"{name}-{version}{suffix}.jar"
    if not path.is_file():
        raise SystemExit(f"Missing installed artifact: {path}")
    return path

def run(module, paths, *arguments, native=False):
    command = ["java"]
    if args.simd:
        command += ["--add-modules", "jdk.incubator.vector"]
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

numerics = [artifact(name, "4.1.0", "simd" if args.simd else "") for name in ("vectrix-core", "vectrix-ml")]
jackson_version = ET.parse(root / "nnjava-bom/pom.xml").getroot().findtext(
    "{http://maven.apache.org/POM/4.0.0}properties/{http://maven.apache.org/POM/4.0.0}jackson.version")
if not jackson_version:
    raise SystemExit("The BOM must define jackson.version")
formats = [artifact("nnjava-formats", "0.1.0-SNAPSHOT"),
           args.repository / f"com/fasterxml/jackson/core/jackson-core/{jackson_version}/jackson-core-{jackson_version}.jar"]
run("example.nnjava.formats/example.FormatsConsumer",
    [root / "examples/formats-consumer/target/classes", *formats])
training = [artifact(name, "0.1.0-SNAPSHOT") for name in
            ("neuralnetworks", "nnjava", "nnjava-core", "nnjava-nn", "nnjava-transformer",
             "nnjava-data", "nnjava-io", "nnjava-training", "nnjava-generation", "nnjava-backend-vectrix")]
run("example.nnjava.training/example.LibraryConsumer",
    [root / "examples/consumer/target/classes", *training, *formats, *numerics])
inference = [artifact(name, "0.1.0-SNAPSHOT") for name in
             ("nnjava-core", "nnjava-nn", "nnjava-transformer", "nnjava-io", "nnjava-backend-vectrix")]
run("example.nnjava.inference/example.InferenceConsumer",
    [root / "examples/inference-consumer/target/classes", *inference, *formats, *numerics])
generation = [artifact(name, "0.1.0-SNAPSHOT") for name in
              ("nnjava-generation", "nnjava-core", "nnjava-nn", "nnjava-transformer", "nnjava-backend-vectrix")]
run("example.nnjava.generation/example.GenerationConsumer",
    [root / "examples/generation-consumer/target/classes", *generation, *numerics])
run("example.nnjava.tokenizers/example.TokenizerConsumer",
    [root / "examples/tokenizer-consumer/target/classes", artifact("nnjava-tokenizers", "0.1.0-SNAPSHOT"), *formats])
pretrained = [artifact(name, "0.1.0-SNAPSHOT") for name in ("nnjava-pretrained", "nnjava-tokenizers", "nnjava-io")]
pretrained_arguments = ["--self-test"] if args.pretrained_dir is None else [str(args.pretrained_dir.resolve())]
if args.pretrained_report:
    pretrained_arguments.append(str(args.pretrained_report.resolve()))
run("example.nnjava.pretrained/example.PretrainedConsumer",
    [root / "examples/pretrained-consumer/target/classes", *pretrained, *generation, *formats, *numerics], *pretrained_arguments)
if args.gpu_metal:
    gpu = [artifact(name, "0.1.0-SNAPSHOT") for name in ("nnjava-core", "nnjava-backend-vectrix-gpu")]
    gpu += [artifact("vectrix-gpu", "4.1.0"), *numerics]
    gpu += [artifact(name, "1.14.1") for name in ("dynamis-gpu-api", "dynamis-gpu-primitives", "dynamis-gpu-metal")]
    for provider in ("vectrix-gpu-exact", "dynamis-gpu-throughput"):
        run("example.nnjava.gpu/example.GpuConsumer",
            [root / "examples/gpu-consumer/target/classes", *gpu], provider, native=True)
