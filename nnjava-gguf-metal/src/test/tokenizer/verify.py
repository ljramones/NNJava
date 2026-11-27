"""Build and compare the Java tokenizer with a pinned llama.cpp CPU-only oracle on macOS."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA="667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
LIBLLAMA_SHA="aef3e6fe2a806c121470e0dfcf98107e8c8f17dea601008bb204619dc7a4e8a1"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("checkpoint",type=Path);p.add_argument("classpath")
    p.add_argument("--receipt",type=Path);a=p.parse_args()
    if sha(a.checkpoint)!=CHECKPOINT_SHA or sha(Path("/opt/homebrew/opt/llama.cpp/lib/libllama.dylib"))!=LIBLLAMA_SHA:
        p.error("the pinned checkpoint and libllama are required")
    with TemporaryDirectory() as tmp:
        work=Path(tmp)
        subprocess.run([sys.executable,"-B",HERE/"cases.py",work/"inputs"],check=True)
        subprocess.run(["clang++","-std=c++17","-O2","-Wall","-Wextra","-Werror","-I/opt/homebrew/include","-L/opt/homebrew/lib",
                        "-Wl,-rpath,/opt/homebrew/lib",HERE/"oracle.cpp","-lllama","-lggml","-lggml-base","-o",work/"oracle"],check=True)
        with (work/"native.log").open("wb") as log:
            subprocess.run([work/"oracle",a.checkpoint,work/"inputs",work],check=True,stderr=log)
        (work/"pieces.sha256").write_text("".join(sha(work/f"pieces{i}.bin")+"\n" for i in (0,1)))
        result=subprocess.check_output(["java","-cp",a.classpath,"org.dynamisengine.nn.inference.llama31metal.TokenizerParity",
                                        a.checkpoint,work],text=True).strip()
        receipt=dict(result=result,checkpoint_sha256=CHECKPOINT_SHA,libllama_sha256=LIBLLAMA_SHA,
                     oracle_rows_sha256=sha(work/"oracle.tsv"),inputs_sha256=sha(work/"inputs"),
                     eog=(work/"eog.txt").read_text().splitlines(),token_pieces_sha256=(work/"pieces.sha256").read_text().splitlines())
        if a.receipt:a.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
        print(result)
