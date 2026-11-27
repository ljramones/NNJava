"""Time the packaged Java session on frozen token IDs and check each retained vector against independent references."""
import argparse
from array import array
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

MODULE=Path(__file__).resolve().parents[1]


def logits(path):
    raw=path.read_bytes()
    if len(raw)!=128256*4: raise ValueError("Invalid vocabulary vector "+str(path))
    z=array('f')
    z.frombytes(raw)
    if sys.byteorder!='little': z.byteswap()
    if not all(math.isfinite(x) for x in z): raise ValueError("Nonfinite logits "+str(path))
    return z


def compare(a,b):
    err=[float(y)-x for x,y in zip(a,b,strict=True)]
    rms=math.sqrt(math.fsum(x*x for x in err)/len(err))
    ma,mb=max(a),max(b)
    lza=ma+math.log(math.fsum(math.exp(x-ma) for x in a))
    lzb=mb+math.log(math.fsum(math.exp(x-mb) for x in b))
    kl=math.fsum(math.exp(x-lza)*(x-y+lzb-lza) for x,y in zip(a,b,strict=True))
    ta=sorted(range(len(a)),key=a.__getitem__,reverse=True)[:10]
    tb=sorted(range(len(b)),key=b.__getitem__,reverse=True)[:10]
    ab=max(map(abs,err))
    return dict(max_abs=ab,rms=rms,kl_nats=kl,top1_match=ta[0]==tb[0],top10_overlap=len(set(ta)&set(tb)),
                passed=ab<=1 and rms<=.15 and kl<=.01 and ta[0] in tb[:5] and tb[0] in ta[:5])


def score_capture(a):
    report={}
    required={a.depth-1,a.depth,a.depth+63,a.depth+64,a.depth+127}
    extra={0,63,64,511} if a.depth==4096 else set()
    for pos in sorted(required|extra):
        name=f'logits-{pos}.f32'
        data=(a.out/name).read_bytes()
        row=dict(result_sha256=sha256(data).hexdigest())
        if pos in required or (a.reference/name).is_file():
            row.update(compare(logits(a.reference/name),logits(a.out/name)))
        if a.runner_reference and (pos in required or (a.runner_reference/name).is_file()):
            row['runner_bit_identical']=data==(a.runner_reference/name).read_bytes()
        if len(row)>1:
            report[str(pos)]=row
    (a.out/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(not (r.get('passed',True) and r.get('runner_bit_identical',True)) for r in report.values()):
        raise ValueError('Public-session logit gate failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokens',type=Path,required=True)
    p.add_argument('--depth',type=int,choices=(4096,32768),required=True)
    p.add_argument('--precision',choices=('packed','direct'),required=True)
    p.add_argument('--reference',type=Path,required=True,help='llama.cpp logits-{position}.f32')
    p.add_argument('--runner-reference',type=Path,help='Same precision prior experimental NNJava capture; checks exact bytes')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--seconds',type=float,default=0)
    a=p.parse_args()
    os.fstat(int(os.environ['DYNAMISGPU_BENCHMARK_LOCK_FD']))
    cpfile=MODULE/'target/runtime-cp.txt'
    jars=list((MODULE/'target').glob('nnjava-gguf-metal-*.jar'))
    if len(jars)!=1 or not cpfile.is_file(): raise ValueError('Build public module and generate target/runtime-cp.txt')
    a.out.mkdir(parents=True,exist_ok=False)
    cls=a.out/'example-classes'
    cls.mkdir()
    cp=os.pathsep.join([str(jars[0]),cpfile.read_text().strip()])
    subprocess.run(['javac','--release','25','-cp',cp,'-d',cls,MODULE/'examples/CheckpointRun.java'],check=True)
    command=['java','--enable-native-access=ALL-UNNAMED','-Xmx10g','-cp',str(cls)+os.pathsep+cp,'CheckpointRun',
                    str(a.model.resolve()),str(a.tokens.resolve()),str(a.depth),str(a.cycles),str(a.seconds),a.precision,str(a.out)]
    with a.model.open('rb') as f:
        from hashlib import file_digest
        modelhash=file_digest(f,'sha256').hexdigest()
    if modelhash!='667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29': raise ValueError('Unexpected fixture checkpoint')
    metadata=dict(model_sha256=modelhash,token_sha256=sha256(a.tokens.read_bytes()).hexdigest(),argv=command,
                  jar_sha256=sha256(jars[0].read_bytes()).hexdigest(),
                  runtime_jars={Path(p).name:sha256(Path(p).read_bytes()).hexdigest() for p in cpfile.read_text().strip().split(os.pathsep)},
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MODULE,text=True).strip())
    (a.out/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (a.out/'process.log').open('w') as log:
        env=dict(os.environ,NNJAVA_EXPECT_HOST_GATE='1')
        with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env) as process:
            try:
                marker=process.stdout.readline().strip()
                if marker!='CAPTURE_READY': raise ValueError('Missing public capture completion: '+marker)
                score_capture(a)
                process.stdin.write('PASS\n')
                process.stdin.flush()
                for line in process.stdout: log.write(line)
            finally:
                process.stdin.close()
            if process.wait()!=0: raise ValueError('Public consumer failed; see process.log')
    if (a.out/'completed').read_text().strip()!='ok':raise ValueError('Incomplete public run')
    with (a.out/'trajectories.csv').open() as f: tr=list(csv.DictReader(f))
    count=sum(int(r['tokens']) for r in tr)
    ns=sum(int(r['wall_ns']) for r in tr)
    summary=dict(cycles=len(tr),tokens=count,mean_ms=ns/(count*1e6),tok_s=count*1e9/ns)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)


if __name__=='__main__':main()
