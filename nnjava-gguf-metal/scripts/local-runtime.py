"""Install the three built GPU artifacts under distinct experiment coordinates in a private Maven repository."""
import argparse
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from hashlib import sha256
import json
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('gpu',type=Path)
p.add_argument('repository',type=Path)
a=p.parse_args()
repo=a.repository.resolve()
if repo==Path.home()/'.m2/repository':
    raise ValueError("Use a private Maven repository for the experimental runtime")
pom=Path(__file__).resolve().parents[1]/'pom.xml'
version=ET.parse(pom).findtext('{*}properties/{*}dynamisgpu.version')
artifacts={}
with TemporaryDirectory() as t:
    for short in ('api','primitives','metal'):
        name='dynamis-gpu-'+short
        classes=a.gpu.resolve()/name/'target/classes'
        if not (classes/'module-info.class').is_file():
            raise ValueError('Build the GPU checkout first: '+str(classes))
        if short=='metal' and not list((classes/'native/macos-arm64').glob('*.dylib')):
            raise ValueError('Build the native Metal bridge first')
        file=Path(t)/(name+'.jar')
        run(['jar','--create','--file',file,'-C',classes,'.'],check=True)
        run(['mvn','-o','-q','-f',pom,'-Dmaven.repo.local='+str(repo),
             'org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file',
             '-DgroupId=org.dynamisengine','-DartifactId='+name,'-Dversion='+version,
             '-Dpackaging=jar','-DgeneratePom=true','-Dfile='+str(file)],check=True)
        artifacts[name]=sha256(file.read_bytes()).hexdigest()
print(json.dumps(artifacts,indent=2))
