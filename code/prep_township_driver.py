"""Run prep_township.py one month per process until it reports completion."""
import os, subprocess, sys
env = dict(os.environ, ONE_MONTH='1')
while True:
    r = subprocess.run([sys.executable, '-u', 'prep_township.py'], cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
    if r.returncode != 3:
        sys.exit(r.returncode)
