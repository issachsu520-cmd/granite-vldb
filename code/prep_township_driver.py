# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Run prep_township.py one month per process until it reports completion."""
import os, subprocess, sys
env = dict(os.environ, ONE_MONTH='1')
while True:
    r = subprocess.run([sys.executable, '-u', 'prep_township.py'], cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
    if r.returncode != 3:
        sys.exit(r.returncode)
