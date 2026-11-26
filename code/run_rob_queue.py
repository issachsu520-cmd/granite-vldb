# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Resumable parallel queue for robustness_train.py jobs."""
import os, subprocess, sys, threading, time

from paths import RES
CODE = os.path.dirname(os.path.abspath(__file__))


def out_name(args):
    a = dict(zip(args[::2], args[1::2]))
    m = a['--mode']
    if m == 'softpen':
        g = f"{float(a['--gamma']):g}"
        return f"rob_softpen_{a['--ds']}_g{g}_s{a['--seed']}.json"
    if m == 'backbone':
        return f"rob_bb_{a['--arch']}_{a['--regime']}_{a['--ds']}_s{a['--seed']}.json"
    if m == 'horizon':
        return f"rob_hor_{a['--years']}y_{a['--regime']}_{a['--ds']}_s{a['--seed']}.json"
    raise ValueError(m)


def worker(q, lock):
    while True:
        with lock:
            if not q:
                return
            line = q.pop(0)
        args = line.split()
        out = os.path.join(RES, out_name(args))
        if os.path.exists(out):
            continue
        log = os.path.join(RES, 'logs', os.path.basename(out).replace('.json', '.log'))
        t = time.time()
        with open(log, 'w') as f:
            r = subprocess.run([sys.executable, '-u', 'robustness_train.py'] + args,
                               cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
        st = 'done' if r.returncode == 0 else f'FAILED ({r.returncode})'
        print(f"{time.strftime('%H:%M:%S')} {st} {os.path.basename(out)} in {time.time() - t:.0f}s", flush=True)


if __name__ == '__main__':
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    os.makedirs(os.path.join(RES, 'logs'), exist_ok=True)
    lock = threading.Lock()
    ths = [threading.Thread(target=worker, args=(jobs, lock)) for _ in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    print('all done', flush=True)
