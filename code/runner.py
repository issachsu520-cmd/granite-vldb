"""Resumable job queue: runs every line of a jobs file with N parallel workers,
skipping jobs whose result JSON already exists.  usage: python runner.py jobs.txt 2"""
import os, subprocess, sys, time, threading

from paths import RES
CODE = os.path.dirname(os.path.abspath(__file__))


def worker(q, lock):
    while True:
        with lock:
            if not q:
                return
            line = q.pop(0)
        parts = line.split()
        script, args = parts[0], parts[1:]
        out = os.path.join(RES, result_name_from(script, args))
        if os.path.exists(out):
            continue
        log = os.path.join(RES, 'logs', os.path.basename(out).replace('.json', '.log'))
        t = time.time()
        with open(log, 'w') as f:
            r = subprocess.run([sys.executable, '-u', script] + args, cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
        status = 'done' if r.returncode == 0 else f'FAILED (exit {r.returncode})'
        print(f"{time.strftime('%H:%M:%S')} {status} {os.path.basename(out)} in {time.time() - t:.0f}s", flush=True)


def result_name_from(script, args):
    if script == 'eval_proj.py':  # args: <N^2-Norm run name>
        return args[0].replace('_n2_', '_n2proj_') + '.json'
    if script == 'eval_sub.py':  # args: <run name> <n counties>
        n, nm = args[1], args[0]
        return nm[:nm.rindex('_s')] + f"_c{n}" + nm[nm.rindex('_s'):] + '.json'
    a = dict(zip(args[::2], args[1::2]))
    v = a.get('--views')
    vs = ('_' + v.replace(',', '-')) if v else ''
    if script == 'baselines.py':
        tc = a.get('--test_counties')
        sp = a.get('--split', 'test')
        return f"{a['--ds']}_{a['--method']}{vs}{'' if sp == 'test' else '_' + sp}{('_c' + tc) if tc and tc != '0' else ''}_s0.json"
    return f"{a['--ds']}_{a.get('--variant', 'granite')}{vs}{a.get('--tag', '')}_s{a.get('--seed', '0')}.json"


if __name__ == '__main__':
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    os.makedirs(os.path.join(RES, 'logs'), exist_ok=True)
    lock = threading.Lock()
    ths = [threading.Thread(target=worker, args=(jobs, lock)) for _ in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print('all done', flush=True)
