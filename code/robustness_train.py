"""GPU robustness & generality training experiments:
  --mode softpen   : Unconstrained backbone + soft aggregate penalty gamma * L_viol(q)
  --mode backbone  : ResNet1D / BiGRU under posthoc vs. end-to-end GRANITE coherence layer
  --mode horizon   : CN31 training-horizon efficiency (2yr / 4yr history) for GRANITE & Post-hoc

Follows train.py protocol exactly (same splits, DS_DEFAULTS, AdamW, OneCycleLR, val selection).
"""
import argparse, json, os, time
import numpy as np
import torch

from backbones import BiGRUDisaggregator, ResNet1DDisaggregator
from data import CN31, SDPrice
from granite import Granite, collate, count_params, loss_fn, sum_target, violation
from train import DS_DEFAULTS, RES, eval_specs, predict


def soft_viol_loss(q, bt, geometry):
    """Differentiable quadratic penalty on aggregate view residuals."""
    total = 0.0
    for v in bt['views']:
        g = v['gid'].reshape(-1)
        sel = g >= 0
        idx = g[sel]
        G = v['val'].numel()
        s = torch.zeros(G, device=q.device, dtype=q.dtype).index_add(0, idx, q.reshape(-1)[sel])
        t = sum_target(v)
        if geometry == 'kl':
            den = t.abs().mean().clamp_min(1e-6)
        else:
            den = (v['cnt'] * (t / v['cnt'].clamp_min(1)).abs().std().clamp_min(1.0))
        total = total + (((s - t) / den) ** 2).mean()
    return total / len(bt['views'])


def run_job(a):
    for k, v in DS_DEFAULTS[a.ds].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = 'cuda'
    ds = CN31() if a.ds == 'cn31' else SDPrice()
    val_specs = eval_specs(ds, 'val')
    test_specs = eval_specs(ds, 'test')

    if a.mode == 'softpen':
        model = Granite(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry,
                        P=ds.P, d=a.d, layers=a.layers, iters_train=8, iters_eval=a.iters_eval,
                        dropout=a.dropout, coherence='none').to(dev)
        name = f"rob_softpen_{a.ds}_g{a.gamma:g}_s{a.seed}"
        project_eval = False
        lam = 0.0
    elif a.mode == 'backbone':
        cls = ResNet1DDisaggregator if a.arch == 'resnet1d' else BiGRUDisaggregator
        coh = 'posthoc' if a.regime == 'posthoc' else 'end2end'
        model = cls(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry,
                    hid=96, dropout=a.dropout, coherence=coh, iters_train=8, iters_eval=a.iters_eval).to(dev)
        name = f"rob_bb_{a.arch}_{a.regime}_{a.ds}_s{a.seed}"
        project_eval = True
        lam = 1.0 if a.regime == 'granite' else 0.0
    elif a.mode == 'horizon':
        coh = 'posthoc' if a.regime == 'posthoc' else 'end2end'
        model = Granite(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry,
                        P=ds.P, d=a.d, layers=a.layers, iters_train=8, iters_eval=a.iters_eval,
                        dropout=a.dropout, coherence=coh).to(dev)
        name = f"rob_hor_{a.years}y_{a.regime}_{a.ds}_s{a.seed}"
        project_eval = True
        lam = 1.0 if a.regime == 'granite' else 0.0
        # Restrict training range to the last `a.years` years of training (ending 2021-12-31)
        tr_a, tr_b = ds.split_range('train')
        cut = max(tr_a, tr_b - a.years * 8760)
        ds.train_specs = lambda r, n, L=744, _a=cut, _b=tr_b: [
            (s, s + L) for s in r.integers(_a // 24, (_b - L) // 24, n) * 24
        ]
    else:
        raise ValueError(a.mode)

    out_path = os.path.join(RES, name + '.json')
    if os.path.exists(out_path) and not a.force:
        print("Skipping existing", name)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps, pct_start=0.05)

    best, best_state, log = 1e9, None, []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        specs = ds.train_specs(rng, 8, L=744)
        bt = collate([ds.block(s) for s in specs], dev)
        pred = model(bt)
        loss = loss_fn(pred, bt, ds.geometry)
        if a.mode == 'softpen':
            loss = loss + a.gamma * soft_viol_loss(pred, bt, ds.geometry)
        elif lam > 0:
            loss = loss + lam * loss_fn(model._seed, bt, ds.geometry)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            sc = predict(model, ds, val_specs, a.ebs, dev, project_eval, keep=False)[0]
            log.append(dict(step=step, loss=loss.item(), **sc, t=time.time() - t0))
            if sc['wape'] < best:
                best = sc['wape']
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    sc, _ = predict(model, ds, test_specs, a.ebs, dev, project_eval, keep=False)
    extra = {}
    if a.mode == 'softpen':
        # Also evaluate post-hoc projected version of the soft-penalty network
        model.coherence = 'posthoc'
        sc_proj, _ = predict(model, ds, test_specs, a.ebs, dev, True, keep=False)
        extra['test_projected'] = sc_proj

    res = dict(name=name, args=vars(a), test=sc, **extra, params=count_params(model),
               train_time=train_time, n_test_cells=sc['n'], log=log)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    print('DONE', name, json.dumps(sc), f"({train_time:.0f}s)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['softpen', 'backbone', 'horizon'])
    ap.add_argument('--ds', default='cn31', choices=['cn31', 'sd'])
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--arch', default='resnet1d', choices=['resnet1d', 'bigru'])
    ap.add_argument('--regime', default='granite', choices=['granite', 'posthoc'])
    ap.add_argument('--years', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--eval_every', type=int, default=None)
    ap.add_argument('--d', type=int, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--dropout', type=float, default=None)
    ap.add_argument('--wd', type=float, default=None)
    ap.add_argument('--ebs', type=int, default=None)
    ap.add_argument('--iters_eval', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    run_job(ap.parse_args())
