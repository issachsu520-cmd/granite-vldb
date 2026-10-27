"""Train / evaluate GRANITE and its neural variants on one dataset.

usage: python train.py --ds cn31 --variant granite --steps 3000 --seed 0
"""
import argparse, json, os, time
import numpy as np, torch
from data import CN31, SDPrice, TSGHI
from granite import Granite, collate, loss_fn, violation, count_params

from paths import RES
VARIANTS = {
    'granite': {},
    'e2e': {},                                # projection-only loss (no seed loss), ablation
    'posthoc': dict(coherence='posthoc'),     # train unconstrained, project at test time
    'none': dict(coherence='none'),           # unconstrained network
    'n2': dict(single_view=0),                # UrbanFM-style N^2 normalisation on one view
    'nochmp': dict(use_chmp=False),
    'noent': dict(entity=False),
    'notemp': dict(temporal=False),
}


DS_DEFAULTS = {  # shared by every neural variant on a dataset
    'cn31': dict(steps=3000, eval_every=250, d=96, layers=3, dropout=0.1, wd=1e-4, ebs=8, iters_eval=100),
    'sd': dict(steps=1500, eval_every=100, d=96, layers=3, dropout=0.3, wd=1e-2, ebs=8, iters_eval=100),
    'ghi': dict(steps=2500, eval_every=500, d=96, layers=3, dropout=0.1, wd=1e-4, ebs=48, iters_eval=30),
}


def get_ds(name, views=None):
    kw = {'views': views} if views else {}
    return {'cn31': CN31, 'sd': SDPrice, 'ghi': TSGHI}[name](**kw)


def ghi_counties(ds, frac=1.0):
    nc = len(ds.counties)
    perm = np.random.default_rng(123).permutation(nc)
    return perm[:int(nc * frac)]


def eval_specs(ds, split, n_counties=None):
    if isinstance(ds, TSGHI):
        nc = len(ds.counties)
        cs = np.random.default_rng(7 if split == 'val' else 11).choice(nc, n_counties, replace=False) if n_counties else None
        return ds.eval_specs(split, counties=cs)
    return ds.eval_specs(split)


class Scorer:
    """Streaming WAPE / RMSE / max-violation accumulator."""

    def __init__(self):
        self.num = self.den = self.se = 0.0
        self.n = 0
        self.viol = 0.0

    def add(self, x, y, viol):
        ok = ~np.isnan(y)
        d = x[ok].astype(np.float64) - y[ok]
        self.num += np.abs(d).sum(); self.den += np.abs(y[ok]).sum(); self.se += (d ** 2).sum(); self.n += int(ok.sum())
        self.viol = max(self.viol, viol)

    def result(self):
        return dict(wape=float(self.num / self.den), rmse=float(np.sqrt(self.se / self.n)), viol=float(self.viol), n=self.n)


@torch.no_grad()
def predict(model, ds, specs, bs, dev, project=True, keep=True):
    model.eval()
    sc, out = Scorer(), [None] * len(specs)
    order = block_order(ds, specs)
    for i in range(0, len(order), bs):
        ids = order[i:i + bs]
        sp = [specs[k] for k in ids]
        blocks = [ds.block(s) for s in sp]
        bt = collate(blocks, dev)
        x = model(bt, project=project)
        viol = violation(x, bt['views'])
        x = x.float().cpu().numpy()
        for j, b in enumerate(blocks):
            e, l = b['y'].shape
            sc.add(x[j, :e, :l], b['y'], viol)
            if keep:
                out[ids[j]] = dict(spec=sp[j], x=x[j, :e, :l])
    return sc.result(), (out if keep else [])


def block_order(ds, specs):
    """Process TS-GHI blocks sorted by county size to minimise padding; other datasets keep their order."""
    if isinstance(ds, TSGHI):
        return sorted(range(len(specs)), key=lambda k: (len(ds.counties[specs[k][0]]), k))
    return list(range(len(specs)))


def save_preds(path, preds):
    arr = np.empty(len(preds), dtype=object)
    for i, p in enumerate(preds):
        arr[i] = p['x'].astype(np.float32)
    np.save(path, arr, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='cn31')
    ap.add_argument('--variant', default='granite')
    ap.add_argument('--views', default=None)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--ebs', type=int, default=None)
    ap.add_argument('--iters_eval', type=int, default=None)
    ap.add_argument('--L', type=int, default=744)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d', type=int, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dropout', type=float, default=None)
    ap.add_argument('--wd', type=float, default=None)
    ap.add_argument('--lam', type=float, default=1.0)   # weight of the gauge-fixing seed loss
    ap.add_argument('--eval_every', type=int, default=None)
    ap.add_argument('--train_frac', type=float, default=1.0)   # TS-GHI: share of counties used for training
    ap.add_argument('--val_counties', type=int, default=120)
    ap.add_argument('--test_counties', type=int, default=0)    # 0 = all counties
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    for k, v in DS_DEFAULTS[a.ds].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = 'cuda'
    views = tuple(a.views.split(',')) if a.views else None
    ds = get_ds(a.ds, views)
    model = Granite(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry, P=ds.P, d=a.d,
                    layers=a.layers, iters_train=a.iters, iters_eval=a.iters_eval, dropout=a.dropout, **VARIANTS[a.variant]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    name = f"{a.ds}_{a.variant}{('_' + a.views.replace(',', '-')) if a.views else ''}{a.tag}_s{a.seed}"
    os.makedirs(RES, exist_ok=True)
    is_ghi = isinstance(ds, TSGHI)
    tr_counties = ghi_counties(ds, a.train_frac) if is_ghi else None
    val_specs = eval_specs(ds, 'val', a.val_counties if is_ghi else None)
    project_eval = a.variant != 'none'
    lam = a.lam if a.variant in ('granite', 'nochmp', 'noent', 'notemp') else 0.0
    best, best_state, log = 1e9, None, []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        specs = ds.train_specs(rng, a.bs, L=a.L, counties=tr_counties) if is_ghi else ds.train_specs(rng, a.bs, L=a.L)
        bt = collate([ds.block(s) for s in specs], dev)
        loss = loss_fn(model(bt), bt, ds.geometry)
        if lam > 0:
            loss = loss + lam * loss_fn(model._seed, bt, ds.geometry)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            sc = predict(model, ds, val_specs, a.ebs, dev, project_eval, keep=False)[0]
            log.append(dict(step=step, loss=loss.item(), **sc, t=time.time() - t0))
            print(json.dumps(log[-1]), flush=True)
            if sc['wape'] < best:
                best = sc['wape']
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RES, name + '.pt'))
    test_specs = eval_specs(ds, 'test', a.test_counties if is_ghi else None)
    torch.cuda.synchronize(); t1 = time.time()
    keep = not is_ghi or a.test_counties > 0
    sc, preds = predict(model, ds, test_specs, a.ebs, dev, project_eval, keep=keep)
    torch.cuda.synchronize(); infer_time = time.time() - t1
    print('TEST', name, json.dumps(sc), f"train {train_time:.0f}s infer {infer_time:.1f}s", flush=True)
    if keep:
        save_preds(os.path.join(RES, name + '_pred.npy'), preds)
    json.dump(dict(name=name, args=vars(a), test=sc, params=count_params(model), train_time=train_time,
                   infer_time=infer_time, n_test_cells=sc['n'], log=log),
              open(os.path.join(RES, name + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
