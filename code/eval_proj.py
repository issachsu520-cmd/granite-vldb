"""Evaluate a trained N^2-Norm (single-view) model with the full multi-view projection at test time.
By gauge invariance, projecting its single-view output onto all views equals projecting its seed.
usage: python eval_proj.py cn31_n2_s0      (writes <ds>_n2proj_s<seed>.json; TS-GHI: full test)"""
import json, os, sys, time
import torch
from granite import Granite
from train import get_ds, eval_specs, predict, save_preds, VARIANTS, RES

name = sys.argv[1]
meta = json.load(open(os.path.join(RES, name + '.json')))
a = meta['args']
ds = get_ds(a['ds'], tuple(a['views'].split(',')) if a['views'] else None)
model = Granite(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry, P=ds.P, d=a['d'], layers=a['layers'],
                iters_train=a['iters'], iters_eval=a.get('iters_eval') or 100, dropout=a['dropout'], **VARIANTS[a['variant']]).to('cuda')
model.load_state_dict(torch.load(os.path.join(RES, name + '.pt')))
model.single_view = None  # project onto every view instead of normalizing one
is_ghi = a['ds'] == 'ghi'
specs = eval_specs(ds, 'test', None)
t = time.time()
sc, preds = predict(model, ds, specs, a['ebs'], 'cuda', True, keep=not is_ghi)
out = name.replace('_n2_', '_n2proj_')
if not is_ghi:
    save_preds(os.path.join(RES, out + '_pred.npy'), preds)
json.dump(dict(name=out, args=a, test=sc, infer_time=time.time() - t), open(os.path.join(RES, out + '.json'), 'w'), indent=1)
print('TEST', out, json.dumps(sc))
