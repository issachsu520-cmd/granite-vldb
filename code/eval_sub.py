"""Re-evaluate a trained TS-GHI model on a fixed subset of counties and save predictions
(used for downstream queries and maps).  usage: python eval_sub.py ghi_granite_s0 300"""
import json, os, sys, time
import numpy as np, torch
from granite import Granite
from train import get_ds, eval_specs, predict, save_preds, VARIANTS, DS_DEFAULTS, RES

name, n = sys.argv[1], int(sys.argv[2])
meta = json.load(open(os.path.join(RES, name + '.json')))
a = meta['args']
ds = get_ds(a['ds'], tuple(a['views'].split(',')) if a['views'] else None)
model = Granite(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry, P=ds.P, d=a['d'], layers=a['layers'],
                iters_train=a['iters'], iters_eval=a.get('iters_eval') or 100, dropout=a['dropout'], **VARIANTS[a['variant']]).to('cuda')
model.load_state_dict(torch.load(os.path.join(RES, name + '.pt')))
specs = eval_specs(ds, 'test', n or None)
t = time.time()
sc, preds = predict(model, ds, specs, a['ebs'], 'cuda', a['variant'] != 'none', keep=n > 0)
out = name.replace(f"_s{a['seed']}", f"_c{n}_s{a['seed']}" if n else f"_full_s{a['seed']}")
if n:
    save_preds(os.path.join(RES, out + '_pred.npy'), preds)
json.dump(dict(name=out, test=sc, infer_time=time.time() - t), open(os.path.join(RES, out + '.json'), 'w'), indent=1)
print('TEST', out, json.dumps(sc))
