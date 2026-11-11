"""Improvement arms on the published CNN. Usage: train.py {pilot,train} --arm {R_rank,D_adversarial,C0_cycle,C1_cycle} [--folds 0,1,2,3,4]

Reuses the frozen CNN code (model.py, windows.py, run.py) unchanged. Only the data selection, the extra loss terms and
the input version differ by arm, as fixed in protocol.json.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE / 'CNN'))
import run as base
from model import GdiCNN, channels
from windows import WindowSet, weighted_mse

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['pilot', 'train'])
ap.add_argument('--arm', required=True, choices=['R_rank', 'D_adversarial', 'C0_cycle', 'C1_cycle'])
ap.add_argument('--folds', default='0,1,2,3,4')
ap.add_argument('--device', default='cuda')
a = ap.parse_args()
DATA = HERE / 'data'
OUT = HERE / 'runs' / a.arm
CYCLE = a.arm.startswith('C')
USE_LJMU = a.arm in ('D_adversarial', 'C1_cycle')
MARGIN, RANK_W, ADV_MAX = 20.0, 1.0, 0.3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


prep = read(DATA / 'checksums.json')
for name, digest in prep.items():
    base.require(sha(DATA / name) == digest, 'Data changed: ' + name)
with np.load(DATA / ('data-cycle.npz' if CYCLE else 'data.npz')) as z:
    d = {k: z[k] for k in z.files}
keep_rec = (d['source'] == 'Stanford') | (USE_LJMU & (d['source'] == 'LJMU'))
rec_idx = np.flatnonzero(keep_rec)
remap = np.full(len(d['y']), -1)
remap[rec_idx] = np.arange(len(rec_idx))
wk = keep_rec[d['window_record']]
X = WindowSet(d['x'][wk].astype(np.float32), remap[d['window_record'][wk]], len(rec_idx), d['window_start'][wk], d['canonical_fallback'][wk])
X.validate()
Y = d['y'][rec_idx].astype(np.float64)
PID = d['person_id'][rec_idx]
FOLD = d['fold'][rec_idx]
STAN = d['source'][rec_idx] == 'Stanford'
ROWS = rec_idx
with np.load(DATA / 'external.npz') as z:
    ext = {k: z[k] for k in z.files}
live_rows = read(DATA / 'live-rows.json')
LIVE = WindowSet(ext['live__xcycle' if CYCLE else 'live__x'].astype(np.float32), ext['live__owner'], len(live_rows))
SELF = WindowSet(ext['self__xcycle' if CYCLE else 'self__x'].astype(np.float32), ext['self__owner'], int(ext['self__owner'].max()) + 1)
LGROUP = np.array([r['group'] for r in live_rows])
LFOLD = np.array([r['fold'] for r in live_rows])
c = base.config()


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.w = w
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.w * g, None


def features(model, x):
    for i, (conv, norm) in enumerate(zip(model.convs, model.norms)):
        x = norm(Fn.relu(conv(x)))
        if i % 2:
            x = model.dropout(Fn.max_pool1d(x, 3 if i == 5 else 2))
    return x.transpose(1, 2).reshape(x.shape[0], -1)


def head_out(model, f):
    return model.output(Fn.relu(model.hidden(f)))


def split(outer, inner=None):
    test = FOLD == outer
    valid = np.zeros(len(test), bool) if inner is None else (FOLD == inner)
    train = ~(test | valid)
    valid = valid & STAN
    return train, valid, test


def live_train(outer, inner=None):
    bad = {outer} | ({inner} if inner is not None else set())
    ok = ~np.isin(LFOLD, list(bad))
    return np.flatnonzero(ok & (LGROUP == 'TD')), np.flatnonzero(ok & (LGROUP == 'CP'))


def fit(train, valid, epochs, folder, identity, rank_sets):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'identity.json').exists():
        base.require(read(folder / 'identity.json') == identity, 'Different fit in folder')
    else:
        base.save(folder / 'identity.json', identity)
    base.setup(identity['fit_seed'], a.device)
    dev = a.device
    model = GdiCNN(c['dropout']).to(dev)
    head = nn.Sequential(nn.Linear(320, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    params = list(model.parameters()) + (list(head.parameters()) if a.arm == 'D_adversarial' else [])
    opt = torch.optim.RMSprop(params, lr=c['learning_rate'], alpha=c['rmsprop_alpha'], eps=c['rmsprop_eps'], weight_decay=0, momentum=0, centered=False)
    xt = X[np.flatnonzero(train)]
    yt_rec = Y[train]
    lo, span = base.scale_from_train(yt_rec)
    target = ((torch.from_numpy(yt_rec[xt.owner].astype(np.float32)) - lo) / span).reshape(-1)
    examples = channels(torch.from_numpy(xt.values)).reshape(-1, 12, 124)
    weights = torch.from_numpy(xt.side_weights())
    dom = torch.from_numpy(np.repeat((~STAN[train])[xt.owner], 2).astype(np.float32))
    pos = float(dom.sum())
    dom_w = torch.where(dom > 0, (len(dom) - pos) / max(pos, 1.), torch.ones_like(dom))
    st_train = np.flatnonzero(train & STAN)
    xs = X[st_train]
    xv = X[np.flatnonzero(valid)] if valid.any() else None
    if rank_sets is not None:
        td, cp = rank_sets
        rx = channels(torch.from_numpy(LIVE[np.concatenate([td, cp])].values)).reshape(-1, 12, 124).to(dev)
        n_td = int(LIVE.counts[td].sum())
    ck_path = folder / 'latest.pt'
    history = []
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=True)
        base.require(ck['identity'] == identity, 'Checkpoint mismatch')
        model.load_state_dict(ck['model'])
        head.load_state_dict(ck['head'])
        opt.load_state_dict(ck['optimizer'])
        base.restore_rng(ck['rng'])
        history = ck['history']
    for epoch in range(len(history), epochs):
        t0 = time.perf_counter()
        lr = c['learning_rate'] * c['lr_drop_factor'] ** ((epoch + 1) // c['lr_drop_every'])
        for g in opt.param_groups:
            g['lr'] = lr
        p = epoch / max(1, c['max_epochs'] - 1)
        adv_w = ADV_MAX * (2 / (1 + math.exp(-10 * p)) - 1)
        order = np.random.default_rng(identity['fit_seed'] + epoch).permutation(len(target))
        tot = np.zeros(3)
        for s in range(0, len(order), c['batch_size']):
            idx = order[s:s + c['batch_size']]
            model.train()
            head.train()
            opt.zero_grad(set_to_none=True)
            xb = examples[idx].to(dev)
            f = features(model, xb)
            pred = head_out(model, f)[:, 0]
            mse = weighted_mse(pred, target[idx].to(dev), weights[idx].to(dev))
            loss = mse + c['kernel_l2'] * model.penalty()
            extra = torch.zeros((), device=dev)
            if a.arm == 'D_adversarial':
                logit = head(Reverse.apply(f, adv_w))[:, 0]
                extra = Fn.binary_cross_entropy_with_logits(logit, dom[idx].to(dev), weight=dom_w[idx].to(dev))
                loss = loss + extra
            if rank_sets is not None:
                rp = model(rx)[:, 0].reshape(-1, 2).mean(1)
                st = rp[:n_td] if n_td else rp[:0]
                sc = rp[n_td:]
                if len(st) and len(sc):
                    extra = Fn.relu(MARGIN / float(span) - (st[:, None] - sc[None, :])).mean()
                    loss = loss + RANK_W * extra
            base.require(bool(torch.isfinite(loss)), 'Nonfinite loss')
            loss.backward()
            opt.step()
            tot += np.array([float(loss), float(mse), float(extra)]) * len(idx)
        raw = base.predict(model, xs, lo, span, dev)
        coef = base.calibrate(raw, Y[st_train])
        row = dict(epoch=epoch + 1, lr=float(lr), adv_weight=float(adv_w), loss=float(tot[0] / len(target)), mse=float(tot[1] / len(target)),
                   extra=float(tot[2] / len(target)), calibration=[float(v) for v in coef],
                   train_person_mae=float(base.person_mae(Y[st_train], base.corrected(raw, coef), PID[st_train])))
        if xv is not None:
            vr = base.predict(model, xv, lo, span, dev)
            vi = np.flatnonzero(valid)
            row['validation_person_mae'] = float(base.person_mae(Y[vi], base.corrected(vr, coef), PID[vi]))
            row['validation_people'] = int(len(np.unique(PID[vi])))
        row['seconds'] = float(time.perf_counter() - t0)
        history.append(row)
        base.save_torch(ck_path, dict(identity=identity, model=model.state_dict(), head=head.state_dict(), optimizer=opt.state_dict(),
                                      rng=base.rng_state(), history=history, lo=lo, span=span))
        base.save(folder / 'curve.json', history)
        print(json.dumps(dict(fit=folder.name, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != 'calibration'})), flush=True)
    base.save(folder / 'fit_complete.json', dict(identity=identity, epochs=len(history), checkpoint_sha256=sha(ck_path)))
    return model, lo, span, history


def identity(outer, inner, epochs, purpose):
    tr, va, te = split(outer, inner)
    td, cp = live_train(outer, inner) if a.arm == 'R_rank' else ([], [])
    return dict(arm=a.arm, purpose=purpose, outer_fold=outer, inner_fold=inner, epochs=epochs,
                fit_seed=c['seed'] + outer * 100 + (inner if inner is not None else 50),
                train_people=sorted(set(PID[tr].tolist())), validation_people=sorted(set(PID[va].tolist())),
                outer_evaluation_people=sorted(set(PID[te].tolist())),
                live_rank_people=sorted({live_rows[i]['person'] for i in list(td) + list(cp)}), data_checksums=prep)


def replay(model, lo, span, folder, train, test):
    st_train = np.flatnonzero(train & STAN)
    raw_tr = base.predict(model, X[st_train], lo, span, a.device)
    out = dict(train_row=ROWS[st_train], train_raw=raw_tr, evaluation_row=ROWS[np.flatnonzero(test)],
               evaluation_raw=base.predict(model, X[np.flatnonzero(test)], lo, span, a.device),
               live_raw=base.predict(model, LIVE, lo, span, a.device), self_raw=base.predict(model, SELF, lo, span, a.device),
               stanford_calibration=base.calibrate(raw_tr, Y[st_train]))
    np.savez_compressed(folder / 'replay.npz', **out)
    return out


def train_fold(f):
    dest = OUT / f'fold{f}'
    if (dest / 'complete.json').exists():
        return
    hist, members = [], []
    for j in range(5):
        if j == f:
            continue
        tr, va, te = split(f, j)
        rs = live_train(f, j) if a.arm == 'R_rank' else None
        folder = dest / f'inner{j}'
        model, lo, span, h = fit(tr, va, c['max_epochs'], folder, identity(f, j, c['max_epochs'], 'inner'), rs)
        hist.append(h)
        rp = replay(model, lo, span, folder, tr, te)
        members.append(dict(tag=f'inner{j}', checkpoint_sha256=sha(folder / 'latest.pt'), excluded_from_bag=bool((rp['train_raw'].std(0) < .01).any())))
    curves = np.array([[r['validation_person_mae'] for r in h] for h in hist])
    wts = np.array([h[0]['validation_people'] for h in hist], float)
    means = np.average(curves, axis=0, weights=wts)
    epoch = int(np.argmin(means)) + 1
    base.save(dest / 'selection.json', dict(selected_epoch=epoch, inner_mean=means.tolist(), outer_evaluation_used_for_selection=False))
    tr, _, te = split(f, None)
    rs = live_train(f, None) if a.arm == 'R_rank' else None
    folder = dest / 'refit'
    model, lo, span, h = fit(tr, np.zeros(len(tr), bool), epoch, folder, identity(f, None, epoch, 'refit'), rs)
    rp = replay(model, lo, span, folder, tr, te)
    members.append(dict(tag='refit', checkpoint_sha256=sha(folder / 'latest.pt'), excluded_from_bag=bool((rp['train_raw'].std(0) < .01).any())))
    base.save(dest / 'complete.json', dict(arm=a.arm, fold=f, selected_epoch=epoch, members=members))
    print('FOLD_COMPLETE', a.arm, f, flush=True)


if a.mode == 'pilot':
    c = dict(c)
    c['max_epochs'] = 3
    tr, va, te = split(0, 1)
    folder = OUT.parent / ('pilot-' + a.arm)
    model, lo, span, h = fit(tr, va, 3, folder, identity(0, 1, 3, 'pilot'), live_train(0, 1) if a.arm == 'R_rank' else None)
    rp = replay(model, lo, span, folder, tr, te)
    ck = torch.load(folder / 'latest.pt', map_location=a.device, weights_only=True)
    m2 = GdiCNN(c['dropout']).to(a.device)
    m2.load_state_dict(ck['model'])
    again = base.predict(m2, X[np.flatnonzero(te)], ck['lo'], ck['span'], a.device)
    err = float(np.abs(again - rp['evaluation_raw']).max())
    base.require(err < 1e-3, 'Replay mismatch %g' % err)
    base.save(folder / 'verification.json', dict(passed=True, arm=a.arm, replay_max_error=err, epoch_seconds=float(np.mean([r['seconds'] for r in h])), last=h[-1]))
    print('PILOT_COMPLETE', a.arm, err, flush=True)
else:
    base.require(read(OUT.parent / ('pilot-' + a.arm) / 'verification.json')['passed'], 'Pilot not verified')
    for f in [int(v) for v in a.folds.split(',')]:
        train_fold(f)
    print('ARM_COMPLETE', a.arm, flush=True)
