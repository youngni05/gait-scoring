"""Align the descriptive thigh-variation indicator with the fixed accepted windows."""
from pathlib import Path
import sys,json,hashlib
H=Path(__file__).resolve().parent

def main():
 p=H/'prepare.py';source=p.read_text(encoding='utf-8-sig')
 old="out.append([float((~np.isfinite(q[:,:,:2])|(q[:,:,:2]==0)).mean()),float((np.nan_to_num(q[:,:,2],nan=0)<.3).mean()),float(mask[start:start+124][:,JOINTS].mean()),float(np.median(a[usable])) if usable.any() else 0.,float(length.std()/length.mean())])"
 new="window_lengths=[]\n  for hip,knee in [(12,13),(9,10)]:\n   good=valid[start:start+124,hip]&valid[start:start+124,knee];segment=p[start:start+124];window_lengths.extend(np.linalg.norm(segment[good,hip,:2]-segment[good,knee,:2],axis=1))\n  window_lengths=np.array(window_lengths)\n  out.append([float((~np.isfinite(q[:,:,:2])|(q[:,:,:2]==0)).mean()),float((np.nan_to_num(q[:,:,2],nan=0)<.3).mean()),float(mask[start:start+124][:,JOINTS].mean()),float(np.median(a[usable])) if usable.any() else 0.,float(window_lengths.std()/window_lengths.mean())])"
 assert old in source;p.write_text(source.replace(old,new),encoding='utf-8')
 import prepare as m
 prep=m.read(H/'preparation.json');d=m.np.load(m.B/'summaries.npz');arrays={k:d[k] for k in d.files};before=arrays['quality'].copy()
 for i in range(len(arrays['y'])):
  raw=m.np.load(m.B/'raw'/f'{i:04d}.npz')['raw'];starts=arrays['window_start'][arrays['window_record']==i];arrays['quality'][i]=m.quality(raw,starts)
 assert m.np.array_equal(before[:,:4],arrays['quality'][:,:4]);m.np.savez_compressed(m.B/'summaries.npz',**arrays);prep['summary_sha256']=m.sha(m.B/'summaries.npz');m.save(H/'preparation.json',prep)
 m.save(H/'quality-window-check.json',dict(reason='Correct descriptive thigh-length variation to use exactly accepted windows, as protocol requires; other four indicators and all model inputs unchanged',max_indicator_change=float(m.np.abs(before[:,4]-arrays['quality'][:,4]).max()),done_before_new_predictions=True))
 print('Quality indicator window alignment corrected; model inputs unchanged',flush=True)
if __name__=='__main__':main()
