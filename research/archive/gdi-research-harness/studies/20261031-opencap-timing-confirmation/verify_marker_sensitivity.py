from audit import *
def main():
    sources=json.loads((HERE/'marker-boundaries.json').read_text());rows=json.loads((HERE/'marker-tail-sensitivity.json').read_text())
    for r in rows:
        s=next(s for s in sources if s['id']==r['id'] and 'tug_cone' in s['file']);fps,t,names,xyz=trc(HERE/s['file']);end=len(t)-r['removed_tail_frames'];t=t[:end]
        h=(xyz[:end,list(names).index('RHJC_study'),1]+xyz[:end,list(names).index('LHJC_study'),1])/2
        lo=float(min(h));hi=float(max(h));assert abs(lo-r['minimum_height'])<1e-10 and abs(hi-r['maximum_height'])<1e-10
        assert abs(lo+.3*(hi-lo)-r['absolute30percent_threshold'])<1e-10
        expected_peak=max(abs(h[i+1]-h[i])*fps for i in range(len(h)-round(.5*fps),len(h)-1));assert abs(expected_peak-r['maximum_last_halfsecond_vertical_speed'])<1e-10
        start=next(i for i,v in enumerate(h) if v>lo+.3*(hi-lo));last=max(i for i,v in enumerate(h) if v>lo+.7*(hi-lo));stop=next(i for i in range(last+1,len(h)) if h[i]<lo+.3*(hi-lo));b=r['boundary']
        assert np.allclose([t[start],t[stop],(stop-start)/fps],[b['start'],b['end'],b['duration']],atol=1e-8,rtol=0)
    save('marker-sensitivity-verification.json',{'passed':True,'conditions':len(rows),'independent_absolute_height_threshold_reconstruction':True,'models_modified':False})
    print('MARKER SENSITIVITY VERIFIED',len(rows),flush=True)
if __name__=='__main__':main()
