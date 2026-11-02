import numpy as np
SIGNALS=('ankle_distance','right_knee','left_knee')
FEATURE_NAMES=[f'{s}_{m}' for s in SIGNALS for m in ('dominant_cycles_per_frame','peak_power_fraction','spectral_entropy')]+['bilateral_knee_detrended_correlation']
def detrend(x):
    x=np.asarray(x,dtype=np.float64);t=np.arange(len(x),dtype=float);t-=t.mean()
    return x-x.mean()-t*(t@x)/(t@t)
def spectrum(x):
    z=detrend(x)*np.hanning(len(x));power=np.abs(np.fft.rfft(z))[1:]**2
    if power.sum()<=1e-20:return [0.,0.,0.],True
    p=power/power.sum();nonzero=p[p>0]
    return [(int(np.argmax(p))+1)/len(x),float(p.max()),float(-(nonzero*np.log(nonzero)).sum()/np.log(len(p)))],False
def features(a):
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(124,17,3) or not np.isfinite(a).all() or not (a[:,1:7,2]>0).all():raise ValueError('Invalid observed lower-body input')
    knees=[];legs=[]
    for h,k,f in ((1,2,3),(4,5,6)):
        u=a[:,h,:2]-a[:,k,:2];v=a[:,f,:2]-a[:,k,:2]
        nu=np.linalg.norm(u,axis=1);nv=np.linalg.norm(v,axis=1)
        if not ((nu>0)&(nv>0)).all():raise ValueError('Zero-length limb')
        knees.append(np.degrees(np.arccos(np.clip((u*v).sum(1)/(nu*nv),-1,1))))
        legs.append(nu+nv)
    scale=float(np.median(np.mean(legs,axis=0)));assert scale>0
    ankle=np.linalg.norm(a[:,3,:2]-a[:,6,:2],axis=1)/scale
    out=[];flat=[]
    for s in (ankle,*knees):
        vals,isflat=spectrum(s);out.extend(vals);flat.append(isflat)
    r,l=(detrend(x) for x in knees);den=float(np.linalg.norm(r)*np.linalg.norm(l))
    undefined=den<=1e-20;corr=0. if undefined else float(np.clip((r@l)/den,-1,1));out.append(corr)
    values=np.array(out);assert values.shape==(10,) and np.isfinite(values).all()
    assert all(0<=values[i]<=.5 for i in (0,3,6))
    assert all(-1e-12<=values[i]<=1+1e-12 for i in (1,2,4,5,7,8))
    return values,dict(flat_spectra=flat,undefined_bilateral_correlation=undefined)
def checks():
    t=np.arange(124);s=np.sin(2*np.pi*3*t/124);f,flat=spectrum(s);assert not flat and f[0]==3/124
    assert spectrum(np.ones(124))[1]
    assert spectrum(np.random.default_rng(12).permutation(s))[0][1]<f[1]
    a=np.zeros((124,17,3));a[...,2]=1
    for joints,theta,dx in (((1,2,3),140+20*s,1),((4,5,6),130+15*np.cos(2*np.pi*3*t/124),-1)):
        h,k,q=joints;a[:,h,:2]=[dx,2];a[:,k,:2]=[dx,1]
        a[:,q,0]=dx+np.sin(np.radians(theta));a[:,q,1]=1+np.cos(np.radians(theta))
    x,_=features(a);b=a.copy();b[...,:2]=b[...,:2]*4.5+[10,-7]
    np.testing.assert_allclose(features(b)[0],x,rtol=0,atol=1e-10)
    angle=.6;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    b=a.copy();b[...,:2]=b[...,:2]@rot;np.testing.assert_allclose(features(b)[0],x,rtol=0,atol=1e-10)
    b=a.copy();b[:,[1,2,3,4,5,6]]=a[:,[4,5,6,1,2,3]]
    np.testing.assert_allclose(features(b)[0],np.r_[x[:3],x[6:9],x[3:6],x[9]],rtol=0,atol=1e-10)
    for kind in ('missing','zero_bone'):
        b=a.copy()
        if kind=='missing':b[0,1,2]=0
        else:b[:,1]=b[:,2]
        try:features(b)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    return dict(known_frequency=True,constant_signal=True,time_shuffle_changes_concentration=True,translation_scale_rotation_invariance=True,side_exchange=True,missing_and_zero_bone_rejected=True)
