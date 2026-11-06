from core import *
from scipy.optimize._numdiff import approx_derivative
rng=np.random.default_rng(3);x=rng.normal(size=(30,7));base=75+x[:,0]*5;y=base+rng.normal(size=30)*8;ids=np.repeat(np.arange(10),3);w=weights(ids)
mean,sd=standardize(x,w);xx,zz=design(x,'score_spread',mean,sd);theta=np.r_[np.zeros(xx.shape[1]),np.log(5.),np.zeros(zz.shape[1]-1)]
value,g=objective(theta,xx,zz,base,y,w,.1)
gn=approx_derivative(lambda a:np.array([objective(a,xx,zz,base,y,w,.1)[0]]),theta).ravel()
np.testing.assert_allclose(g,gn,atol=1e-6,rtol=1e-5)
for family in ['score_only','score_spread']:
 model=fit_head(x,base,y,ids,family,.1);p=predict(model,x,base)
 assert np.isfinite(p).all() and (p>0).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
 for t in [.25,1,4]:np.testing.assert_array_equal(np.argmax(p,axis=1),np.argmax(temperature(p,t),axis=1))
np.testing.assert_array_equal(labels(np.array([-1.,0.,9.999,10.,99.999,100.,150.,151.])),[0,1,1,2,10,11,16,16])
assert all(abs(w[ids==i].sum()-.1)<1e-12 for i in range(10))
try:disjoint([1],[2],[1])
except AssertionError:pass
else:raise AssertionError('Leak not rejected')
save(O/'core_tests.json',dict(passed=True,gradient_max_difference=float(np.abs(g-gn).max()),probability_simplex=True,temperature_argmax_preserved=True,
   band_edges_and_open_tails=True,person_equal_weights=True,injected_leak_rejected=True))
print('CORE_TESTS_PASSED',flush=True)
