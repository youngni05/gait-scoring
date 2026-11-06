"""Follow-up description of output left-right differences; no correction/fitting."""
import run as r
np,m,HERE=r.np,r.m,r.HERE


def main():
    assert not (HERE/'side_diagnostic.json').exists()
    r.save('side_protocol.json',dict(purpose='Describe whether output reproduces actual left-right score differences',
        declared_before_calculation=True, scope='same 695 held-person predictions',
        metrics=['person-weighted absolute side difference','side-difference absolute error','mean-score absolute error',
                 'person-weighted fraction where half-side-difference error exceeds mean-score error'],
        identity='mean(abs(left_error),abs(right_error)) = max(abs(mean_error), abs(half_side_difference_error))',
        limitation='Uses true scores for retrospective diagnosis, not predictive inputs or causal evidence'))
    d=r.load_npz(r.BLEND/'oof_predictions.npz');ids=d['person_id'];people=np.unique(ids);y=d['y']
    def macro(a):return float(np.mean([a[ids==pid].mean() for pid in people]))
    rows=[]
    for name in ['pls32','gpu_last_stage','equal_blend']:
        pred=d['predictions'][d['methods'].tolist().index(name)]
        diff=pred[:,0]-pred[:,1]; true_diff=y[:,0]-y[:,1]
        mean_error=abs(pred.mean(1)-y.mean(1));half_diff_error=abs(diff-true_diff)/2
        actual_error=abs(pred-y).mean(1)
        np.testing.assert_allclose(actual_error,np.maximum(mean_error,half_diff_error),atol=1e-12,rtol=0)
        rows.append(dict(method=name,actual_abs_LR_difference=macro(abs(true_diff)),
                         predicted_abs_LR_difference=macro(abs(diff)),LR_difference_MAE=macro(abs(diff-true_diff)),
                         mean_score_MAE=macro(mean_error),person_MAE=macro(actual_error),
                         fraction_side_difference_error_dominates=macro(half_diff_error>mean_error)))
    r.save('side_diagnostic.json',dict(rows=rows,identity_checked_records=1064*3,training=False,GPU_used=False))
    print(rows)


if __name__=='__main__':main()
