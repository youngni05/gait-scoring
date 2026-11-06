from study import *
def run():
    d=pd.read_csv(HERE/'main-estimates.csv');n=len(json.loads((HERE/'availability.json').read_text())['selected_people']);assert len(d)==n*16 and not d.duplicated(['subject','seconds','camera','method']).any()
    reference=d[d.method.eq('Reference3D')&d.seconds.eq(4)&d.camera.eq('PG1')][['subject','cadence','quality']].rename(columns={'cadence':'reference_cadence','quality':'reference_quality'})
    z=d.merge(reference,on='subject',validate='many_to_one');rows=[];pairs=[];matched=[]
    for seconds in [2,4]:
        for method in METHODS[:3]:
            g=z[z.seconds.eq(seconds)&z.method.eq(method)]
            for camera in ['PG1','PG2']:
                c=g[g.camera.eq(camera)];valid=c.reference_quality&c.reference_cadence.notna()&c.cadence.notna();accepted=valid&c.quality;e=c.cadence-c.reference_cadence
                rows.append(dict(seconds=seconds,method=method,camera=camera,selected_people=n,reference_valid_people=int(c.reference_quality.sum()),candidate_quality_people=int(c.quality.sum()),candidate_quality_coverage=float(c.quality.mean()),accepted_people=int(accepted.sum()),raw_proxy_discrepancy=float(abs(e[valid]).mean()),accepted_proxy_discrepancy=float(abs(e[accepted]).mean()),accepted_bias=float(e[accepted].mean())))
            a=g[g.camera.eq('PG1')].set_index('subject');b=g[g.camera.eq('PG2')].set_index('subject');valid=a.reference_quality&a.cadence.notna()&b.cadence.notna();accepted=valid&a.quality&b.quality;diff=b.cadence-a.cadence
            pairs.append(dict(seconds=seconds,method=method,selected_people=n,accepted_people=int(accepted.sum()),raw_view_difference=float(abs(diff[valid]).mean()),accepted_view_difference=float(abs(diff[accepted]).mean()),accepted_signed_difference=float(diff[accepted].mean())))
    for method in METHODS[:3]:
        for camera in ['PG1','PG2']:
            a=z[z.method.eq(method)&z.seconds.eq(2)&z.camera.eq(camera)].set_index('subject');b=z[z.method.eq(method)&z.seconds.eq(4)&z.camera.eq(camera)].set_index('subject');ok=a.reference_quality&a.quality&b.quality&a.cadence.notna()&b.cadence.notna();e2=abs(a.cadence-a.reference_cadence);e4=abs(b.cadence-b.reference_cadence)
            matched.append(dict(method=method,camera=camera,common_people=int(ok.sum()),proxy_discrepancy_2sec=float(e2[ok].mean()),proxy_discrepancy_4sec=float(e4[ok].mean()),difference_4minus2=float((e4-e2)[ok].mean())))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(pairs).to_csv(HERE/'paired.csv',index=False);pd.DataFrame(matched).to_csv(HERE/'length-matched.csv',index=False);save(HERE/'analysis.json',dict(selected_people=n,estimates=len(d),metric_rows=len(rows),paired_rows=len(pairs),length_matched_rows=len(matched),analysis_sha256=sha(__file__)))
if __name__=='__main__':run()
