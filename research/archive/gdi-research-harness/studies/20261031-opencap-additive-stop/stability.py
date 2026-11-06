from study import *
def main():
    spec=importlib.util.spec_from_file_location('old_assess',HERE.parent/'20261031-opencap-leave283/assess.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames=[];stability=[];exclusion=[]
    for cohort in ['exclude283','include283']:
        p=pd.read_csv(HERE/cohort/'predictions.csv');p['scenario']=cohort;p['family']='Ridge';frames.append(p)
        for id in ['sub-397','sub-283']:
            for view,z in p[p.id==id].groupby('view'):
                z=z.sort_values('repeat');v=z.prediction.to_numpy();err=v-z.target.to_numpy()
                stability.append({'scenario':cohort,'id':id,'view':view,'target':float(z.target.iloc[0]),'mean_prediction':float(v.mean()),'prediction_SD':float(v.std(ddof=0)),
                    'min_prediction':float(v.min()),'max_prediction':float(v.max()),'prediction_range':float(np.ptp(v)),
                    'MAE':float(abs(err).mean()),'underprediction':float(np.maximum(-err,0).mean()),'overprediction':float(np.maximum(err,0).mean()),'predictions_by_partition':v.tolist()})
        for view,z in p[(p.target>=20)&~p.id.isin(['sub-283','sub-397'])].groupby('view'):
            err=z.prediction-z.target;exclusion.append({'scenario':cohort,'view':view,'people':z.id.nunique(),'MAE':float(abs(err).mean()),'underprediction':float(np.maximum(-err,0).mean()),'overprediction':float(np.maximum(err,0).mean())})
    combined=pd.concat(frames,ignore_index=True);common=combined[combined.id!='sub-283'].copy();part,persons,runs,metrics=module.aggregate(common)
    comparisons=[]
    for view in ['Speed_20','Learned10_20','Add10_20','Move10_20','Learned15_20','Add15_20','Move15_20']:
        for band in ['all','ge20']:comparisons.append(module.contrast(persons,runs,('include283','Ridge',view),('exclude283','Ridge',view),band))
    for a,b in [('Add10_20','Speed_20'),('Add10_20','Learned10_20'),('Move10_20','Learned10_20')]:
        for band in ['all','ge20']:comparisons.append(module.contrast(persons,runs,('include283','Ridge',a),('include283','Ridge',b),band))
    save('stability.json',stability);pd.DataFrame(stability).drop(columns='predictions_by_partition').to_csv(HERE/'stability.csv',index=False)
    save('common176-inclusion-effect.json',{'metrics':metrics,'contrasts':comparisons,'scope':'both runs evaluated on same original176 including same8long;283 excluded from aggregation only, retained in include-run training when not held out'})
    save('exclude397-evaluation-only.json',{'training_unchanged':True,'metrics':exclusion})
    print(pd.DataFrame(stability).query("view in ['Speed_20','Learned10_20','Add10_20','Move10_20']")[['scenario','id','view','mean_prediction','prediction_SD','min_prediction','max_prediction','MAE','underprediction']].to_string(index=False),flush=True)
    print('SAME176',pd.DataFrame(metrics).query("band=='ge20' and view in ['Speed_20','Learned10_20','Add10_20','Move10_20']")[['scenario','view','people','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
if __name__=='__main__':main()
