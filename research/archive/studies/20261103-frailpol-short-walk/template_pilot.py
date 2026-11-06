from signal_pilot import *
from gaitmap.stride_segmentation import BarthOriginalTemplate
from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory

def main():
    with zipfile.ZipFile(BIG/'raw/IMU_signals.zip') as z:
        m=filemap(z); report=[]
        for pid in [1,9,63,100,190,479,482,483,621,622,655,763]:
            raw=rawread(z,m,pid);a=fsf(raw);fbf=convert_to_fbf(a,left_like='left',right_like='right')
            dtw=BarthDtw(template=BarthOriginalTemplate(use_cols=['gyr_ml'])).segment(fbf,sampling_rate_hz=100)
            ed=HerzerEventDetection(min_vel_search_win_size_ms=130,mid_swing_peak_prominence=.03*180/np.pi,mid_swing_n_considered_peaks=2).detect(fbf,stride_list=dtw.stride_list_,sampling_rate_hz=100)
            r=dict(id=pid,strides={s:d.reset_index().to_dict('records') for s,d in ed.min_vel_event_list_.items()})
            try:
                events={s:d for s,d in ed.min_vel_event_list_.items() if len(d)}
                tr=StrideLevelTrajectory().estimate({s:a[s] for s in events},stride_event_list=events,sampling_rate_hz=100)
                tp=TemporalParameterCalculation().calculate(stride_event_list=events,sampling_rate_hz=100)
                sp=SpatialParameterCalculation().calculate(stride_event_list=events,positions=tr.position_,orientations=tr.orientation_,sampling_rate_hz=100)
                r['parameters']={s:tp.parameters_[s].join(sp.parameters_[s]).reset_index().replace({np.nan:None}).to_dict('records') for s in events}
            except Exception as e:r['error']=repr(e);traceback.print_exc()
            report.append(r)
            print(pid,{s:len(d) for s,d in ed.min_vel_event_list_.items()},r.get('error',''),flush=True)
        save(HERE/'template-pilot.json',report)
        print(json.dumps(report[:2],indent=2))
if __name__=='__main__':main()
