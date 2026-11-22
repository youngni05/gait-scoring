"""Standalone local knee observations; never replaces full-pass metrics or history."""
import math
def percentile(values,q):
 values=sorted(values);at=(len(values)-1)*q;i=int(math.floor(at));j=int(math.ceil(at));return values[i]+(values[j]-values[i])*(at-i)
def recover(segment,frames):
    """Pure interface: one analyzed segment + its original pose frames -> local rows.

    No different-pass combination, new contact labels, interpolation or GDI.
    This is a diagnostic measurement attachment, not a history-comparable metric.
    """
    curves=segment['knee']['views']['supported']['curve_deg'];out=[]
    clocks=[f['source_time_ms'] for f in frames];indices=[f['index'] for f in frames]
    dt=[b-a for a,b in zip(clocks,clocks[1:])];median=percentile(dt,.5) if dt else None
    for side in ('left','right'):
        values=curves[side];assert len(values)==len(frames)
        runs=[];current=[]
        for i,value in enumerate(values):
            good=isinstance(value,(float,int)) and math.isfinite(value)
            continuous=i==0 or indices[i]==indices[i-1]+1 and clocks[i]-clocks[i-1]<=2*median
            if not good or not continuous:
                if current:runs.append(current);current=[]
            if good:current.append(i)
        if current:runs.append(current)
        chosen=min(runs,key=lambda r:(-len(r),clocks[r[0]])) if runs else []
        full=len(chosen)==len(frames) and len(chosen)>=2
        available=len(chosen)>=2
        chosen_values=[values[i] for i in chosen]
        out.append(dict(schema='knee-local-observation/1',segment_id=segment['segment_id'],direction=segment['direction'],side=side,
            status='full_pass_observation' if full else 'limited_local_observation' if available else 'unavailable',
            value=percentile(chosen_values,.95)-percentile(chosen_values,.05) if available else None,unit='deg',
            definition='supported 2D knee p95-p05 over one original contiguous run',
            source_frame_indices=[indices[i] for i in chosen],source_times_ms=[clocks[i] for i in chosen],valid_count=len(chosen),
            start_ms=clocks[chosen[0]] if chosen else None,stop_ms=clocks[chosen[-1]] if chosen else None,
            duration_ms=clocks[chosen[-1]]-clocks[chosen[0]] if len(chosen)>=2 else 0,
            full_pass_frames=len(frames),not_full_pass=not full,minimum_two_samples_is_mathematical_only=True,
            reason=None if full else 'partial_contiguous_observation_only' if available else 'fewer_than_two_contiguous_supported_samples',
            whole_pass_state_recovered=False,clinical_change_ready=False,contact_cycle_verified=False,anatomical_label_verified=False,
            use='Separate local observation detail only; never bilateral/contact/GDI fusion or current history metric.'))
    return dict(schema='local-observation-attachment/1',segment_id=segment['segment_id'],observations=out,
        replaces_existing_metrics=False,updates_history_eligibility=False)
