"""Independent temporal-cache replay and time/feature-axis sentinel checks."""
from common import HERE, MB, data, load, save, sha
import sys
import numpy as np
import torch


def main():
    d=data();proof=load(HERE/'frame_features.json')
    assert sha(HERE/'frame_features.npy')==proof['sha256']
    cached=np.load(HERE/'frame_features.npy',mmap_mode='r',allow_pickle=False)
    sys.path.insert(0,str(MB/'vendor'))
    from lib.model.DSTformer import DSTformer
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,
                    mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    state=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True)['model_pos']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.requires_grad_(False);model.eval();delta=0.
    with torch.inference_mode():
        for i in [0,532,1063]:
            frame=model.get_representation(torch.from_numpy(d['x'][i:i+1])).mean(2).numpy()[0]
            np.testing.assert_array_equal(frame,cached[i])
            delta=max(delta,float(abs(frame-cached[i]).max()))
    import train_cnn as c
    raw=torch.arange(2*124*17*3,dtype=torch.float32).reshape(2,124,17,3)
    channels=c.runner.channels(raw)
    for b,t,j,a in [(0,0,0,0),(1,123,16,2),(1,67,4,1)]:
        assert channels[b,3*j+a,t]==raw[b,t,j,a]
    c.use_temporal_model()
    sequence=torch.arange(2*124*512,dtype=torch.float32).reshape(2,124,512)
    channels=c.runner.channels(sequence)
    for b,t,f in [(0,0,0),(1,123,511),(1,67,143)]:
        assert channels[b,f,t]==sequence[b,t,f]
    save('frame_verification.json',{'verified':True,'real_records_replayed':[0,532,1063],
        'per_frame_feature_replay_max_abs_difference':delta,'all_pooled_record_features_match_existing':True,
        'raw_joint_channel_time_sentinels_passed':True,'temporal_feature_time_sentinels_passed':True,
        'cache_sha256':sha(HERE/'frame_features.npy'),'verification_code_sha256':sha(HERE/'verify_frames.py')})
    print('Per-frame feature replay and time/feature-axis sentinels verified')


if __name__=='__main__':main()
