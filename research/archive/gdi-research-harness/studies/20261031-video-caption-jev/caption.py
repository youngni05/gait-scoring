"""Local frozen video-capable VLM over timestamped RGB sequences; no clinical label reads."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from bootstrap import HERE,ROOT,BIG,sha
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HOME']=str(BIG/'cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
sys.path[:0]=[str(BIG/'deps'),str(ROOT/'outputs/motionbert-smoke-20261015/deps')]
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForImageTextToText

def main(limit=None,verify_cache=False):
    protocol=json.loads((HERE/'protocol.json').read_text());rows=json.loads((HERE/'video_preparation.json').read_text())
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261031)
    processor=AutoProcessor.from_pretrained(str(BIG/'model'),local_files_only=True,trust_remote_code=False,use_fast=False)
    processor.image_processor.do_image_splitting=False
    processor.image_processor.size={'longest_edge':512}
    model=AutoModelForImageTextToText.from_pretrained(str(BIG/'model'),local_files_only=True,trust_remote_code=False,
              torch_dtype=torch.float32,attn_implementation='sdpa').eval()
    path=HERE/'captions.jsonl';previous=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(r['person'],r['variant']) for r in previous};count=0
    print(json.dumps({'model_loaded':True,'torch':torch.__version__,'parameters':sum(p.numel() for p in model.parameters()),'device':'cpu'}),flush=True)
    replay_checks=[]
    with path.open('a',encoding='utf8') as f:
        for row in rows:
            pid=row['person'];frames=np.load(BIG/'frames'/f'person{pid:02d}.npy',allow_pickle=False)
            cached_pixels=cached_mask=cached_features=None
            for variant in ['focused','generic','static']:
                if (pid,variant) in done and not verify_cache:continue
                prompt=protocol['prompts']['focused' if variant=='static' else variant]
                content=[{'type':'text','text':'These are nine video frames in chronological order spanning 4.1 seconds. '+prompt}]
                images=[]
                for j,k in enumerate(row['sampled_frames']):
                    content.extend([{'type':'text','text':f'Frame at {k/30:.3f} seconds:'},{'type':'image'}])
                    images.append(Image.fromarray(frames[0 if variant=='static' else j]))
                text=processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True,tokenize=False)
                started=time.monotonic();print('preprocessing',pid,variant,flush=True)
                inputs=processor(text=text,images=[images],return_tensors='pt')
                print('input_shapes',{k:list(v.shape) for k,v in inputs.items()},'seconds',time.monotonic()-started,flush=True)
                with torch.inference_mode():
                    pixels=inputs.pop('pixel_values');mask=inputs.pop('pixel_attention_mask')
                    if cached_features is not None and torch.equal(pixels,cached_pixels) and torch.equal(mask,cached_mask):
                        image_features=cached_features
                    elif cached_features is not None and variant=='static':
                        assert torch.equal(pixels,cached_pixels[:,:1].expand_as(pixels))
                        assert torch.equal(mask,cached_mask[:,:1].expand_as(mask))
                        image_features=cached_features[:1].repeat(pixels.shape[1],1,1)
                    else:
                        image_features=model.get_image_features(pixels,mask)
                        cached_pixels=pixels;cached_mask=mask;cached_features=image_features
                    generated=model.generate(**inputs,image_hidden_states=image_features,do_sample=False,max_new_tokens=160,use_cache=True)
                output=processor.tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                assert output
                result={'person':pid,'variant':variant,'caption':output,'input_tokens':int(inputs['input_ids'].shape[1]),
                        'output_tokens':int(generated.shape[1]-inputs['input_ids'].shape[1]),'seconds':time.monotonic()-started,
                        'frames_sha256':sha(BIG/'frames'/f'person{pid:02d}.npy'),'model_revision':protocol['model_revision'],
                        'protocol_sha256':sha(HERE/'protocol.json'),'clinical_labels_read':False,'generation_cache':True,'image_feature_reuse':True}
                if verify_cache:
                    prior=next(r for r in previous if (r['person'],r['variant'])==(pid,variant))
                    assert output==prior['caption'], 'cache_replay_caption_mismatch'
                    check={'person':pid,'variant':variant,'text_identical':True,'original_seconds':prior['seconds'],
                           'cache_seconds':result['seconds'],'new_output_tokens':result['output_tokens'],
                           'note':'One-case exact greedy text replay after enabling supported generation cache; no change to input, weights, prompt or decoding limit.'}
                    replay_checks.append(check)
                    (HERE/'cache_replay.json').write_text(json.dumps(replay_checks,indent=2),encoding='utf8')
                    print(json.dumps(check),flush=True)
                    if len(replay_checks)==3:return
                    continue
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();count+=1
                print(json.dumps(result,ensure_ascii=False),flush=True)
                if limit and count>=limit:return

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--verify-cache',action='store_true')
    a=p.parse_args();main(a.limit,a.verify_cache)
