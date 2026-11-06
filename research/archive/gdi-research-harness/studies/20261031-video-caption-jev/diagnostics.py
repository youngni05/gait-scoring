"""Summarize Jev's responses and static-control similarity, without reading targets."""
import json
from jev_and_compare import HERE,TRAITS,CHOICES,VARIANTS,save
import numpy as np


def main():
    rows=[json.loads(s) for s in (HERE/'jev_responses.jsonl').read_text().splitlines()]
    caps=[json.loads(s) for s in (HERE/'captions.jsonl').read_text().splitlines()]
    lookup={(r['person'],r['variant']):r['response']['answers'] for r in rows}
    people=sorted({r['person'] for r in rows})
    assert len(rows)==60 and len(caps)==36 and len(people)==12
    out={'clinical_labels_read':False,'people':12,'choice_counts':{},'focused_static':{}}
    for v in VARIANTS:
        out['choice_counts'][v]={q:{c:sum(lookup[p,v][q]['choice']==c for p in people) for c in CHOICES} for q in TRAITS}
    for q in TRAITS:
        a=np.array([[lookup[p,'focused'][q]['probabilities'][c] for c in CHOICES] for p in people])
        b=np.array([[lookup[p,'static'][q]['probabilities'][c] for c in CHOICES] for p in people])
        out['focused_static'][q]={'same_choice_people':sum(lookup[p,'focused'][q]['choice']==lookup[p,'static'][q]['choice'] for p in people),
                                  'mean_probability_total_variation':float(np.abs(a-b).sum(1).mean()/2)}
    out['caption_seconds']=sum(c['seconds'] for c in caps)
    out['note']='Response similarity is not accuracy. Static captions have no observed temporal changes. Visual correctness requires reviewing the stored frames and captions.'
    save('diagnostics.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
