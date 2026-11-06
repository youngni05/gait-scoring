"""Independent real-annotation reconciliation plus synthetic analysis checks.

No GAVD score is predicted here. SQLite independently groups all original rows.
"""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import unittest
import uuid

from audit_gavd import HERE, contiguous_run, write_json
from compare_scores import effect, merge, summarize
from prepare_comparison import read_csv, export


class ScientificChecks(unittest.TestCase):
    def test_auc_direction_and_ties(self):
        self.assertEqual(effect([80,90],[40,50]),(40.,1.))
        self.assertEqual(effect([40,50],[80,90]),(-40.,0.))
        self.assertEqual(effect([75,75],[75,75]),(0.,.5))
        self.assertAlmostEqual(effect([70,90],[70,80])[1],.625)

    def test_gaps_cannot_be_concatenated(self):
        self.assertEqual(contiguous_run(set(range(62))|set(range(100,162))),62)
        self.assertEqual(contiguous_run(set(range(124))),124)

    def test_paired_cluster_resampling(self):
        rows=[dict(cluster_id=c,label=label,score=score)
              for c in ['A','B','C'] for label,score in [('Normal Gait',80),('Abnormal Gait',60)]]
        got=summarize(rows,200,3)
        self.assertEqual(got['independent_cluster_ids'],3)
        self.assertEqual(got['mean_difference_ci95'],[20.,20.])
        self.assertEqual(got['auc_ci95'],[1.,1.])
        self.assertEqual(got['valid_bootstrap_draws'],200)

    def test_repeated_records_not_extra_clusters(self):
        rows=[dict(cluster_id='N',label='Normal Gait',score=80)]*50
        rows += [dict(cluster_id='A',label='Abnormal Gait',score=60)]
        got=summarize(rows,200,5)
        self.assertEqual(got['normal_clusters'],1)
        self.assertEqual(got['normal_minus_abnormal'],20)

    def test_prediction_contract(self):
        labels=[dict(record_id='s',cluster_id='video:v',label='Normal Gait',view='left side',gait_pattern='normal',fps='30')]
        pred=[dict(record_id='s',person_id='video:v',fold=str(k),left_GDI='80',right_GDI='90') for k in range(5)]
        self.assertEqual(merge(pred,labels)[0][0]['score'],85)
        with self.assertRaises(ValueError):merge(pred[:-1],labels)
        with self.assertRaises(ValueError):merge(pred+[pred[0]],labels)
        pred[0]['left_GDI']='nan'
        with self.assertRaises(ValueError):merge(pred,labels)

    def test_no_unreviewed_manifest(self):
        template=HERE/'review_template.csv'
        if not template.exists():self.skipTest('inventory not ready')
        out=HERE/('_must_not_be_created_'+uuid.uuid4().hex)
        with self.assertRaisesRegex(ValueError,'No reviewed real records'):
            export(template,out)
        self.assertFalse(out.exists())


def reconcile():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE a(seq TEXT, frame INTEGER, label TEXT, pattern TEXT, view TEXT, video TEXT)')
    source=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for info in source['files']:
        path=Path(info['local_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise AssertionError('Source hash changed')
        with path.open(encoding='utf-8-sig',newline='') as f:
            db.executemany('INSERT INTO a VALUES(?,?,?,?,?,?)',
                ((r['seq'],int(r['frame_num']),r['dataset'],r['gait_pat'],r['cam_view'],r['id']) for r in csv.DictReader(f)))
    observed={r['seq']:r for r in read_csv(HERE/'sequences.csv')}
    for seq,n,lo,hi,frames,views,labels,videos in db.execute(
        'SELECT seq,count(*),min(frame),max(frame),count(DISTINCT frame),count(DISTINCT view),'
        'count(DISTINCT label),count(DISTINCT video) FROM a GROUP BY seq'):
        row=observed[seq]
        assert int(row['annotated_frames'])==frames==n
        assert (int(row['frame_first']),int(row['frame_last']))==(lo,hi)
        # Current source contains no gaps. This is independently checked, not assumed.
        assert hi-lo+1==frames==int(row['longest_contiguous_frames'])
        assert labels==videos==1
        assert bool(row['metadata_conflicts'])==(views>1)
        assert bool(int(row['length_candidate']))==(frames>=124 and views==1)
    counts=db.execute('SELECT count(*),count(DISTINCT seq),count(DISTINCT video) FROM a').fetchone()
    summary=json.loads((HERE/'annotation_summary.json').read_text(encoding='utf-8'))
    assert counts==(summary['annotation_rows'],summary['sequences'],summary['unique_videos'])
    conflicts=db.execute("SELECT count(DISTINCT seq) FROM a WHERE (label='Normal Gait') != (pattern='normal')").fetchone()[0]
    assert conflicts==summary['inconsistent_label_sequences']
    db.close()
    return dict(annotation_rows=counts[0],sequences=counts[1],videos=counts[2],label_conflicts=conflicts,
                checks='Independent SQLite counts, boundaries, no gaps/duplicates, label/view consistency, SHA256')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    data=reconcile()
    write_json(HERE/'verification.json',dict(synthetic_tests=result.testsRun,skipped=len(result.skipped),
               failures=0,real_annotation_reconciliation=data,GAVD_inference_run=False))
    print(json.dumps(data))
