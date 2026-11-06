import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import study as s
import jev_client as client


class StudyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=s.load()

    def test_label_and_other_person_independence(self):
        for name,d in self.data.items():
            index=0; altered=copy.deepcopy(self.data)
            fold=np.asarray(d['fold']); held=np.flatnonzero(fold==fold[index])
            for i in held: altered[name]['y'][i]=-999999
            for arm in ['numeric','narrative','sequence','examples']:
                self.assertEqual(s.payload(self.data,name,index,arm),s.payload(altered,name,index,arm))
            # Main representations must not depend even on other people's X.
            for i in range(1,len(d['ids'])): altered[name]['X'][i]=[99999]*len(d['names'])
            for arm in ['numeric','narrative','sequence']:
                self.assertEqual(s.payload(self.data,name,index,arm),s.payload(altered,name,index,arm))

    def test_shapes_folds_and_request_boundaries(self):
        self.assertEqual(len(s.jobs(self.data)),2040)
        self.assertEqual(len({s.job_id(j) for j in s.jobs(self.data)}),2040)
        for name,d in self.data.items():
            self.assertEqual(len(d['ids']),len(set(d['ids'])))
            self.assertEqual(set(d['fold']),set(range(5)))
            self.assertTrue(np.isfinite(np.asarray(d['y'])).all())
            for arm in ['numeric','narrative','sequence','examples','missing']:
                p=s.payload(self.data,name,0,arm)
                self.assertLess(len(json.dumps(p).encode()),80000)
                self.assertNotIn('ids',p['state'])
                self.assertNotIn('target',p['state'])
                json.dumps(p,allow_nan=False)
            p=s.payload(self.data,name,0,'examples')
            self.assertEqual(len(p['state']['training_examples']),6)

    def test_numeric_and_narrative_same_information(self):
        for name in self.data:
            p=s.payload(self.data,name,0);q=s.payload(self.data,name,0,'narrative')
            for (k,v),line in zip(p['state']['measurements'].items(),q['state']['observations']):
                self.assertIn(k.replace('_',' '),line)
                self.assertIn(str(v) if v is not None else 'unavailable',line)
            self.assertEqual(p['questions'],q['questions'])

    def test_order_control_changes_only_order(self):
        for name in self.data:
            p=s.payload(self.data,name,0);q=s.payload(self.data,name,0,reverse=True)
            self.assertEqual(p,q)
            for key in p['questions']:
                self.assertEqual(list(p['questions'][key]['criteria']),list(reversed(q['questions'][key]['criteria'])))

    def test_invalid_api_answer_rejected_without_body_logging(self):
        p=s.payload(self.data,'TUG',0)
        class FakeResponse:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps({'model':'jev-1.14.0','answers':{}}).encode()
        with patch.object(client,'api_key',return_value='synthetic-test-placeholder'),patch.object(client.urllib.request,'urlopen',return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError,'unexpected_questions'):client.call(p)


if __name__=='__main__':unittest.main()
