import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from recap.data import load_manifest, load_bank, folds
from recap.metrics import balanced_accuracy, macro_auroc, paired_bootstrap, within_count_auroc
from recap.models import MODELS, truncate
from recap.readers import normalize, fit_scores
from recap.protocols import recoverability, selection, transfer


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.limits=threadpool_limits(limits=1)
        rng=np.random.default_rng(3)
        self.y=np.repeat(np.arange(3),40)
        self.groups=np.array(['group{}'.format(i//2) for i in range(len(self.y))])
        self.x=normalize(np.eye(3,128)[self.y]+rng.normal(0,.02,(len(self.y),128)))
        self.q=np.repeat(np.eye(3,128,dtype=np.float32)[None],4,axis=0)
        self.manifest={'task':'toy','modality':'image','labels':['a','b','c'],'y':self.y,
                       'ids':np.array(['item{}'.format(i) for i in range(len(self.y))]),'groups':self.groups,
                       'items':[{'target':int(y),'group':g} for y,g in zip(self.y,self.groups)]}
        self.bank={'model':'gemma2','x':self.x,'prompts':self.q}

    def tearDown(self):
        self.limits.restore_original_limits()

    def test_metrics_and_paired_source_resampling(self):
        self.assertEqual(balanced_accuracy([0,0,0,1],[0,0,1,0]),1/3)
        y=np.array([[0,1],[1,0],[0,1],[1,0]])
        self.assertEqual(macro_auroc(y,y),1.)
        self.assertEqual(within_count_auroc(y,y,[3,3,4,4])['strata'],4)
        result=paired_bootstrap(self.y,self.q[0][self.y]@self.q[0].T,
                 np.zeros((len(self.y),3)),self.groups,draws=100,stratified=True)
        self.assertGreater(result['ci_low'],.6)
        self.assertAlmostEqual(result['estimate'],2/3)

    def test_supported_widths_and_normalization(self):
        for model,spec in MODELS.items():
            x=np.ones((2,max(spec['widths'])),dtype=np.float32)
            for width in spec['widths']:
                self.assertTrue(np.allclose(np.linalg.norm(truncate(x,model,width),axis=1),1))
        with self.assertRaises(ValueError):
            truncate(np.ones((2,768)),'gemma2',64)
        with self.assertRaises(ValueError):
            normalize(np.zeros((2,3)))

    def test_folds_never_split_sources(self):
        for train,test in folds(self.y,self.groups,20261011):
            self.assertFalse(set(self.groups[train]) & set(self.groups[test]))
            for tr,valid in folds(self.y[train],self.groups[train],20262011):
                self.assertFalse(set(self.groups[train[tr]]) & set(self.groups[train[valid]]))

    def test_nested_readers_cover_every_item(self):
        result,arrays=recoverability.nested(self.bank,self.manifest,128,seeds=(20261011,))
        self.assertEqual({r['reader'] for r in result['rows']},{'prompt','prototype','ridge10','ridge','logistic'})
        self.assertTrue(all(r['score']==1 for r in result['rows']))
        self.assertEqual(len(arrays['fold_20261011']),len(self.y))
        self.assertTrue(all(np.isfinite(v).all() for v in arrays.values()))

    def test_temporal_invariant_bank_cannot_get_both_directions_correct(self):
        manifest = dict(self.manifest)
        manifest['groups'] = np.array(['source{}'.format(i) for i in range(120)])
        manifest['order_tasks'] = [{'name':'order','indices':list(range(120)),
                                    'targets':np.tile([0,1],60).tolist()}]
        bank = dict(self.bank, reverse=self.x.copy())
        result, _ = recoverability.temporal(bank, manifest, 128, seeds=(20261011,))
        self.assertEqual(result['joint']['estimate'], 0.)
        self.assertEqual(result['joint']['ci_high'], 0.)

    def test_transfer_matrix_holds_target_vectors_fixed(self):
        train = dict(self.manifest)
        test = dict(self.manifest, groups=np.array(['test{}'.format(i) for i in range(120)]))
        # Rotate class axes in the second modality without changing its separability.
        video = self.x.copy()
        video[:, :3] = self.x[:, [1,2,0]]
        banks = {'audio':self.bank, 'video':dict(self.bank, x=video)}
        result, _ = transfer.run(banks, train, banks, test, 128)
        self.assertTrue(np.allclose(result['matrix'], [[1,0],[0,1]]))
        self.assertEqual(result['penalties']['video']['estimate'], 1.)
        self.assertEqual(result['penalties']['audio']['estimate'], 1.)

    def test_binary_logistic_column_order(self):
        ix=self.y<2
        score=fit_scores('logistic',self.x[ix],self.y[ix],self.x[ix],2)
        self.assertEqual(score.shape,(80,2))
        self.assertTrue(np.array_equal(score.argmax(1),self.y[ix]))

    def test_budgeted_selection_uses_only_shared_support(self):
        result,_=selection.run({'gemma2':self.bank},self.manifest,
                   {'classes':[0,1,2],'widths':{'gemma2':[128]}},seeds=(20261011,),draws=(20261021,),
                   budgets=(1,2),caps=(512,))
        self.assertEqual(len(result['rows']),20)
        for i in range(0,len(result['rows']),2):
            native,joint=result['rows'][i:i+2]
            self.assertEqual(native['support'],joint['support'])
            support=np.array(native['support']);test=np.array(native['test'])
            self.assertFalse(set(self.groups[support]) & set(self.groups[test]))
            for label in range(3):
                self.assertEqual(len(set(self.groups[support][self.y[support]==label])),native['groups_per_class'])
            if native['groups_per_class']==1:
                self.assertEqual(native['selected'],joint['selected'])
            self.assertLessEqual(4*joint['selected']['width'],joint['cap'])

    def test_selection_ties_prefer_narrower_then_native(self):
        candidates=[{'model':'jina','width':128,'reader':'ridge','template':-1},
                    {'model':'gemma2','width':128,'reader':'prompt','template':0},
                    {'model':'jina','width':32,'reader':'prototype','template':-1}]
        self.assertEqual(selection.choose(candidates,[.5,.5,.5]),candidates[2])
        self.assertEqual(selection.choose(candidates[:2],[.5,.5]),candidates[1])

    def test_transfer_rejects_source_overlap(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            transfer.run({'audio':self.bank,'video':self.bank},self.manifest,
                         {'audio':self.bank,'video':self.bank},self.manifest,128)

    def test_manifest_and_bank_reject_alignment_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            doc={'task':'toy','modality':'image','labels':['a','b'],
                 'items':[{'id':'a','group':'g1','target':0},{'id':'b','group':'g2','target':1}]}
            (path/'items.json').write_text(json.dumps(doc))
            manifest=load_manifest(path/'items.json')
            np.savez(path/'bank.npz',ids=['b','a'],x=np.ones((2,128)),model='gemma2')
            with self.assertRaisesRegex(ValueError,'IDs'):
                load_bank(path/'bank.npz',manifest)


if __name__=='__main__':
    unittest.main()
