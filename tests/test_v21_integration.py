"""최신 모듈과 기존 대시보드 사이의 계약을 검증한다."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import app
from modules.pseudonymization_pipeline import AttributeRisk, RiskLevel


class LatestModuleIntegrationTest(unittest.TestCase):
    def test_inspect_does_not_run_sensitive_detector(self):
        with patch.object(app, '_build_sensitive_detection_pipeline', side_effect=AssertionError('unexpected detection')):
            response = app.app.test_client().post('/api/inspect', data={
                'data_file': (io.BytesIO(b'age,target\n20,0\n30,1\n'), 'sample.csv'),
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual([c['name'] for c in response.json['columns']], ['age', 'target'])
        self.assertNotIn('legal_sensitive_columns', response.json)

    def test_separate_exposure_axes_and_continuous_scores(self):
        result = {'modules': {
            'identification': {'attributes': [{'attribute': 'health', 'final_grade': 'M', 'risk_score': 2.5}]},
            'aggregation': {'attribute_grades': {'health': 'H'}, 'attribute_scores': {'health': .8}},
            'exposure': {'attribute_scores': {'health': .9}, 'risk_components': {
                'health': {'internal_risk': .9, 'external_risk': .1}}, 'exposure_matrix': []},
        }}
        risk = app._build_attribute_risk_report(result, pd.DataFrame({'health': [0, 1]}))['health']
        self.assertEqual(risk.exposure_dist, RiskLevel.H)
        self.assertEqual(risk.exposure_link, RiskLevel.L)
        self.assertEqual(risk.risk_scores, {'reid': 2.5, 'aggregation': .8, 'exposure_dist': .9, 'exposure_link': .1})
        self.assertTrue(risk.purpose_fit)
        self.assertTrue(risk.needed_for_purpose)

    def test_latest_identification_score_and_k_gate(self):
        # Risk classification itself does not need a Julia session.
        spec = importlib.util.spec_from_file_location('standalone_risk',
            app.BASE_DIR / 'modules/correctmatch_merged/risk_classification.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        combos = pd.DataFrame([
            {'combination': ('age',), 'size': 1, 'correctness': .02, 'uniqueness': .5, 'k_violation_ratio': 0.0},
            {'combination': ('region',), 'size': 1, 'correctness': .005, 'uniqueness': .4, 'k_violation_ratio': 0.0},
        ])
        result = module.classify_attribute_risk(combos).set_index('attribute')
        self.assertEqual(result.at['age', 'combo_grade'], 'L')
        self.assertEqual(result.at['age', 'final_grade'], 'M')
        self.assertEqual(result.at['age', 'risk_score'], 2.0)

    def test_automatic_processing_passes_scores_to_latest_search(self):
        df = pd.DataFrame({'income': [10,20,30,40,50,60]})
        source = {'analysis_config': {}, 'modules': {'exposure': {
            'attribute_scores': {'income': .9},
            'risk_components': {'income': {'internal_risk': .9, 'external_risk': .0}},
        }}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source').mkdir()
            (root / 'source/integrated_result.json').write_text(json.dumps(source))
            def reassess(frame, out_dir):
                return {'risk_components': {'income': {'internal_risk': .15, 'external_risk': .0}}}, []
            with patch.object(app, 'RESULT_DIR', root), patch.object(app, '_run_exposure', side_effect=reassess):
                # 라운드 루프 결과 dict를 돌려준다 (v23 동작 확인을 위해 목표 M, 1라운드)
                out = app._run_final_repseudonymization('source', {'income'}, df, root/'out',
                                                        target_level='M', max_rounds=1)
                processed, summary = out['processed'], out['summary']
            self.assertEqual(len(out['rounds']), 1)
            self.assertEqual(out['utility_tv']['status'], 'success')
            self.assertEqual(len(summary), 1)
            self.assertGreater(summary[0]['changed_rows'], 0)
            self.assertIn('L', summary[0]['protection'])
            self.assertEqual(len(processed), len(df))


if __name__ == '__main__':
    unittest.main()
