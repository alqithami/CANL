"""Synthetic tests of the confirmation contract and seed-level inference."""
import copy
import hashlib
import itertools
import json
import random
import tempfile
from pathlib import Path
import unittest

import numpy as np

import analyze as a


def fixture():
    protocol = {
        "schema_version": 1, "protocol_id": a.STUDY_ID,
        "locked": True, "stage": "confirmation", "source_binding_sha256": "a"*64, "test_release_receipt_sha256":"c"*64,
        "methods": list(a.METHODS), "label_fractions": list(a.FRACTIONS),
        "label_counts": list(a.LABEL_COUNTS), "confirmation_seeds": list(a.CONFIRMATION_SEEDS),
        "analysis": copy.deepcopy(a.ANALYSIS_CONTRACT),
    }
    raw = json.dumps(protocol, sort_keys=True).encode()
    sha = hashlib.sha256(raw).hexdigest()
    records = []
    # Seed-varying common difficulty must cancel under actual seed pairing.
    for method, seed, j in itertools.product(a.METHODS, protocol["confirmation_seeds"], range(4)):
        value = 30 + (seed-92001)*0.25 + j*8 + (1 if method == "ratio_dcr" else 0)
        records.append({"method": method, "seed": seed, "label_fraction": a.FRACTIONS[j],
                        "label_count": a.LABEL_COUNTS[j], "accuracy_percent": value,
                        "stage": "confirmation", "status": "COMPLETED_VERIFIED",
                        "protocol_sha256": sha, "source_binding_sha256": "a"*64,
                        "receipt_sha256": "b"*64, "evaluation_partition":"official_test"})
    data = {"schema_version": 1, "protocol_id": a.STUDY_ID, "stage": "confirmation",
            "status": "COMPLETED_VERIFIED", "protocol_sha256": sha,
            "source_binding_sha256": "a"*64, "records": records, "evaluation_partition":"official_test", "test_release_verified":True, "test_release_receipt_sha256":"c"*64}
    return protocol, sha, data


class EndpointTests(unittest.TestCase):
    def test_aulc_flat_and_known_trapezoid(self):
        self.assertAlmostEqual(a.normalized_aulc([37,37,37,37]),37)
        self.assertAlmostEqual(a.normalized_aulc([0,10,20,30]),17)
        self.assertAlmostEqual(a.normalized_aulc([10,20,40,60]),35)

    def test_exact_all_positive_ten_seeds(self):
        self.assertEqual(a.exact_sign_flip_p(np.ones(10)),2/1024)
        self.assertEqual(a.exact_sign_flip_p(np.arange(1,11)),2/1024)

    def test_exact_all_zero(self):
        self.assertEqual(a.exact_sign_flip_p(np.zeros(10)),1)

    def test_exact_matches_bruteforce_and_zero_duplication(self):
        d = np.asarray([1.,-2.,0.,4.])
        observed = abs(d.sum())
        brute = np.mean([abs(np.dot(signs,d)) >= observed for signs in itertools.product([-1,1],repeat=4)])
        self.assertEqual(a.exact_sign_flip_p(d),brute)
        self.assertEqual(a.exact_sign_flip_p([1,-2,4]),brute)

    def test_holm_family_order_and_monotonicity(self):
        np.testing.assert_allclose(a.holm_adjust([.04,.001,.02,.5,.03]),[.09,.005,.08,.5,.09])
        self.assertEqual(a.holm_adjust([1,1,1,1,1]),[1]*5)

    def test_seed_pairing_and_record_order(self):
        protocol,sha,data = fixture()
        first = a.analyze(data,protocol,sha)
        random.Random(19).shuffle(data["records"])
        second = a.analyze(data,protocol,sha)
        self.assertEqual(first,second)
        self.assertEqual(first["independent_unit"],"paired seed")
        self.assertEqual(first["primary_family_size"],3)
        for contrast in first["planned_contrasts"]:
            np.testing.assert_allclose(contrast["seed_differences_pp"],np.ones(10))
            self.assertEqual(contrast["exact_sign_flip_p_raw"],2/1024)
            self.assertEqual(contrast["exact_sign_flip_p_Holm"],6/1024)
        self.assertFalse(first["costs"]["provided"])
        self.assertFalse(first["automatic_experiment_launch"])

    def test_zero_variance_is_finite_json(self):
        zero = a.paired_t_summary([0]*10)
        positive = a.paired_t_summary([1]*10)
        self.assertEqual(zero["p_raw"],1)
        self.assertEqual(positive["p_raw"],0)
        self.assertIsNone(positive["t_statistic"])
        json.dumps([zero,positive],allow_nan=False)

    def test_t_confidence_interval_known(self):
        summary = a.paired_t_summary([-1,0,1,2,3])
        self.assertAlmostEqual(summary["mean_difference_pp"],1)
        self.assertAlmostEqual(summary["seed_difference_sd_pp"],np.sqrt(2.5))
        self.assertAlmostEqual(summary["ci95_unadjusted_pp"][0],1-2.7764451051977987*np.sqrt(.5))
        self.assertAlmostEqual(summary["ci95_unadjusted_pp"][1],1+2.7764451051977987*np.sqrt(.5))

    def test_costs_only_when_provided_and_complete(self):
        protocol,sha,data=fixture()
        data["costs"]={"scope":"All four stages; measured training only, excludes evaluation and tuning",
                       "records":[{"method":m,"seed":s,"training_seconds":110 if m=="ratio_dcr" else 100}
                                  for m,s in itertools.product(a.METHODS,protocol["confirmation_seeds"])]}
        result=a.analyze(data,protocol,sha)
        self.assertTrue(result["costs"]["descriptive_only"])
        self.assertAlmostEqual(result["costs"]["fields"]["training_seconds"]["ratio_dcr_over_controls"]["ce"]["mean_paired_overhead_percent"],10)
        del data["costs"]["records"][-1]
        with self.assertRaises(a.ContractError):a.analyze(data,protocol,sha)

    def test_protocol_actual_bytes_bound_by_file_api(self):
        protocol,sha,data=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'locked.json';f=Path(tmp)/'records.json'
            p.write_text(json.dumps(protocol,sort_keys=True));f.write_text(json.dumps(data))
            self.assertEqual(a.read_and_analyze(f,p)["n_paired_seeds"],10)
            p.write_text(p.read_text()+'\n')
            with self.assertRaises(a.ContractError):a.read_and_analyze(f,p)


class RefusalTests(unittest.TestCase):
    def setUp(self):self.protocol,self.sha,self.data=fixture()
    def refused(self):
        with self.assertRaises(a.ContractError):a.analyze(self.data,self.protocol,self.sha)

    def test_engineering_or_development_root_refused(self):
        for stage in ('engineering_pilot','development'):
            with self.subTest(stage=stage):
                self.data['stage']=stage;self.refused()

    def test_partial_root_or_record_refused(self):
        self.data['status']='RUNNING';self.refused()
        self.data['status']='COMPLETED_VERIFIED';self.data['records'][0]['status']='PARTIAL';self.refused()

    def test_mixed_stage_record_refused(self):
        self.data['records'][0]['stage']='development';self.refused()

    def test_missing_budget_refused(self):
        del self.data['records'][0];self.refused()

    def test_duplicate_budget_refused(self):
        self.data['records'].append(copy.deepcopy(self.data['records'][0]));self.refused()

    def test_unmatched_seed_refused(self):
        self.data['records'][0]['seed']=99999;self.refused()

    def test_zero_or_bool_seed_refused(self):
        for seed in (0,-1,True):
            with self.subTest(seed=seed):self.data['records'][0]['seed']=seed;self.refused()

    def test_unlocked_protocol_refused(self):
        self.protocol['locked']=False;self.refused()

    def test_wrong_source_or_protocol_binding_refused(self):
        self.data['source_binding_sha256']='c'*64;self.refused()
        self.data['source_binding_sha256']='a'*64;self.data['protocol_sha256']='c'*64;self.refused()

    def test_missing_record_receipt_refused(self):
        del self.data['records'][0]['receipt_sha256'];self.refused()

    def test_bad_accuracy_and_budget_refused(self):
        for value in (float('nan'),float('inf'),-1,101,True):
            with self.subTest(value=value):self.data['records'][0]['accuracy_percent']=value;self.refused()
        self.data['records'][0]['accuracy_percent']=50;self.data['records'][0]['label_count']=5000;self.refused()

    def test_nonprespecified_methods_or_analysis_refused(self):
        self.protocol['methods'][-1]='ntce';self.refused()
        self.protocol['methods']=list(a.METHODS);self.protocol['analysis']['primary_contrasts']=[];self.refused()

    def test_single_or_duplicate_confirmation_seed_refused(self):
        self.protocol['confirmation_seeds']=[1001];self.refused()
        self.protocol['confirmation_seeds']=[1001,1001];self.refused()


    def test_unreleased_or_development_data_refused(self):
        self.data['evaluation_partition']='development';self.refused()
        self.data['evaluation_partition']='official_test';self.data['test_release_verified']=False;self.refused()
        self.data['test_release_verified']=True;self.data['records'][0]['evaluation_partition']='development';self.refused()

    def test_explicit_release_binding_refused(self):
        self.data['test_release_receipt_sha256']='d'*64;self.refused()

if __name__=='__main__':unittest.main()
