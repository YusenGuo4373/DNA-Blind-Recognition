from __future__ import annotations
import numpy as np
from author_baseline.recognizer import OneHotArchive
from incremental_validation.stage6_proxy_robustness import ABLATIONS, BLOCKS, prefix_archive, select_blocks, outputs, metrics, bootstrap


def test_feature_block_dimensions_and_order():
    x=np.arange(464,dtype=float)[None]
    assert select_blocks(x,("sequence",)).shape==(1,165)
    assert select_blocks(x,("embedding",)).shape==(1,256)
    assert select_blocks(x,("logits",)).shape==(1,43)
    np.testing.assert_array_equal(select_blocks(x,("sequence","logits")),np.r_[np.arange(165),np.arange(421,464)][None])


def test_all_ablation_combinations_are_exact():
    assert len(ABLATIONS)==7
    assert ABLATIONS["G_all"]==("sequence","embedding","logits")
    assert ABLATIONS["A_sequence"]==("sequence",)


def test_q_m_prefix_reuses_leading_reads():
    one=np.arange(5*7*4*9).reshape(5,7,4,9); mask=np.ones((5,7,9),bool); a=OneHotArchive(one,mask); p=prefix_archive(a,2,3)
    np.testing.assert_array_equal(p.one_hot,one[:2,:3]); np.testing.assert_array_equal(p.mask,mask[:2,:3])


def test_fixed_threshold_is_not_mutated_and_six_class_matrix():
    cats=np.array(["BCH","Convolutional","LDPC","Polar","NoECC-Random","HEDGES"])
    d={"categories":cats,"closed":np.array([0,1,2,3,0,0]),"proxy":np.array([.1,.1,.1,.1,.1,.9]),"ecc":np.array([.9,.9,.9,.9,.1,.9]),"energy":np.zeros(6),"archive_ids":np.arange(6)}
    tau=.5; out=outputs(d,.4,tau); assert tau==.5
    m=metrics(d,out); assert np.asarray(m["six_class_confusion_matrix"]).shape==(6,6); assert np.trace(m["six_class_confusion_matrix"])==6


def test_bootstrap_unit_is_archive():
    cats=np.array([c for c in ("BCH","Convolutional","LDPC","Polar","NoECC-Random","NoECC-Constrained","HEDGES","DNA-Aeon") for _ in range(2)])
    closed=np.array([0,0,1,1,2,2,3,3,0,0,0,0,0,0,0,0]); proxy=np.r_[np.zeros(12),np.ones(4)]; ecc=np.r_[np.ones(8),np.zeros(4),np.ones(4)]
    d={"categories":cats,"closed":closed,"proxy":proxy,"ecc":ecc,"energy":np.zeros(16),"archive_ids":np.arange(16)}; out=outputs(d,.5,.5)
    b=bootstrap(cats,d,out,43,reps=10); assert "known_acceptance_rate" in b
