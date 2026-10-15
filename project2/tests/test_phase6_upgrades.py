"""
test_phase6_upgrades.py - Testes unitários para as 5 Inovações de Alto Impacto (Fase 6).
Verifica:
1. DistributionalClusterNgramExtractor (K-Means K=32 sobre PPMI-SVD 64d)
2. HierarchicalCascadeClassifier (Estágio 1 Link + Estágio 2 Especialista)
3. detect_p1_label_noise_oof (Confident Learning Denoising)
4. ThreeTowerBilinearMLP & ThreeTowerPipeline (Forward pass trilinear h1 * h_ctx * h2)
5. apply_collective_triangle_propagation (Fechamento de Triângulos Clínicos no Grafo)
"""

import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch

from project2.src.genius_upgrades import (
    DistributionalClusterNgramExtractor,
    HierarchicalCascadeClassifier,
    detect_p1_label_noise_oof,
    ThreeTowerBilinearMLP,
    ThreeTowerPipeline
)
from project2.src.graph_integrator import (
    apply_collective_triangle_propagation,
    CLINICAL_TRIANGLE_RULES
)


def test_cluster_ngram_extractor():
    # Vocabulário sintético e vetores PPMI-SVD
    word2idx = {
        "revealed": 0, "showed": 1, "consistent": 2, "indicative": 3,
        "with": 4, "of": 5, "the": 6, "lesion": 7
    }
    np.random.seed(42)
    svd_emb = np.random.randn(8, 16).astype(np.float32)

    extractor = DistributionalClusterNgramExtractor(n_clusters=4, random_state=42)
    extractor.fit(word2idx, svd_emb)

    assert extractor.is_fitted
    # Palavras funcionais curtas devem permanecer literais
    assert extractor.word2cluster["with"] == "with"
    assert extractor.word2cluster["of"] == "of"
    assert extractor.word2cluster["the"] == "the"

    # Palavras de conteúdo devem receber cluster
    assert extractor.word2cluster["revealed"].startswith("CLUST")

    # Extração de templates
    btw_tokens = ["was", "consistent", "with"]
    tpls = extractor.augment_with_cluster_ngrams(btw_tokens, max_n=3)
    assert len(tpls) >= 1
    assert any("BTW_TPL_" in t for t in tpls)

    # Aumento de string
    feat_str = "DIR_L2R TYPE_Exam_Diag BTW_was BTW_consistent BTW_with"
    augmented = extractor.augment_feature_string(feat_str)
    assert "BTW_TPL_" in augmented


def test_hierarchical_cascade_classifier():
    np.random.seed(42)
    # 60 amostras, 20 colunas
    X = sp.csr_matrix(np.random.rand(60, 20))
    # Classes: 30 NONE, 15 REVEALS, 15 CONFIRMS
    y = np.array(["NONE"] * 30 + ["REVEALS"] * 15 + ["CONFIRMS"] * 15)

    clf = HierarchicalCascadeClassifier(none_label="NONE", link_threshold=0.45)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (60, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert len(preds) == 60
    assert set(np.unique(preds)).issubset({"NONE", "REVEALS", "CONFIRMS"})


def test_confident_learning_denoising():
    y_silver = np.array(["SUPPORTS", "NONE", "CONFIRMS", "SUPPORTS"])
    classes = np.array(["CONFIRMS", "NONE", "REVEALS", "SUPPORTS"])
    # OOF probs indicando que o índice 0 e 3 eram falsos positivos do P1
    oof_probs = np.array([
        [0.05, 0.90, 0.02, 0.03],  # P1 disse SUPPORTS, mas modelo tem 90% NONE
        [0.10, 0.85, 0.02, 0.03],  # Concorda com NONE
        [0.88, 0.02, 0.05, 0.05],  # Concorda com CONFIRMS
        [0.02, 0.82, 0.06, 0.10],  # P1 disse SUPPORTS, mas modelo tem 82% NONE
    ])

    y_cleaned, audit_log = detect_p1_label_noise_oof(
        y_silver=y_silver,
        oof_probs=oof_probs,
        classes=classes,
        confidence_threshold=0.75
    )

    assert len(audit_log) == 2
    assert audit_log[0]["p1_noisy_label"] == "SUPPORTS"
    assert audit_log[0]["p2_confident_label"] == "NONE"
    assert y_cleaned[0] == "NONE"
    assert y_cleaned[3] == "NONE"


def test_three_tower_bilinear_mlp_forward():
    batch_size = 4
    entity_dim = 64 + 8
    ctx_dim = 32
    num_classes = 5

    model = ThreeTowerBilinearMLP(
        entity_dim=entity_dim,
        ctx_svd_dim=ctx_dim,
        num_classes=num_classes,
        hidden_dim=24,
        dropout=0.1
    )

    t_e1 = torch.randn(batch_size, entity_dim)
    t_e2 = torch.randn(batch_size, entity_dim)
    t_ctx = torch.randn(batch_size, ctx_dim)

    logits = model(t_e1, t_e2, t_ctx)
    assert logits.shape == (batch_size, num_classes)

    # Teste ThreeTowerPipeline
    pipeline = ThreeTowerPipeline(
        entity_dim=entity_dim,
        ctx_dim=ctx_dim,
        hidden_dim=24,
        epochs=3,
        batch_size=8
    )

    N = 30
    e1_mat = np.random.randn(N, entity_dim).astype(np.float32)
    e2_mat = np.random.randn(N, entity_dim).astype(np.float32)
    ctx_sp = sp.csr_matrix(np.random.rand(N, 50))
    y_labels = ["REVEALS"] * 10 + ["CONFIRMS"] * 10 + ["NONE"] * 10

    pipeline.fit(e1_mat, e2_mat, ctx_sp, y_labels)
    preds = pipeline.predict(e1_mat[:5], e2_mat[:5], ctx_sp[:5])
    assert len(preds) == 5


def test_collective_triangle_propagation():
    # Cria tríade sintética: u --REVEALS--> v (0.90) e v --SUPPORTS--> w (0.85)
    # Aresta u --CONFIRMS--> w (0.70) deve receber boost
    edges_data = [
        {"case_id": "CASE_01", "source_id": "N1", "target_id": "N2", "relation": "REVEALS", "confidence": 0.90},
        {"case_id": "CASE_01", "source_id": "N2", "target_id": "N3", "relation": "SUPPORTS", "confidence": 0.85},
        {"case_id": "CASE_01", "source_id": "N1", "target_id": "N3", "relation": "CONFIRMS", "confidence": 0.70},
    ]
    df_edges = pd.DataFrame(edges_data)

    df_refined, triangles_closed = apply_collective_triangle_propagation(df_edges, boost_damping=0.20)

    assert triangles_closed >= 1
    p_boosted = df_refined.loc[df_refined["target_id"] == "N3", "confidence"].values[0]
    assert p_boosted > 0.70


def test_typed_features_and_neuro_symbolic_ensemble():
    from project2.src.genius_upgrades import build_typed_positional_features, neuro_symbolic_ensemble_and_tune

    row = {
        "e1_type": "ExamInstance",
        "e2_type": "Diagnosis",
        "token_dist": 3
    }
    base_text = "DIR_L2R BTW_was BTW_consistent BTW_with"
    typed_text = build_typed_positional_features(row, base_text)

    assert "SIG_EXAMINSTANCE_TO_DIAGNOSIS" in typed_text
    assert "DIST_SHORT" in typed_text
    assert "SIG_EXAMINSTANCE_TO_DIAGNOSIS__BTW_was" in typed_text

    # Teste de fusão e calibração de multiplicadores
    N_tr = 40
    N_val = 10
    num_classes = 3

    probs_casc_tr = np.random.dirichlet(np.ones(num_classes), size=N_tr)
    probs_mlp_tr = np.random.dirichlet(np.ones(num_classes), size=N_tr)
    prior_onto_tr = np.random.dirichlet(np.ones(num_classes), size=N_tr)
    y_tr_idx = np.random.choice(num_classes, size=N_tr)

    probs_casc_val = np.random.dirichlet(np.ones(num_classes), size=N_val)
    probs_mlp_val = np.random.dirichlet(np.ones(num_classes), size=N_val)
    prior_onto_val = np.random.dirichlet(np.ones(num_classes), size=N_val)

    calib_val = neuro_symbolic_ensemble_and_tune(
        probs_cascade_train=probs_casc_tr,
        probs_mlp_train=probs_mlp_tr,
        prior_onto_train=prior_onto_tr,
        y_train_idx=y_tr_idx,
        probs_cascade_val=probs_casc_val,
        probs_mlp_val=probs_mlp_val,
        prior_onto_val=prior_onto_val
    )

    assert calib_val.shape == (N_val, num_classes)
    assert np.allclose(calib_val.sum(axis=1), 1.0)
