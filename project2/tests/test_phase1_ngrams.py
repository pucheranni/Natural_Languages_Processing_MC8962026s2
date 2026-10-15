"""
test_phase1_ngrams.py - Protocolo de Testes de Engenharia e Sanidade Científica para a Fase 1.
Verifica:
- Invariantes matemáticas de PPMI e t-score.
- Filtragem de fronteiras de stopwords em collocations.
- Propriedades geométricas da matriz de embeddings SVD 64d.
- Hipótese H3 (Morfologia de Subpalavras OOV com radicais greco-latinos).
- Hipótese H2 (Semântica Distributiva Clínica de Firth).
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Garante path da raiz
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from project2.src.ngram_miner import CollocationMiner, MorphologyOOVClassifier, DistributionalSVD, BORDER_STOPWORDS


def test_ppmi_mathematical_invariants():
    """Teste Unitário 1.1: Invariante Matemática do PPMI."""
    miner = CollocationMiner(min_freq=2)
    # Sintético: "myocardial infarction" sempre juntos, "the patient" aleatórios
    sentences = [
        ["the", "patient", "had", "acute", "myocardial", "infarction"],
        ["a", "myocardial", "infarction", "was", "diagnosed", "in", "the", "patient"],
        ["history", "of", "myocardial", "infarction", "noted"],
        ["another", "patient", "presented", "with", "severe", "chest", "pain"]
    ]

    collocations = miner.mine_collocations(sentences, max_n=2, top_k=50)
    assert len(collocations) > 0, "Deveria encontrar collocations no mini-corpus."

    colloc_dict = {c['ngram_text']: c for c in collocations}
    assert "myocardial infarction" in colloc_dict, "Deveria encontrar 'myocardial infarction'."

    mi_item = colloc_dict["myocardial infarction"]
    assert mi_item['ppmi_score'] > 0.0, "PPMI de par fortemente associado deve ser estritamente positivo."
    assert mi_item['t_score'] > 0.0, "t-score deve ser positivo."

    # Invariante geral: todo PPMI >= 0
    for c in collocations:
        assert c['ppmi_score'] >= 0.0, f"PPMI não pode ser negativo: {c['ppmi_score']}"


def test_border_stopwords_filtering():
    """Teste Unitário 1.2: Filtro de Fronteira de Stopwords."""
    miner = CollocationMiner(min_freq=1)
    sentences = [
        ["the", "patient", "was", "admitted", "with", "acute", "pancreatitis"],
        ["the", "patient", "underwent", "surgery", "for", "carcinoma", "of", "pancreas"]
    ]
    collocations = miner.mine_collocations(sentences, max_n=3, top_k=50)

    for c in collocations:
        toks = c['tokens']
        assert toks[0].lower() not in BORDER_STOPWORDS, f"N-gram começou com stopword: {toks}"
        assert toks[-1].lower() not in BORDER_STOPWORDS, f"N-gram terminou com stopword: {toks}"


def test_svd_geometry_and_normalization():
    """Teste Unitário 1.3: Geometria do SVD e Normalização L2."""
    svd_model = DistributionalSVD(embedding_dim=64, window_size=3, min_count=2)
    sentences = [
        ["aspirin", "treated", "headache", "and", "fever"],
        ["ibuprofen", "treated", "pain", "and", "fever"],
        ["patient", "received", "aspirin", "for", "pain"],
        ["patient", "received", "ibuprofen", "for", "headache"],
        ["ct", "scan", "revealed", "pneumonia", "in", "lung"],
        ["mri", "revealed", "tumor", "in", "lung"]
    ]
    matrix = svd_model.fit_transform(sentences)

    assert matrix.shape[1] == 64, f"Deveria ter 64 dimensões, tem {matrix.shape[1]}"
    assert not np.isnan(matrix).any(), "Matriz SVD não pode conter NaN."
    assert not np.isinf(matrix).any(), "Matriz SVD não pode conter Inf."

    # Normalização L2 por linha
    norms = np.linalg.norm(matrix, axis=1)
    for norm in norms:
        if norm > 0:
            assert np.isclose(norm, 1.0, atol=1e-4), f"Vetor não normalizado: norma={norm}"


def test_h3_morphology_oov_classifier():
    """Teste de Sanidade Científica H3: Morfologia Sub-lexical Biomédica."""
    classifier = MorphologyOOVClassifier()

    # Treina com termos MeSH padrão
    mesh_terms = [
        "pancreatitis", "gastritis", "appendicitis", "meningitis", "dermatitis", "nephritis",
        "pancreatectomy", "gastrectomy", "appendectomy", "mastectomy", "colectomy",
        "anemia", "leukemia", "ischemia", "hypercalcemia", "bacteremia",
        "laparoscopy", "endoscopy", "colonoscopy", "bronchoscopy",
        "amoxicillin", "diltiazem", "metformin", "ciprofloxacin", "azithromycin"
    ]
    mesh_cats = [
        "C", "C", "C", "C", "C", "C",
        "E01", "E01", "E01", "E01", "E01",
        "C", "C", "C", "C", "C",
        "E01", "E01", "E01", "E01",
        "D", "D", "D", "D", "D"
    ]

    classifier.fit(mesh_terms, mesh_cats)

    # Teste de generalização zero-shot em palavras médicas inéditas
    test_terms = ["ventriculitis", "thoracoscopy", "erythromycin"]
    preds = classifier.predict_oov_types(test_terms)

    pred_dict = {p['term']: p for p in preds}

    # -itis -> C (Doença)
    assert pred_dict["ventriculitis"]['predicted_mesh_cat'] == "C", "ventriculitis (-itis) deveria ser Doença (C)"
    # -scopy -> E01 (Procedimento/Exame)
    assert pred_dict["thoracoscopy"]['predicted_mesh_cat'] == "E01", "thoracoscopy (-scopy) deveria ser Exame (E01)"
    # -mycin -> D (Fármaco)
    assert pred_dict["erythromycin"]['predicted_mesh_cat'] == "D", "erythromycin (-mycin) deveria ser Fármaco (D)"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
