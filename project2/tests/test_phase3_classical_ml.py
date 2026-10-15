"""
test_phase3_classical_ml.py - Protocolo de Testes para a Fase 3 (Modelos Clássicos, Simplex de Probabilidades e XAI).
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from project2.src.relation_classifier import ClinicalRelationClassifier, run_cross_validation_ablation


@pytest.fixture
def sample_dataset():
    """Gera mini dataset sintético balanceado com 5 folds."""
    rows = []
    classes = ['REVEALS', 'SUPPORTS', 'TREATED_WITH', 'NONE']
    for i in range(40):
        c = classes[i % len(classes)]
        fold = i % 5
        cid = f"CASE_{fold:02d}"
        if c == 'REVEALS':
            text = "DIR_L2R TYPE_Exam_Diagnosis PRE_patient BTW_ct_scan BTW_revealed BTW_tumor POST_lung"
        elif c == 'SUPPORTS':
            text = "DIR_L2R TYPE_Finding_Diagnosis PRE_exam BTW_finding BTW_consistent_with BTW_infection POST_confirmed"
        elif c == 'TREATED_WITH':
            text = "DIR_L2R TYPE_Patient_Drug PRE_admission BTW_treated_with BTW_antibiotic POST_resolved"
        else:
            text = "DIR_L2R TYPE_Entity_Entity PRE_unrelated BTW_and BTW_also POST_noted"
        rows.append({
            'pair_id': f"P_{i:03d}",
            'case_id': cid,
            'masked_positional_tokens': text,
            'silver_label': c,
            'fold': fold
        })
    return pd.DataFrame(rows)


def test_probability_simplex_all_models(sample_dataset):
    """Teste Unitário 3.1: Invariante Matemática - Simplex de Probabilidades."""
    texts = sample_dataset['masked_positional_tokens'].tolist()
    labels = sample_dataset['silver_label'].tolist()

    models = ['naive_bayes', 'logistic_regression', 'linear_svm', 'random_forest']

    for m in models:
        clf = ClinicalRelationClassifier(model_name=m, ngram_range=(1, 2))
        clf.fit(texts, labels)

        probs = clf.predict_proba(texts)
        assert probs.shape == (len(texts), len(clf.classes_)), f"Shape incorreto para {m}"
        assert np.all(probs >= -1e-7), f"Probabilidades negativas detectadas em {m}"
        assert np.all(probs <= 1.0 + 1e-7), f"Probabilidades maiores que 1 detectadas em {m}"

        row_sums = np.sum(probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Linha de probabilidade não soma 1.0 em {m}"


def test_xai_linear_mathematical_fidelity(sample_dataset):
    """Teste Unitário 3.2: Fidelidade Matemática do XAI Linear."""
    texts = sample_dataset['masked_positional_tokens'].tolist()
    labels = sample_dataset['silver_label'].tolist()

    clf = ClinicalRelationClassifier(model_name='logistic_regression', ngram_range=(1, 2))
    clf.fit(texts, labels)

    sample_text = texts[0]
    explanation = clf.explain_prediction(sample_text)

    assert 'predicted_class' in explanation
    assert 'top_positive' in explanation
    assert 'top_negative' in explanation
    assert len(explanation['top_positive']) <= 3

    # Verifica se os scores contribuídos são consistentes com a predição
    if explanation['top_positive']:
        assert explanation['top_positive'][0]['contribution'] > 0


def test_cross_validation_ablation_execution(sample_dataset):
    """Teste de Integração 3.3: Execução de Ablação 5-Fold."""
    results = run_cross_validation_ablation(
        sample_dataset,
        feature_col='masked_positional_tokens',
        target_col='silver_label',
        fold_col='fold',
        model_name='logistic_regression',
        ngram_range=(1, 2)
    )

    assert 'macro_f1' in results
    assert 'confusion_matrix' in results
    assert 0.0 <= results['macro_f1'] <= 1.0
    assert len(results['classes']) == 4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
