"""
test_phase5_graph.py - Protocolo de Testes para a Fase 5 (Fusão Neuro-Simbólica, Integridade e Visualizador).
"""

import sys
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from project2.src.graph_integrator import NeuroSymbolicGraphIntegrator
from project2.src.visualizer_p2 import generate_cytoscape_p2_html


def test_ontological_prior_laplace_smoothing():
    """Teste Unitário 5.1: Prior Ontológico MeSH com Suavização de Laplace."""
    integrator = NeuroSymbolicGraphIntegrator(laplace_alpha=0.5)

    # Treina com poucas instâncias conhecidas
    p1_edges = [
        ('ExamInstance', 'Diagnosis', 'REVEALS'),
        ('ExamInstance', 'Diagnosis', 'REVEALS'),
        ('SymptomObservation', 'Diagnosis', 'SUPPORTS')
    ]
    integrator.fit_ontological_prior(p1_edges)

    # Par não visto nunca no treino: DrugAdministration -> Anatomy
    unseen_prior = integrator.get_ontological_prior_dist('DrugAdministration', 'Anatomy')

    assert np.all(unseen_prior > 0.0), "Suavização de Laplace deve garantir probabilidade estritamente positiva para pares inéditos."
    assert np.allclose(np.sum(unseen_prior), 1.0, atol=1e-5), "Distribuição a priori deve somar 1.0."


def test_neuro_symbolic_calibration_suppresses_impossible_relations():
    """Teste Unitário 5.2: Calibração Bayesiana Suprime Relações Clinicamente Impossíveis."""
    integrator = NeuroSymbolicGraphIntegrator(gamma_prior=0.7)

    # Prior: Exame revela Diagnóstico
    integrator.fit_ontological_prior([
        ('ExamInstance', 'Diagnosis', 'REVEALS'),
        ('ExamInstance', 'Diagnosis', 'REVEALS'),
        ('ExamInstance', 'Diagnosis', 'REVEALS')
    ])

    # Suponha que o modelo de texto dê empate entre REVEALS e TREATED_WITH
    # Mas um exame NÃO trata um diagnóstico
    ml_probs = np.array([0.45, 0.05, 0.05, 0.05, 0.40, 0.0, 0.0, 0.0])  # REVEALS=0.45, TREATED_WITH=0.40
    calibrated = integrator.calibrate(ml_probs, 'ExamInstance', 'Diagnosis')

    assert calibrated['calibrated_relation'] == 'REVEALS', "Prior ontológico deveria favorecer REVEALS sobre TREATED_WITH."
    assert calibrated['p_final'] > calibrated['p_ml_raw'], "Probabilidade de REVEALS deveria ter aumentado após calibração."


def test_graph_fusion_referential_integrity(tmp_path):
    """Teste Unitário 5.3: Integridade Referencial do Grafo P2 (Zero Arestas Órfãs)."""
    integrator = NeuroSymbolicGraphIntegrator()

    nodes_p1 = pd.DataFrame([
        {'node_id': 'N1', 'case_id': 'C1', 'type': 'ExamInstance', 'label': 'CT'},
        {'node_id': 'N2', 'case_id': 'C1', 'type': 'Diagnosis', 'label': 'Pancreatitis'}
    ])
    edges_p1 = pd.DataFrame([
        {'edge_id': 'E1', 'case_id': 'C1', 'source_id': 'N1', 'target_id': 'N2', 'relation': 'REVEALS'}
    ])

    # Novas arestas e nós OOV
    oov_nodes = pd.DataFrame([
        {'node_id': 'N3_OOV', 'case_id': 'C1', 'type': 'Finding', 'label': 'Pseudocyst'}
    ])
    ml_candidates = pd.DataFrame([
        {'case_id': 'C1', 'e1_id': 'N1', 'e2_id': 'N3_OOV', 'calibrated_relation': 'REVEALS', 'p_final': 0.88}
    ])

    nodes_p2, edges_p2, metrics = integrator.build_enhanced_graph(
        nodes_p1, edges_p1, ml_candidates, oov_nodes, confidence_threshold=0.70
    )

    node_ids = set(nodes_p2['node_id'])
    for _, edge in edges_p2.iterrows():
        assert edge['source_id'] in node_ids, f"Source órfão: {edge['source_id']}"
        assert edge['target_id'] in node_ids, f"Target órfão: {edge['target_id']}"

    assert metrics['novel_ml_edges_added'] == 1
    assert metrics['oov_nodes_added'] == 1

    # Testa geração do visualizador Cytoscape
    html_file = tmp_path / "test_graph.html"
    generate_cytoscape_p2_html(nodes_p2, edges_p2, str(html_file))
    assert html_file.exists(), "Arquivo HTML do Cytoscape não foi gerado."
    assert html_file.stat().st_size > 1000, "Arquivo HTML muito pequeno."


if __name__ == "__main__":
    pytest.main(["-v", __file__])
