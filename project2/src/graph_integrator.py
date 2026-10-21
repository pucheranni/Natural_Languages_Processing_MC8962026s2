"""
graph_integrator.py - Fusão Neuro-Simbólica (Prior Ontológico MeSH + ML Textual) e Integração do Grafo P2.
Fundamentado em:
- Slide nlp-080: Suavização de Laplace (Add-alpha smoothing) e Calibração Bayesiana
- Ji et al. (2022): Knowledge Graph Integration & Ontological Constraints
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd


class NeuroSymbolicGraphIntegrator:
    """
    Calibrador Bayesiano Neuro-Simbólico.
    Combina a probabilidade puramente estatística do modelo de texto P_ML(R | texto)
    com a probabilidade ontológica MeSH de compatibilidade de tipos P_onto(R | Tipo_E1, Tipo_E2):
    P_final(R) = (P_ML(R) * [P_onto(R)]^gamma) / Z
    """

    def __init__(
        self,
        relations: Optional[List[str]] = None,
        gamma_prior: float = 0.5,
        laplace_alpha: float = 0.5
    ):
        self.relations = relations if relations is not None else [
            'REVEALS', 'CONFIRMS', 'EXCLUDES', 'SUPPORTS', 'TREATED_WITH', 'LOCATED_IN', 'TARGETS', 'NONE'
        ]
        self.rel2idx = {r: i for i, r in enumerate(self.relations)}
        self.gamma_prior = gamma_prior
        self.laplace_alpha = laplace_alpha
        self.type_pair_counts: Dict[Tuple[str, str], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.relations), dtype=np.float64)
        )

    def fit_ontological_prior(self, edge_tuples: List[Tuple[str, str, str]]):
        """
        Aprende a distribuição a priori P_onto a partir de tuplas (tipo_origem, tipo_destino, relacao).
        """
        for t1, t2, rel in edge_tuples:
            t1_clean = str(t1).strip()
            t2_clean = str(t2).strip()
            if rel in self.rel2idx:
                self.type_pair_counts[(t1_clean, t2_clean)][self.rel2idx[rel]] += 1.0

    def get_ontological_prior_dist(self, t1: str, t2: str) -> np.ndarray:
        """Retorna distribuição P_onto(R | t1, t2) suavizada com Laplace."""
        counts = self.type_pair_counts.get((t1, t2), np.zeros(len(self.relations), dtype=np.float64))
        smoothed = counts + self.laplace_alpha
        total = np.sum(smoothed)
        return smoothed / total if total > 0 else np.ones(len(self.relations)) / len(self.relations)

    def calibrate(
        self,
        ml_probs: np.ndarray,
        t1: str,
        t2: str,
        gamma: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calcula a probabilidade a posteriori calibrada:
        P_final(R) proporcional a P_ML(R) * [P_onto(R)]^gamma
        """
        g = gamma if gamma is not None else self.gamma_prior
        prior_dist = self.get_ontological_prior_dist(t1, t2)

        # Se ml_probs tem tamanho diferente, ajusta
        if len(ml_probs) != len(self.relations):
            # Fallback seguro
            p_ml = np.ones(len(self.relations)) / len(self.relations)
            for i, r in enumerate(self.relations):
                if i < len(ml_probs):
                    p_ml[i] = ml_probs[i]
        else:
            p_ml = ml_probs

        unnormalized = p_ml * np.power(prior_dist, g)
        z = np.sum(unnormalized)
        p_final = unnormalized / z if z > 0 else p_ml

        best_idx = int(np.argmax(p_final))
        best_rel = self.relations[best_idx]

        return {
            'calibrated_relation': best_rel,
            'p_final': round(float(p_final[best_idx]), 4),
            'p_ml_raw': round(float(p_ml[best_idx]), 4),
            'p_onto_prior': round(float(prior_dist[best_idx]), 4),
            'distribution': {self.relations[i]: round(float(p_final[i]), 4) for i in range(len(self.relations))}
        }

    def build_enhanced_graph(
        self,
        p1_nodes_df: pd.DataFrame,
        p1_edges_df: pd.DataFrame,
        ml_candidates_df: pd.DataFrame,
        oov_nodes_df: Optional[pd.DataFrame] = None,
        confidence_threshold: float = 0.70
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Constrói as tabelas finais do Grafo P2 (nodes_p2.csv e edges_p2.csv):
        1. Preserva todos os nós do P1 e adiciona novos nós OOV promovidos pelo PMI + Morfologia.
        2. Preserva todas as arestas determinísticas do P1 (confidence=1.0, source='RULE_P1').
        3. Injeta novas arestas preditas pelo ML com P_final >= confidence_threshold (source='STATISTICAL_ML').
        """
        # Nós finais
        nodes_p2 = p1_nodes_df.copy()
        if 'discovery_method' not in nodes_p2.columns:
            nodes_p2['discovery_method'] = 'DETERMINISTIC_P1'

        if oov_nodes_df is not None and not oov_nodes_df.empty:
            oov_formatted = oov_nodes_df.copy()
            if 'discovery_method' not in oov_formatted.columns:
                oov_formatted['discovery_method'] = 'STATISTICAL_PMI_MORPHOLOGY'
            nodes_p2 = pd.concat([nodes_p2, oov_formatted], ignore_index=True).drop_duplicates(subset=['node_id'])

        # Arestas finais
        edges_list = []
        existing_pairs = set()

        # 1. Arestas do P1
        for _, row in p1_edges_df.iterrows():
            cid = str(row['case_id'])
            s_id = str(row['source_id'])
            t_id = str(row['target_id'])
            rel = str(row['relation'])
            existing_pairs.add((cid, s_id, t_id))

            attrs = row.get('attributes', {})
            edges_list.append({
                'edge_id': str(row['edge_id']),
                'case_id': cid,
                'source_id': s_id,
                'target_id': t_id,
                'relation': rel,
                'confidence': 1.0,
                'source_method': 'RULE_P1',
                'explanation_xai': '',
                'attributes': attrs
            })

        # 2. Novas Arestas Inéditas Descobertas pelo ML
        new_edge_counter = len(edges_list) + 1
        new_edges_added = 0

        for _, row in ml_candidates_df.iterrows():
            cid = str(row['case_id'])
            s_id = str(row['e1_id'])
            t_id = str(row['e2_id'])
            pred_rel = str(row['calibrated_relation'])
            conf = float(row.get('p_final', row.get('confidence', 0.0)))

            # Ignora NONE e arestas abaixo do threshold de confiança
            if pred_rel == 'NONE' or conf < confidence_threshold:
                continue

            # Evita duplicatas com P1
            if (cid, s_id, t_id) in existing_pairs or (cid, t_id, s_id) in existing_pairs:
                continue

            edge_id = f"E_{cid}_ML_{new_edge_counter:04d}"
            new_edge_counter += 1
            new_edges_added += 1
            existing_pairs.add((cid, s_id, t_id))

            edges_list.append({
                'edge_id': edge_id,
                'case_id': cid,
                'source_id': s_id,
                'target_id': t_id,
                'relation': pred_rel,
                'confidence': round(conf, 4),
                'source_method': 'STATISTICAL_ML',
                'explanation_xai': str(row.get('top3_pos_ngrams', '')),
                'attributes': {
                    'p_ml_raw': row.get('p_ml_raw', conf),
                    'p_onto_prior': row.get('p_onto_prior', 1.0),
                    'discovery': 'NOVEL_EDGE_DISCOVERY'
                }
            })

        edges_p2 = pd.DataFrame(edges_list)
        edges_p2, triangles_closed = apply_collective_triangle_propagation(edges_p2, boost_damping=0.15)

        # Métricas topológicas comparativas (densidade direcionada e não-direcionada)
        v_p1 = len(p1_nodes_df)
        v_total = len(nodes_p2)
        e_total = len(edges_p2)
        e_p1 = len(p1_edges_df)

        density_dir_p1 = float(e_p1) / (v_p1 * (v_p1 - 1)) if v_p1 > 1 else 0.0
        density_dir_p2 = float(e_total) / (v_total * (v_total - 1)) if v_total > 1 else 0.0
        density_undir_p1 = (2.0 * e_p1) / (v_p1 * (v_p1 - 1)) if v_p1 > 1 else 0.0
        density_undir_p2 = (2.0 * e_total) / (v_total * (v_total - 1)) if v_total > 1 else 0.0

        metrics = {
            'total_nodes_p1': v_p1,
            'total_nodes_p2': v_total,
            'oov_nodes_added': v_total - v_p1,
            'total_edges_p1': e_p1,
            'total_edges_p2': e_total,
            'novel_ml_edges_added': new_edges_added,
            'relative_edge_gain_pct': round((new_edges_added / max(1, e_p1)) * 100, 2),
            'graph_density_directed_p1': round(density_dir_p1, 6),
            'graph_density_directed_p2': round(density_dir_p2, 6),
            'graph_density_undirected_p1': round(density_undir_p1, 6),
            'graph_density_undirected_p2': round(density_undir_p2, 6),
            'graph_density_p2': round(density_undir_p2, 6),
            'confidence_threshold': confidence_threshold,
            'triangles_closed': triangles_closed
        }

        return nodes_p2, edges_p2, metrics


# Regras de Composição de Caminhos de 2 Saltos (Cláusulas de Horn Clínicas):
CLINICAL_TRIANGLE_RULES = {
    # Cadeias Clínico-Diagnósticas e Terapêuticas Reais do MultiCaRe:
    ("UNDERWENT_TEST", "SUPPORTS"): ("DIAGNOSED_WITH", 1.25),
    ("DIAGNOSED_WITH", "LOCATED_IN"): ("ASSOCIATED_WITH", 1.20),
    ("TREATED_WITH", "TARGETS"): ("DIAGNOSED_WITH", 1.30),
    ("PRESENTS_WITH", "SUPPORTS"): ("DIAGNOSED_WITH", 1.25),
    ("UNDERWENT_TEST", "REVEALS"): ("PRESENTS_WITH", 1.20),
    ("REVEALS", "SUPPORTS"): ("CONFIRMS", 1.25),
    ("CONFIRMS", "LOCATED_IN"): ("LOCATED_IN", 1.20),
    ("REVEALS", "LOCATED_IN"): ("REVEALS", 1.15),
    ("TREATED_WITH", "TARGETS"): ("TREATED_WITH", 1.20),
}


def apply_collective_triangle_propagation(edges_df: pd.DataFrame, boost_damping: float = 0.15) -> Tuple[pd.DataFrame, int]:
    """
    Refina iterativamente os scores de confiança das arestas usando fechamento de triângulos
    intra-caso (Relational Belief Propagation) e consistência semântica clínica.
    """
    df = edges_df.copy()
    src_col = 'source_id' if 'source_id' in df.columns else 'source'
    tgt_col = 'target_id' if 'target_id' in df.columns else 'target'

    # Indexa arestas por (case_id, source, target)
    edge_lookup = {}
    for idx, row in df.iterrows():
        key = (str(row["case_id"]), str(row[src_col]), str(row[tgt_col]))
        edge_lookup[key] = idx

    triangles_closed = 0
    for (case_id, u, v), idx_uv in list(edge_lookup.items()):
        r1 = df.loc[idx_uv, "relation"]
        p1 = float(df.loc[idx_uv, "confidence"])
        if p1 < 0.60:
            continue

        # Procura terceiro nó w no mesmo caso formando tríade (u -> v -> w)
        case_edges = df[(df["case_id"] == case_id) & (df[src_col] == v)]
        for _, row_vw in case_edges.iterrows():
            w = str(row_vw[tgt_col])
            r2 = row_vw["relation"]
            p2 = float(row_vw["confidence"])

            if (r1, r2) in CLINICAL_TRIANGLE_RULES and (case_id, u, w) in edge_lookup:
                expected_r3, multiplier = CLINICAL_TRIANGLE_RULES[(r1, r2)]
                idx_uw = edge_lookup[(case_id, u, w)]
                if df.loc[idx_uw, "relation"] == expected_r3:
                    old_p = float(df.loc[idx_uw, "confidence"])
                    # Atualização suave limitada a [0, 0.99]
                    new_p = min(0.99, old_p + boost_damping * (p1 * p2) * (multiplier - 1.0))
                    df.loc[idx_uw, "confidence"] = round(float(new_p), 4)
                    triangles_closed += 1

    return df, triangles_closed
