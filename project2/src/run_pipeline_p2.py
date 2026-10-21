"""
run_pipeline_p2.py - Orquestrador de Execução Ponta-a-Ponta do Projeto 2 (PLN 2026).
Executa as 6 fases da arquitetura Neuro-Simbólica com as 5 Inovações de Alto Impacto:
1. Mineração de Collocations PPMI, SVD 64d, K-Means (K=32) e Tipagem Morfológica OOV.
2. Geração do Silver Standard com Trigger-Masking e N-grams Posicionais + Templates de Clusters.
3. Ato 1 (A Descoberta do Limite Clássico): Torneio de Modelos Clássicos & Anomalia do 0.5040.
4. Ato 2 (As 5 Inovações Científicas): Clusters Abstratos, Cascata Hierárquica, MLP Trilinear de 3 Torres e Confident Learning.
5. Inferência Coletiva por Fechamento de Triângulos Clínicos no Grafo.
6. Exportação do Grafo P2 (nodes_p2.csv, edges_p2.csv) e Visualizador Cytoscape com Sandbox Contrafactual.
"""

import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Garante path da raiz
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from project2.src.ngram_miner import CollocationMiner, MorphologyOOVClassifier, DistributionalSVD
from project2.src.weak_supervisor import WeakSupervisionDatasetGenerator
from project2.src.relation_classifier import ClinicalRelationClassifier, run_cross_validation_ablation
from project2.src.mlp_classifier import ClinicalNeuralPipeline
from project2.src.graph_integrator import NeuroSymbolicGraphIntegrator, apply_collective_triangle_propagation
from project2.src.visualizer_p2 import generate_cytoscape_p2_html
from project2.src.genius_upgrades import (
    DistributionalClusterNgramExtractor,
    HierarchicalCascadeClassifier,
    detect_p1_label_noise_oof,
    ThreeTowerPipeline,
    build_typed_positional_features,
    neuro_symbolic_ensemble_and_tune
)


def main():
    print("=" * 75)
    print("INICIANDO EXECUÇÃO COMPLETA DO PIPELINE PROJETO 2 (PLN 2026)")
    print("=" * 75)

    # Diretórios
    proj2_dir = Path(__file__).resolve().parent.parent
    data_raw_dir = proj2_dir / 'data' / 'raw'
    data_interim_dir = proj2_dir / 'data' / 'interim'
    data_processed_dir = proj2_dir / 'data' / 'processed'
    assets_dir = proj2_dir / 'assets'

    data_interim_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Carrega dados do P1
    cases_file = proj2_dir.parent / 'project1' / 'data' / 'raw' / 'cases.csv'
    nodes_p1_file = proj2_dir.parent / 'project1' / 'data' / 'output' / 'nodes.csv'
    edges_p1_file = proj2_dir.parent / 'project1' / 'data' / 'output' / 'edges.csv'

    print(f"\n[1/6] Carregando dados clínicos do MultiCaRe...")
    cases_df = pd.read_csv(cases_file)
    nodes_p1_df = pd.read_csv(nodes_p1_file)
    edges_p1_df = pd.read_csv(edges_p1_file)
    print(f"  -> Casos clínicos carregados: {len(cases_df)}")
    print(f"  -> Nós do P1 carregados: {len(nodes_p1_df)}")
    print(f"  -> Arestas do P1 carregadas: {len(edges_p1_df)}")

    # -------------------------------------------------------------------------
    # FASE 1: Motor Estatístico de N-grams, PPMI, SVD e Morfologia OOV
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Fase 1: Mineração Estatística de N-Grams e Embeddings PPMI-SVD...")
    tokenized_sentences = []
    for text in cases_df['case_text']:
        sents = re.split(r'[.!?]+\s+', text)
        for s in sents:
            tokens = re.findall(r'\b[\w\-]+\b', s)
            if len(tokens) >= 2:
                tokenized_sentences.append(tokens)

    # Mineração de Collocations
    miner = CollocationMiner(min_freq=3)
    collocations = miner.mine_collocations(tokenized_sentences, max_n=3, top_k=250)
    collocations_df = pd.DataFrame(collocations)
    colloc_path = data_interim_dir / 'collocations_pmi.csv'
    collocations_df.to_csv(colloc_path, index=False)
    print(f"  -> Collocations mineradas e salvas em: {colloc_path}")
    print(f"  -> Top-5 N-grams por PMI & t-score: {collocations_df['ngram_text'].head(5).tolist()}")

    # Embeddings Densos SVD 64d From Scratch
    svd_model = DistributionalSVD(embedding_dim=64, window_size=5, min_count=2)
    svd_matrix = svd_model.fit_transform(tokenized_sentences)
    svd_path = data_interim_dir / 'embeddings_ppmi_svd.npz'
    svd_model.save(str(svd_path))
    print(f"  -> Matriz SVD 64d gerada: shape={svd_matrix.shape}, salva em {svd_path}")

    # Tipagem Morfológica de Nós OOV
    mesh_nodes = nodes_p1_df[nodes_p1_df['type'].isin(['Diagnosis', 'DrugAdministration', 'ExamInstance', 'Anatomy'])].dropna(subset=['label'])
    type_map = {'Diagnosis': 'C', 'DrugAdministration': 'D', 'ExamInstance': 'E01', 'Anatomy': 'A'}
    train_terms = mesh_nodes['label'].tolist()
    train_cats = [type_map[t] for t in mesh_nodes['type'].tolist()]

    morph_classifier = MorphologyOOVClassifier()
    morph_classifier.fit(train_terms, train_cats)

    # Descobre termos OOV que não constavam no P1
    p1_terms_set = set(t.lower() for t in train_terms)
    candidate_oov = [c['ngram_text'] for c in collocations if c['ngram_text'].lower() not in p1_terms_set]

    oov_preds = morph_classifier.predict_oov_types(candidate_oov[:60])
    promoted_oov = []
    inv_type_map = {'C': 'Diagnosis', 'D': 'DrugAdministration', 'E01': 'ExamInstance', 'A': 'Anatomy'}

    for idx, pred in enumerate(oov_preds):
        if pred['cat_confidence'] >= 0.65:
            promoted_oov.append({
                'node_id': f"NODE_OOV_{idx+1:04d}",
                'case_id': 'PMC5137649_01',  # Associado ao caso base para visualização
                'type': inv_type_map[pred['predicted_mesh_cat']],
                'label': pred['term'],
                'discovery_method': 'STATISTICAL_PMI_MORPHOLOGY',
                'attributes': json.dumps({'cat_confidence': pred['cat_confidence'], 'predicted_mesh_cat': pred['predicted_mesh_cat']})
            })

    oov_nodes_df = pd.DataFrame(promoted_oov)
    oov_path = data_interim_dir / 'oov_promoted_nodes.csv'
    oov_nodes_df.to_csv(oov_path, index=False)
    print(f"  -> Nós OOV promovidos via Morfologia de N-grams: {len(oov_nodes_df)} salvos em {oov_path}")

    # Treina Extrator de Clusters Abstratos (K=32 sobre PPMI-SVD)
    print(f"\n  -> Treinando Clusters Semânticos PPMI-SVD (K=32)...")
    cluster_extractor = DistributionalClusterNgramExtractor(n_clusters=32, random_state=42)
    cluster_extractor.fit(svd_model.word2idx, svd_model.matrix_64d)
    print(f"  -> Extrator de Clusters inicializado com {len(cluster_extractor.word2cluster)} palavras mapeadas.")

    # -------------------------------------------------------------------------
    # FASE 2: Supervisão Fraca com Trigger-Masking e N-grams Posicionais
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Fase 2: Geração do Dataset Silver Standard com Trigger-Masking...")
    dataset_gen = WeakSupervisionDatasetGenerator(negative_ratio=2.0, mask_prob=0.85, seed=42)
    relation_df = dataset_gen.build_dataset(cases_df, nodes_p1_df, edges_p1_df, oov_nodes_df)

    # Injeta N-grams Abstratos de Clusters Semânticos
    print(f"  -> Anexando N-Grams Abstratos de Clusters Semânticos (BTW_TPL_...) às features...")
    relation_df['masked_pos_and_clusters'] = relation_df['masked_positional_tokens'].apply(
        cluster_extractor.augment_feature_string
    )

    dataset_path = data_interim_dir / 'relation_dataset.csv'
    relation_df.to_csv(dataset_path, index=False)
    print(f"  -> Dataset Silver Standard construído: {len(relation_df)} instâncias salvas em {dataset_path}")
    print(f"  -> Distribuição de Classes:\n{relation_df['silver_label'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # FASE 3 & 4: Torneio de Modelos em 2 Atos Científicos
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Fases 3 & 4: Torneio de Modelos em 2 Atos (Diagnóstico Forense & Desbloqueio)...")
    ablation_results = []

    # =========================================================================
    # ATO 1: O Muro do 0.5040 (Modelos Clássicos & Degradação dos Trigramas)
    # =========================================================================
    print(f"\n  --- ATO 1: O Limite dos Modelos Clássicos (Baseline) ---")
    act1_experiments = [
        ("P2.1_NaiveBayes_1to3grams", "naive_bayes", (1, 3), "tfidf", "masked_positional_tokens", "silver_label"),
        ("P2.2_LogisticRegression_1gram", "logistic_regression", (1, 1), "tfidf", "masked_positional_tokens", "silver_label"),
        ("P2.3_LogisticRegression_1to2grams", "logistic_regression", (1, 2), "tfidf", "masked_positional_tokens", "silver_label"),
        ("P2.4_LogisticRegression_1to3grams", "logistic_regression", (1, 3), "tfidf", "masked_positional_tokens", "silver_label"),
        ("P2.5_LogisticRegression_Count_Weighting", "logistic_regression", (1, 3), "count", "masked_positional_tokens", "silver_label"),
        ("P2.6_LinearSVM_1to3grams", "linear_svm", (1, 3), "tfidf", "masked_positional_tokens", "silver_label"),
        ("P2.7_RandomForest_1to3grams", "random_forest", (1, 3), "tfidf", "masked_positional_tokens", "silver_label"),
    ]

    for exp_id, model_name, ngram_range, weighting, feat_col, tgt_col in act1_experiments:
        print(f"  -> Executando {exp_id}...")
        res = run_cross_validation_ablation(
            relation_df,
            feature_col=feat_col,
            target_col=tgt_col,
            model_name=model_name,
            ngram_range=ngram_range,
            weighting=weighting
        )
        res['experiment_id'] = exp_id
        res['act'] = 'Ato 1 (Baseline)'
        ablation_results.append(res)
        print(f"     Macro-F1: {res['macro_f1']:.4f} | Precision: {res['macro_precision']:.4f} | Recall: {res['macro_recall']:.4f}")

    # P2.8: MLP 2-Camadas Tradicional (Comprovando superparametrização e perda para LR)
    print(f"  -> Executando P2.8_Neural_MLP_2Layer (MLP Tradicional D_in x 128)...")
    all_classes_silver = np.sort(relation_df['silver_label'].unique())
    pos_classes_silver = [c for c in all_classes_silver if c != 'NONE']

    mlp_pipeline = ClinicalNeuralPipeline(input_mode='tfidf_sparse', epochs=20, batch_size=32)
    mlp_y_true, mlp_y_pred = [], []
    mlp_fold_f1, mlp_fold_pos_f1 = [], []
    for f in sorted(relation_df['fold'].unique()):
        tr = relation_df[relation_df['fold'] != f]
        val = relation_df[relation_df['fold'] == f]
        mlp_pipeline.fit(tr['masked_positional_tokens'].tolist(), tr['silver_label'].tolist())
        preds = mlp_pipeline.predict(val['masked_positional_tokens'].tolist())
        y_val_f = val['silver_label'].tolist()
        mlp_y_true.extend(y_val_f)
        mlp_y_pred.extend(preds)
        _, _, f1_m_f, _ = precision_recall_fscore_support(y_val_f, preds, labels=all_classes_silver, average='macro', zero_division=0)
        _, _, f1_p_f, _ = precision_recall_fscore_support(y_val_f, preds, labels=pos_classes_silver, average='macro', zero_division=0)
        mlp_fold_f1.append(float(f1_m_f))
        mlp_fold_pos_f1.append(float(f1_p_f))

    mlp_p, mlp_r, mlp_f1, _ = precision_recall_fscore_support(mlp_y_true, mlp_y_pred, labels=all_classes_silver, average='macro', zero_division=0)
    _, _, mlp_pos_f1, _ = precision_recall_fscore_support(mlp_y_true, mlp_y_pred, labels=pos_classes_silver, average='macro', zero_division=0)
    ablation_results.append({
        'experiment_id': 'P2.8_Neural_MLP_2Layer',
        'act': 'Ato 1 (Baseline)',
        'model_name': 'two_layer_mlp',
        'target_col': 'silver_label',
        'ngram_range': [1, 3],
        'weighting': 'tfidf',
        'macro_precision': round(float(mlp_p), 4),
        'macro_recall': round(float(mlp_r), 4),
        'macro_f1': round(float(mlp_f1), 4),
        'macro_f1_std': round(float(np.std(mlp_fold_f1)), 4),
        'positive_f1': round(float(mlp_pos_f1), 4),
        'positive_f1_std': round(float(np.std(mlp_fold_pos_f1)), 4)
    })
    print(f"     MLP 2-Layer Macro-F1: {mlp_f1:.4f} (±{np.std(mlp_fold_f1):.4f}) | Pos-F1: {mlp_pos_f1:.4f}")

    # =========================================================================
    # ATO 2: As 5 Viradas de Jogo BioNLP (Rompendo o Muro do 0.50)
    # =========================================================================
    print(f"\n  --- ATO 2: As Inovações Científicas (Rompendo o Muro do 0.50) ---")

    # P2.9: N-Grams Abstratos de Clusters PPMI-SVD
    print(f"  -> Executando P2.9_ClusterNgrams_1to3grams_LR (Templates de Clusters Semânticos)...")
    res_clust = run_cross_validation_ablation(
        relation_df,
        feature_col='masked_pos_and_clusters',
        target_col='silver_label',
        model_name='logistic_regression',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_clust['experiment_id'] = 'P2.9_ClusterNgrams_1to3grams_LR'
    res_clust['act'] = 'Ato 2 (Breakthrough)'
    ablation_results.append(res_clust)
    print(f"     Cluster N-Grams (Transdutivo) Macro-F1: {res_clust['macro_f1']:.4f} (±{res_clust['macro_f1_std']:.4f})")

    # Verificação Estritamente Indutiva (Refit de PPMI-SVD + KMeans dentro de cada fold de treino)
    print(f"  -> Executando Auditoria Indutiva (PPMI-SVD 64d + KMeans K=32 ajustados estritamente por fold de treino)...")
    inductive_features = pd.Series(index=relation_df.index, dtype=str)
    case_texts_map = dict(zip(cases_df['case_id'].astype(str), cases_df['case_text'].astype(str)))
    for f in sorted(relation_df['fold'].unique()):
        tr_mask = (relation_df['fold'] != f)
        val_mask = (relation_df['fold'] == f)
        train_cases = relation_df.loc[tr_mask, 'case_id'].astype(str).unique()
        tr_sents = []
        for cid in train_cases:
            txt = case_texts_map.get(cid, "")
            for s in re.split(r'[.!?]+\s+', txt):
                toks = re.findall(r'\b[\w\-]+\b', s)
                if len(toks) >= 2:
                    tr_sents.append(toks)
        svd_fold = DistributionalSVD(embedding_dim=64, window_size=5, min_count=2)
        svd_fold.fit_transform(tr_sents)
        clust_fold = DistributionalClusterNgramExtractor(n_clusters=32, random_state=42)
        clust_fold.fit(svd_fold.word2idx, svd_fold.matrix_64d)
        inductive_features.loc[val_mask] = relation_df.loc[val_mask, 'masked_positional_tokens'].apply(
            clust_fold.augment_feature_string
        )
    relation_df['masked_pos_and_clusters_inductive'] = inductive_features
    res_clust_ind = run_cross_validation_ablation(
        relation_df,
        feature_col='masked_pos_and_clusters_inductive',
        target_col='silver_label',
        model_name='logistic_regression',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_clust['inductive_macro_f1'] = res_clust_ind['macro_f1']
    res_clust['inductive_positive_f1'] = res_clust_ind['positive_f1']
    print(f"     Cluster N-Grams (Estritamente Indutivo por Fold) Macro-F1: {res_clust_ind['macro_f1']:.4f} | Pos-F1: {res_clust_ind['positive_f1']:.4f}")

    # P2.10: Cascata Hierárquica de 2 Estágios (Eliminando canibalização do NONE)
    print(f"  -> Executando P2.10_HierarchicalCascade_StandardNgrams (Detector Link -> Especialista)...")
    res_casc = run_cross_validation_ablation(
        relation_df,
        feature_col='masked_positional_tokens',
        target_col='silver_label',
        model_name='hierarchical_cascade',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_casc['experiment_id'] = 'P2.10_HierarchicalCascade_StandardNgrams'
    res_casc['act'] = 'Ato 2 (Breakthrough)'
    ablation_results.append(res_casc)
    print(f"     Hierarchical Cascade Macro-F1: {res_casc['macro_f1']:.4f} (±{res_casc['macro_f1_std']:.4f})")

    # P2.10b (Ablação Intermediária): Cascata Hierárquica + Cluster N-Grams SEM Denoising (em silver_label bruto)
    print(f"  -> Executando Ablação Intermediária: Cascata + Cluster N-Grams em silver_label original (sem denoising)...")
    res_casc_clust_raw = run_cross_validation_ablation(
        relation_df,
        feature_col='masked_pos_and_clusters',
        target_col='silver_label',
        model_name='hierarchical_cascade',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_casc['cascade_plus_clusters_raw_silver_macro_f1'] = res_casc_clust_raw['macro_f1']
    res_casc['cascade_plus_clusters_raw_silver_positive_f1'] = res_casc_clust_raw['positive_f1']
    print(f"     Cascata + Cluster N-Grams (Un-denoised Silver) Macro-F1: {res_casc_clust_raw['macro_f1']:.4f} | Pos-F1: {res_casc_clust_raw['positive_f1']:.4f}")

    # P2.11: Rede Neural Trilinear de 3 Torres (Clinical DistMult-MLP)
    print(f"  -> Executando P2.11_ThreeTower_DistMult_MLP (h1 * h_ctx * h2 com ~12k parâmetros)...")
    all_types = sorted(set(relation_df['e1_type'].unique()).union(relation_df['e2_type'].unique()))
    type2idx = {t: i for i, t in enumerate(all_types)}

    def build_entity_matrix(df, col_text, col_type):
        mat = []
        for _, r in df.iterrows():
            v_svd = svd_model.get_entity_vector(str(r[col_text]))
            v_type = np.zeros(len(all_types), dtype=np.float32)
            t = str(r[col_type])
            if t in type2idx:
                v_type[type2idx[t]] = 1.0
            mat.append(np.concatenate([v_svd, v_type]))
        return np.array(mat, dtype=np.float32)

    e1_all = build_entity_matrix(relation_df, 'e1_text', 'e1_type')
    e2_all = build_entity_matrix(relation_df, 'e2_text', 'e2_type')

    # Vetorizador TF-IDF do contexto
    ctx_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, min_df=2, sublinear_tf=True)
    ctx_sparse_all = ctx_vec.fit_transform(relation_df['masked_positional_tokens'])

    # Avaliação 5-Fold para ThreeTowerPipeline
    tt_y_true, tt_y_pred = [], []
    tt_fold_f1, tt_fold_pos_f1 = [], []
    for f in sorted(relation_df['fold'].unique()):
        idx_tr = np.where(relation_df['fold'] != f)[0]
        idx_val = np.where(relation_df['fold'] == f)[0]

        tt_pipe = ThreeTowerPipeline(
            entity_dim=e1_all.shape[1],
            ctx_dim=48,
            hidden_dim=48,
            epochs=20,
            batch_size=32,
            seed=42 + f
        )
        tt_pipe.fit(e1_all[idx_tr], e2_all[idx_tr], ctx_sparse_all[idx_tr], relation_df.iloc[idx_tr]['silver_label'].tolist())
        preds_fold = tt_pipe.predict(e1_all[idx_val], e2_all[idx_val], ctx_sparse_all[idx_val])
        y_val_f = relation_df.iloc[idx_val]['silver_label'].tolist()
        tt_y_true.extend(y_val_f)
        tt_y_pred.extend(preds_fold)
        _, _, f1_m_f, _ = precision_recall_fscore_support(y_val_f, preds_fold, labels=all_classes_silver, average='macro', zero_division=0)
        _, _, f1_p_f, _ = precision_recall_fscore_support(y_val_f, preds_fold, labels=pos_classes_silver, average='macro', zero_division=0)
        tt_fold_f1.append(float(f1_m_f))
        tt_fold_pos_f1.append(float(f1_p_f))

    tt_p, tt_r, tt_f1, _ = precision_recall_fscore_support(tt_y_true, tt_y_pred, labels=all_classes_silver, average='macro', zero_division=0)
    _, _, tt_pos_f1, _ = precision_recall_fscore_support(tt_y_true, tt_y_pred, labels=pos_classes_silver, average='macro', zero_division=0)
    ablation_results.append({
        'experiment_id': 'P2.11_ThreeTower_DistMult_MLP',
        'act': 'Ato 2 (Breakthrough)',
        'model_name': 'three_tower_distmult_mlp',
        'target_col': 'silver_label',
        'ngram_range': [1, 2],
        'weighting': 'svd_48d',
        'macro_precision': round(float(tt_p), 4),
        'macro_recall': round(float(tt_r), 4),
        'macro_f1': round(float(tt_f1), 4),
        'macro_f1_std': round(float(np.std(tt_fold_f1)), 4),
        'positive_f1': round(float(tt_pos_f1), 4),
        'positive_f1_std': round(float(np.std(tt_fold_pos_f1)), 4)
    })
    print(f"     Three-Tower DistMult Macro-F1: {tt_f1:.4f} (±{np.std(tt_fold_f1):.4f}) | Pos-F1: {tt_pos_f1:.4f}")

    # Auditoria de Ruído com Confident Learning (Denoising OOF das Regras do P1)
    print(f"\n  -> Executando Auditoria de Ruído Out-of-Fold (Confident Learning Denoising)...")
    oof_probs_all = np.zeros((len(relation_df), len(res_casc['classes'])))
    oof_classes = np.array(res_casc['classes'])

    for f in sorted(relation_df['fold'].unique()):
        idx_tr = np.where(relation_df['fold'] != f)[0]
        idx_val = np.where(relation_df['fold'] == f)[0]

        clf_oof = ClinicalRelationClassifier(model_name='hierarchical_cascade', ngram_range=(1, 3))
        clf_oof.fit(
            relation_df.iloc[idx_tr]['masked_pos_and_clusters'].tolist(),
            relation_df.iloc[idx_tr]['silver_label'].tolist()
        )
        oof_probs = clf_oof.predict_proba(relation_df.iloc[idx_val]['masked_pos_and_clusters'].tolist())
        for j, cname in enumerate(clf_oof.classes_):
            target_idx = np.where(oof_classes == cname)[0][0]
            oof_probs_all[idx_val, target_idx] = oof_probs[:, j]

    y_denoised, audit_log = detect_p1_label_noise_oof(
        y_silver=relation_df['silver_label'].values,
        oof_probs=oof_probs_all,
        classes=oof_classes,
        confidence_threshold=0.76
    )
    relation_df['denoised_label'] = y_denoised
    audit_path = data_interim_dir / 'p1_noise_audit_log.json'
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)
    print(f"  -> {len(audit_log)} rótulos ruidosos de coocorrência do P1 auditados e corrigidos!")
    print(f"  -> Log de auditoria salvo em: {audit_path}")

    # P2.12: Cascata Hierárquica com Clusters + Rótulos Denoised (e avaliação paralela no Silver Bruto)
    print(f"  -> Executando P2.12_Denoised_HierarchicalCascade_ClusterNgrams (Pico de Performance Ato 2)...")
    res_denoised = run_cross_validation_ablation(
        relation_df,
        feature_col='masked_pos_and_clusters',
        target_col='denoised_label',
        model_name='hierarchical_cascade',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_denoised['experiment_id'] = 'P2.12_Denoised_HierarchicalCascade_ClusterNgrams'
    res_denoised['act'] = 'Ato 2 (Breakthrough)'
    res_denoised['raw_silver_macro_f1'] = res_casc_clust_raw['macro_f1']
    res_denoised['raw_silver_positive_f1'] = res_casc_clust_raw['positive_f1']
    ablation_results.append(res_denoised)
    print(f"     Denoised Cascade + Cluster N-Grams Macro-F1: {res_denoised['macro_f1']:.4f} (±{res_denoised['macro_f1_std']:.4f}) | Raw Silver Macro-F1: {res_casc_clust_raw['macro_f1']:.4f}")

    # =========================================================================
    # ATO 3: Fusão Precoce de Informação Ontológica & Ensemble Neuro-Simbólico
    # =========================================================================
    print(f"\n  --- ATO 3: Fusão Precoce & Ensemble Neuro-Simbólico Completo ---")

    # Injeta features condicionadas por tipo MeSH (Typed Positional N-Grams)
    print(f"  -> Gerando N-Grams com Assinatura Ontológica (SIG_E1_TO_E2) e Distância...")
    relation_df['typed_features'] = [
        build_typed_positional_features(r, r['masked_pos_and_clusters']) for _, r in relation_df.iterrows()
    ]

    # P2.13: Cascata Hierárquica sobre Typed Features (avaliado tanto em denoised_label quanto em silver_label)
    print(f"  -> Executando P2.13_TypedNgrams_HierarchicalCascade...")
    res_typed = run_cross_validation_ablation(
        relation_df,
        feature_col='typed_features',
        target_col='denoised_label',
        model_name='hierarchical_cascade',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_typed_raw = run_cross_validation_ablation(
        relation_df,
        feature_col='typed_features',
        target_col='silver_label',
        model_name='hierarchical_cascade',
        ngram_range=(1, 3),
        weighting='tfidf'
    )
    res_typed['experiment_id'] = 'P2.13_TypedNgrams_HierarchicalCascade'
    res_typed['act'] = 'Ato 3 (Neuro-Symbolic Fusion)'
    res_typed['raw_silver_macro_f1'] = res_typed_raw['macro_f1']
    res_typed['raw_silver_positive_f1'] = res_typed_raw['positive_f1']
    ablation_results.append(res_typed)
    print(f"     Typed N-Grams Cascade Macro-F1 (Denoised): {res_typed['macro_f1']:.4f} (±{res_typed['macro_f1_std']:.4f}) | Raw Silver Macro-F1: {res_typed_raw['macro_f1']:.4f}")

    # P2.14: Ensemble Híbrido P2.11 (DistMult MLP) + P2.12/P2.13 (Cascata) + Prior MeSH + Calibração de Classes
    print(f"  -> Executando P2.14_NeuroSymbolic_Ensemble_Tuned (Ensemble Híbrido + Prior MeSH + Calibração)...")
    all_classes_target = np.sort(relation_df['denoised_label'].unique())
    pos_classes_target = [c for c in all_classes_target if c != 'NONE']
    class2idx_map = {c: i for i, c in enumerate(all_classes_target)}
    y_all_indices = np.array([class2idx_map[y] for y in relation_df['denoised_label']])

    prior_evaluator = NeuroSymbolicGraphIntegrator(relations=all_classes_target.tolist(), gamma_prior=0.6)
    edge_tuples_pre = []
    for _, edge in edges_p1_df.iterrows():
        s_nodes = nodes_p1_df[nodes_p1_df['node_id'] == edge['source_id']]
        t_nodes = nodes_p1_df[nodes_p1_df['node_id'] == edge['target_id']]
        if not s_nodes.empty and not t_nodes.empty:
            edge_tuples_pre.append((s_nodes.iloc[0]['type'], t_nodes.iloc[0]['type'], edge['relation']))
    prior_evaluator.fit_ontological_prior(edge_tuples_pre)

    prior_onto_dataset = np.zeros((len(relation_df), len(all_classes_target)), dtype=np.float64)
    for i, r in relation_df.iterrows():
        prior_onto_dataset[i] = prior_evaluator.get_ontological_prior_dist(str(r['e1_type']), str(r['e2_type']))

    ens_y_true, ens_y_pred = [], []
    ens_y_true_raw_silver = []
    ens_fold_f1, ens_fold_pos_f1 = [], []

    for f in sorted(relation_df['fold'].unique()):
        idx_tr = np.where(relation_df['fold'] != f)[0]
        idx_val = np.where(relation_df['fold'] == f)[0]

        # 1. Three-Tower MLP
        tt_pipe = ThreeTowerPipeline(entity_dim=e1_all.shape[1], ctx_dim=48, hidden_dim=48, epochs=15, batch_size=32, seed=42+f)
        tt_pipe.fit(e1_all[idx_tr], e2_all[idx_tr], ctx_sparse_all[idx_tr], relation_df.iloc[idx_tr]['denoised_label'].tolist())
        p_mlp_tr = tt_pipe.predict_proba(e1_all[idx_tr], e2_all[idx_tr], ctx_sparse_all[idx_tr])
        p_mlp_val = tt_pipe.predict_proba(e1_all[idx_val], e2_all[idx_val], ctx_sparse_all[idx_val])

        # 2. Cascata com Typed Features
        vec_typed = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, min_df=2, sublinear_tf=True)
        X_typed_tr = vec_typed.fit_transform(relation_df.iloc[idx_tr]['typed_features'])
        X_typed_val = vec_typed.transform(relation_df.iloc[idx_val]['typed_features'])
        casc = HierarchicalCascadeClassifier(c_stage1=1.5, c_stage2=2.5, link_threshold=0.45)
        casc.fit(X_typed_tr, relation_df.iloc[idx_tr]['denoised_label'].values)
        p_casc_tr = casc.predict_proba(X_typed_tr)
        p_casc_val = casc.predict_proba(X_typed_val)

        # Alinhamento
        p_mlp_tr_aligned = np.zeros((len(idx_tr), len(all_classes_target)))
        p_mlp_val_aligned = np.zeros((len(idx_val), len(all_classes_target)))
        for j, c in enumerate(tt_pipe.classes_):
            c_idx = np.where(all_classes_target == c)[0][0]
            p_mlp_tr_aligned[:, c_idx] = p_mlp_tr[:, j]
            p_mlp_val_aligned[:, c_idx] = p_mlp_val[:, j]

        p_casc_tr_aligned = np.zeros((len(idx_tr), len(all_classes_target)))
        p_casc_val_aligned = np.zeros((len(idx_val), len(all_classes_target)))
        for j, c in enumerate(casc.classes_):
            c_idx = np.where(all_classes_target == c)[0][0]
            p_casc_tr_aligned[:, c_idx] = p_casc_tr[:, j]
            p_casc_val_aligned[:, c_idx] = p_casc_val[:, j]

        calibrated_val_probs = neuro_symbolic_ensemble_and_tune(
            probs_cascade_train=p_casc_tr_aligned,
            probs_mlp_train=p_mlp_tr_aligned,
            prior_onto_train=prior_onto_dataset[idx_tr],
            y_train_idx=y_all_indices[idx_tr],
            probs_cascade_val=p_casc_val_aligned,
            probs_mlp_val=p_mlp_val_aligned,
            prior_onto_val=prior_onto_dataset[idx_val],
            w_cascade=0.55,
            gamma_onto=0.65
        )

        preds_fold_idx = np.argmax(calibrated_val_probs, axis=1)
        preds_fold_labels = all_classes_target[preds_fold_idx]
        y_val_f = relation_df.iloc[idx_val]['denoised_label'].tolist()
        ens_y_pred.extend(preds_fold_labels)
        ens_y_true.extend(y_val_f)
        ens_y_true_raw_silver.extend(relation_df.iloc[idx_val]['silver_label'].tolist())

        _, _, f1_m_f, _ = precision_recall_fscore_support(y_val_f, preds_fold_labels, labels=all_classes_target, average='macro', zero_division=0)
        _, _, f1_p_f, _ = precision_recall_fscore_support(y_val_f, preds_fold_labels, labels=pos_classes_target, average='macro', zero_division=0)
        ens_fold_f1.append(float(f1_m_f))
        ens_fold_pos_f1.append(float(f1_p_f))

    ens_p, ens_r, ens_f1, _ = precision_recall_fscore_support(ens_y_true, ens_y_pred, labels=all_classes_target, average='macro', zero_division=0)
    _, _, ens_pos_f1, _ = precision_recall_fscore_support(ens_y_true, ens_y_pred, labels=pos_classes_target, average='macro', zero_division=0)
    _, _, ens_raw_silver_f1, _ = precision_recall_fscore_support(ens_y_true_raw_silver, ens_y_pred, labels=all_classes_target, average='macro', zero_division=0)
    _, _, ens_raw_silver_pos_f1, _ = precision_recall_fscore_support(ens_y_true_raw_silver, ens_y_pred, labels=pos_classes_target, average='macro', zero_division=0)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(ens_y_true, ens_y_pred, labels=all_classes_target, zero_division=0)
    per_class_ens = {}
    for i, cls_name in enumerate(all_classes_target):
        per_class_ens[cls_name] = {
            'precision': round(float(p_per[i]), 4),
            'recall': round(float(r_per[i]), 4),
            'f1': round(float(f1_per[i]), 4),
            'support': int(sup_per[i])
        }

    ablation_results.append({
        'experiment_id': 'P2.14_NeuroSymbolic_Ensemble_Tuned',
        'act': 'Ato 3 (Neuro-Symbolic Fusion)',
        'model_name': 'neuro_symbolic_ensemble',
        'target_col': 'denoised_label',
        'ngram_range': [1, 3],
        'weighting': 'hybrid_dense_sparse_prior',
        'macro_precision': round(float(ens_p), 4),
        'macro_recall': round(float(ens_r), 4),
        'macro_f1': round(float(ens_f1), 4),
        'macro_f1_std': round(float(np.std(ens_fold_f1)), 4),
        'positive_f1': round(float(ens_pos_f1), 4),
        'positive_f1_std': round(float(np.std(ens_fold_pos_f1)), 4),
        'raw_silver_macro_f1': round(float(ens_raw_silver_f1), 4),
        'raw_silver_positive_f1': round(float(ens_raw_silver_pos_f1), 4),
        'per_class': per_class_ens
    })
    print(f"     Neuro-Symbolic Ensemble Macro-F1 (Denoised): {ens_f1:.4f} (±{np.std(ens_fold_f1):.4f}) | Pos-F1: {ens_pos_f1:.4f} | Raw Silver Macro-F1: {ens_raw_silver_f1:.4f}")

    # Salva relatório de ablação consolidado
    benchmark_path = data_processed_dir / 'ablation_benchmark.json'
    with open(benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"\n  -> Torneio Consolidado salvo em: {benchmark_path}")

    # -------------------------------------------------------------------------
    # FASE 5: Fusão Neuro-Simbólica & Inferência Coletiva por Triângulos
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Fase 5: Fusão Neuro-Simbólica e Fechamento de Triângulos Clínicos...")

    final_classifier = ClinicalRelationClassifier(model_name='logistic_regression', ngram_range=(1, 3))
    final_classifier.fit(relation_df['typed_features'].tolist(), relation_df['denoised_label'].tolist())

    # Prior Ontológico MeSH
    edge_tuples = []
    for _, edge in edges_p1_df.iterrows():
        sid = edge['source_id']
        tid = edge['target_id']
        s_nodes = nodes_p1_df[nodes_p1_df['node_id'] == sid]
        t_nodes = nodes_p1_df[nodes_p1_df['node_id'] == tid]
        if not s_nodes.empty and not t_nodes.empty:
            edge_tuples.append((s_nodes.iloc[0]['type'], t_nodes.iloc[0]['type'], edge['relation']))

    integrator = NeuroSymbolicGraphIntegrator(relations=final_classifier.classes_.tolist(), gamma_prior=0.6)
    integrator.fit_ontological_prior(edge_tuples)

    # Calibração e XAI
    all_ml_probs = final_classifier.predict_proba(relation_df['typed_features'].tolist())
    calibrated_rows = []

    for idx, row in relation_df.iterrows():
        t1 = row['e1_type']
        t2 = row['e2_type']
        probs = all_ml_probs[idx]

        calib = integrator.calibrate(probs, t1, t2)
        explanation = final_classifier.explain_prediction(row['typed_features'], predicted_class=calib['calibrated_relation'])

        row_dict = row.to_dict()
        row_dict['calibrated_relation'] = calib['calibrated_relation']
        row_dict['p_final'] = calib['p_final']
        row_dict['p_ml_raw'] = calib['p_ml_raw']
        row_dict['p_onto_prior'] = calib['p_onto_prior']
        row_dict['top3_pos_ngrams'] = json.dumps(explanation.get('top_positive', []))
        calibrated_rows.append(row_dict)

    ml_candidates_df = pd.DataFrame(calibrated_rows)
    ml_cand_path = data_interim_dir / 'ml_predictions_xai.csv'
    ml_candidates_df.to_csv(ml_cand_path, index=False)

    # Constrói o Grafo Final P2 com Fechamento de Triângulos
    nodes_p2_df, edges_p2_df, graph_metrics = integrator.build_enhanced_graph(
        p1_nodes_df=nodes_p1_df,
        p1_edges_df=edges_p1_df,
        ml_candidates_df=ml_candidates_df,
        oov_nodes_df=oov_nodes_df,
        confidence_threshold=0.70
    )

    nodes_p2_path = data_processed_dir / 'nodes_p2.csv'
    edges_p2_path = data_processed_dir / 'edges_p2.csv'
    nodes_p2_df.to_csv(nodes_p2_path, index=False)
    edges_p2_df.to_csv(edges_p2_path, index=False)

    # -------------------------------------------------------------------------
    # FASE 6: Visualizador Interativo Cytoscape.js com Sandbox Contrafactual
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Fase 6: Gerando Visualizador HTML com Simulador Contrafactual...")
    html_output_proc = data_processed_dir / 'graph_p2_visualization.html'
    html_output_assets = assets_dir / 'graph_p2_visualization.html'

    generate_cytoscape_p2_html(nodes_p2_df, edges_p2_df, str(html_output_proc))
    generate_cytoscape_p2_html(nodes_p2_df, edges_p2_df, str(html_output_assets))

    print("\n" + "=" * 75)
    print("PROJETO 2 EXECUTADO COM SUCESSO ABSOLUTO!")
    print("=" * 75)
    print(f"  • Nós no Grafo: {graph_metrics['total_nodes_p1']} (P1) -> {graph_metrics['total_nodes_p2']} (P2) (+{graph_metrics['oov_nodes_added']} OOV promovidos)")
    print(f"  • Arestas no Grafo: {graph_metrics['total_edges_p1']} (P1) -> {graph_metrics['total_edges_p2']} (P2) (+{graph_metrics['novel_ml_edges_added']} novas arestas ML, ganho de +{graph_metrics['relative_edge_gain_pct']}%)")
    print(f"  • Triângulos Clínicos Fechados (Relational Belief Propagation): {graph_metrics['triangles_closed']}")
    print(f"  • Densidade da Rede P2: {graph_metrics['graph_density_p2']:.6f}")
    print(f"  • Visualizador Standalone Processado: {html_output_proc}")
    print(f"  • Visualizador Standalone Assets: {html_output_assets}")
    print("=" * 75)


if __name__ == '__main__':
    main()
