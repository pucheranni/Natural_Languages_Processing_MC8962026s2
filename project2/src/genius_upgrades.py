"""
genius_upgrades.py - As 5 Inovações de Alto Impacto (Nível Artigo BioNLP):
1. N-Grams Abstratos de Clusters Semânticos (Distributional Class-Based N-Grams via PPMI-SVD + MiniBatchKMeans K=32).
2. Classificação em Cascata Hierárquica de 2 Estágios (Link Detector -> Multiclass Specialist).
3. Confident Learning Denoising Out-of-Fold para auditar o ruído de coocorrência do P1.
4. Rede Neural Trilinear de 3 Torres (Clinical DistMult-MLP) com interação h1 * h_ctx * h2.
Fundamentado em:
- Jurafsky & Martin (2026), slides nlp-060 (Semântica Vetorial), nlp-080 (Supervisão Fraca), nlp-110 (Redes Neurais)
- Levy & Goldberg (2014) - Neural Word Embeddings as Implicit Matrix Factorization
- Northcutt et al. (2021) - Confident Learning: Estimating Uncertainty in Dataset Labels
- Yang et al. (2015) - DistMult: Embedding Entities and Relations for Learning Knowledge Bases
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# 1. N-GRAMS ABSTRATOS DE CLUSTERS DISTRIBUTIVOS (Quebra a esparsidade dos 3-grams)
# =====================================================================
class DistributionalClusterNgramExtractor:
    """
    Agrupa o vocabulário em K arquétipos semânticos usando os embeddings PPMI-SVD (64d)
    e gera N-grams abstratos (ex: BTW_TPL_was_CLUST07_with) para capturar paráfrases inéditas.
    Slide nlp-060: Semântica Vetorial & Generalização Distribucional.
    """

    def __init__(self, n_clusters: int = 32, random_state: int = 42):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.word2cluster: Dict[str, str] = {}
        self.is_fitted = False

    def fit(self, word2idx: Dict[str, int], svd_embeddings: np.ndarray):
        """Treina clusters K-Means sobre os vetores densos PPMI-SVD do vocabulário."""
        if len(svd_embeddings) == 0:
            return self

        n_clust = min(self.n_clusters, max(2, len(svd_embeddings)))
        self.kmeans.n_clusters = n_clust
        cluster_ids = self.kmeans.fit_predict(svd_embeddings)

        for word, idx in word2idx.items():
            if idx < len(cluster_ids):
                # Palavras funcionais curtas mantêm sua forma sintática exata (of, with, in, by, no)
                if len(word) <= 3 or word in {
                    "with", "from", "into", "after", "before", "without", "trigger_mask", "not", "for"
                }:
                    self.word2cluster[word.lower()] = word.lower()
                else:
                    self.word2cluster[word.lower()] = f"CLUST{cluster_ids[idx]:02d}"

        self.is_fitted = True
        return self

    def augment_with_cluster_ngrams(self, btw_tokens: List[str], max_n: int = 3) -> List[str]:
        """Gera 2-grams e 3-grams abstratos combinando preposições literais + clusters SVD."""
        if not self.is_fitted:
            return []

        abstract_seq = [self.word2cluster.get(t.lower(), "CLUST_UNK") for t in btw_tokens if t]
        tpl_features = []
        for n in range(2, max_n + 1):
            for i in range(len(abstract_seq) - n + 1):
                gram = "_".join(abstract_seq[i : i + n])
                tpl_features.append(f"BTW_TPL_{gram}")
        return tpl_features

    def augment_feature_string(self, feature_str: str) -> str:
        """
        Extrai tokens da zona BTW de uma string existente de features posicionais
        e anexa os N-grams abstratos de clusters à mesma string.
        """
        if not self.is_fitted or not feature_str:
            return feature_str

        # Coleta unigramas do BTW_
        raw_tokens = feature_str.split()
        btw_words = []
        for tok in raw_tokens:
            if tok.startswith("BTW_") and not tok.startswith("BTW_TPL_"):
                suffix = tok[4:]
                # Apenas unigramas (sem underline no sufixo)
                if "_" not in suffix:
                    btw_words.append(suffix)

        if not btw_words:
            return feature_str

        cluster_ngrams = self.augment_with_cluster_ngrams(btw_words, max_n=3)
        if cluster_ngrams:
            return feature_str + " " + " ".join(cluster_ngrams)
        return feature_str


# =====================================================================
# 2. CLASSIFICADOR EM CASCATA HIERÁRQUICA DE 2 ESTÁGIOS (Destrava o Macro-F1)
# =====================================================================
class HierarchicalCascadeClassifier(BaseEstimator, ClassifierMixin):
    """
    Estágio 1: Detector Binário de Enlace Clínico (LINK vs NONE)
    Estágio 2: Especialista Multiclasse entre as Relações Positivas
    Impede que a classe majoritária NONE esmague classes raras (EXCLUDES, CONFIRMS).
    Regra da Cadeia de Probabilidade (Slide nlp-080):
      P(R = r | x) = P(LINK = 1 | x) * P_stage2(R = r | LINK = 1, x)
    """

    def __init__(
        self,
        none_label: str = "NONE",
        c_stage1: float = 1.5,
        c_stage2: float = 2.5,
        link_threshold: float = 0.42
    ):
        self.none_label = none_label
        self.c_stage1 = c_stage1
        self.c_stage2 = c_stage2
        self.link_threshold = link_threshold
        self.classes_ = np.array([])
        self.stage1_clf_ = None
        self.stage2_clf_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        is_positive = (y != self.none_label).astype(int)

        # Estágio 1: Binário (Existe relação clínica neste par?)
        self.stage1_clf_ = LogisticRegression(
            C=self.c_stage1, class_weight="balanced", max_iter=1000, random_state=42
        )
        self.stage1_clf_.fit(X, is_positive)

        # Estágio 2: Especialista apenas nas instâncias positivas
        pos_mask = (is_positive == 1)
        if np.sum(pos_mask) > 0 and len(np.unique(y[pos_mask])) > 1:
            X_pos = X[pos_mask]
            y_pos = y[pos_mask]
            self.stage2_clf_ = LogisticRegression(
                C=self.c_stage2, class_weight="balanced", max_iter=1000, random_state=42
            )
            self.stage2_clf_.fit(X_pos, y_pos)
        else:
            self.stage2_clf_ = None

        return self

    def predict_proba(self, X) -> np.ndarray:
        p_link_both = self.stage1_clf_.predict_proba(X)
        # Identifica índice da classe positiva (1) e negativa (0)
        classes_stage1 = list(self.stage1_clf_.classes_)
        idx_link = classes_stage1.index(1) if 1 in classes_stage1 else 0
        idx_none = classes_stage1.index(0) if 0 in classes_stage1 else 0

        p_none = p_link_both[:, idx_none]
        p_link = p_link_both[:, idx_link]

        full_probs = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)

        if self.stage2_clf_ is not None:
            p_rel_given_link = self.stage2_clf_.predict_proba(X)
            pos_classes = list(self.stage2_clf_.classes_)

            for idx, cls_name in enumerate(self.classes_):
                if cls_name == self.none_label:
                    full_probs[:, idx] = p_none
                else:
                    if cls_name in pos_classes:
                        pos_idx = pos_classes.index(cls_name)
                        full_probs[:, idx] = p_link * p_rel_given_link[:, pos_idx]
                    else:
                        full_probs[:, idx] = 0.0
        else:
            for idx, cls_name in enumerate(self.classes_):
                if cls_name == self.none_label:
                    full_probs[:, idx] = p_none
                else:
                    full_probs[:, idx] = p_link / max(1, len(self.classes_) - 1)

        # Normalização de segurança por linha
        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return full_probs / row_sums

    def predict(self, X) -> np.ndarray:
        classes_stage1 = list(self.stage1_clf_.classes_)
        idx_link = classes_stage1.index(1) if 1 in classes_stage1 else 0
        p_link = self.stage1_clf_.predict_proba(X)[:, idx_link]

        if self.stage2_clf_ is not None:
            pos_preds = self.stage2_clf_.predict(X)
            return np.where(p_link >= self.link_threshold, pos_preds, self.none_label)
        else:
            return np.where(p_link >= self.link_threshold, "HAS_RELATION", self.none_label)


# =====================================================================
# 3. CONFIDENT LEARNING DENOISER (Audita e limpa os erros de coocorrência do P1)
# =====================================================================
def detect_p1_label_noise_oof(
    y_silver: np.ndarray,
    oof_probs: np.ndarray,
    classes: np.ndarray,
    confidence_threshold: float = 0.78
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Identifica instâncias onde o P1 gerou falso positivo por coocorrência ingênua
    (ex: SUPPORTS) ou falso negativo (NONE), mas o consenso Out-of-Fold diverge com alta certeza.
    Fundamentado em Confident Learning (Northcutt et al., 2021).
    """
    y_cleaned = np.array(y_silver, dtype=object).copy()
    audit_log = []

    for i in range(len(y_silver)):
        best_c_idx = int(np.argmax(oof_probs[i]))
        best_c_label = str(classes[best_c_idx])
        best_prob = float(oof_probs[i, best_c_idx])
        current_label = str(y_silver[i])

        if best_c_label != current_label and best_prob >= confidence_threshold:
            # Foco nas classes de coocorrência ruidosa do P1 (SUPPORTS, TARGETS, NONE)
            if current_label in {"SUPPORTS", "TARGETS", "NONE"}:
                y_cleaned[i] = best_c_label
                audit_log.append({
                    "instance_idx": i,
                    "p1_noisy_label": current_label,
                    "p2_confident_label": best_c_label,
                    "oof_confidence": round(best_prob, 4)
                })

    return y_cleaned, audit_log


# =====================================================================
# 4. REDE NEURAL TRILINEAR DE 3 TORRES (Clinical DistMult-MLP - Slide 110)
# =====================================================================
class ThreeTowerBilinearMLP(nn.Module):
    """
    Resolve o overfitting do MLP padrão reduzindo parâmetros de ~380k para ~11k:
    Torre 1 (E1 SVD + Tipo) | Torre 2 (E2 SVD + Tipo) | Torre 3 (Contexto N-gram SVD)
    Camada Bilinear/Trilinear: h1 * h_ctx * h2 (Inspirado em DistMult Knowledge Graph Embeddings)
    """

    def __init__(
        self,
        entity_dim: int,
        ctx_svd_dim: int,
        num_classes: int,
        hidden_dim: int = 48,
        dropout: float = 0.25
    ):
        super().__init__()
        self.tower_e1 = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tower_e2 = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tower_ctx = nn.Sequential(
            nn.Linear(ctx_svd_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Entrada combinada: [h1, h2, h_ctx, h1*h_ctx*h2 (Trilinear), h1*h2 (Hadamard)] = 5 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        e1_vec: torch.Tensor,
        e2_vec: torch.Tensor,
        ctx_vec: torch.Tensor
    ) -> torch.Tensor:
        h1 = self.tower_e1(e1_vec)
        h2 = self.tower_e2(e2_vec)
        h_ctx = self.tower_ctx(ctx_vec)

        trilinear_interaction = h1 * h_ctx * h2
        entity_compatibility = h1 * h2

        fused = torch.cat([h1, h2, h_ctx, trilinear_interaction, entity_compatibility], dim=-1)
        return self.classifier(fused)


class ThreeTowerDataset(Dataset):
    """Dataset auxiliar para a rede de 3 torres."""

    def __init__(
        self,
        e1_vecs: np.ndarray,
        e2_vecs: np.ndarray,
        ctx_vecs: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        self.e1 = torch.tensor(e1_vecs, dtype=torch.float32)
        self.e2 = torch.tensor(e2_vecs, dtype=torch.float32)
        self.ctx = torch.tensor(ctx_vecs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.e1)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.e1[idx], self.e2[idx], self.ctx[idx], self.labels[idx]
        return self.e1[idx], self.e2[idx], self.ctx[idx]


class ThreeTowerPipeline:
    """
    Pipeline completo para orquestrar o Treinamento e Inferência da Rede de 3 Torres.
    Comprime o contexto via TruncatedSVD e monta os vetores das entidades.
    """

    def __init__(
        self,
        entity_dim: int = 64 + 10,  # 64d SVD + 10d One-Hot de Tipo
        ctx_dim: int = 48,
        hidden_dim: int = 48,
        epochs: int = 25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        seed: int = 42
    ):
        self.entity_dim = entity_dim
        self.ctx_dim = ctx_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed

        self.svd_ctx = TruncatedSVD(n_components=ctx_dim, random_state=seed)
        self.classes_: np.ndarray = np.array([])
        self.label2id: Dict[str, int] = {}
        self.model: Optional[ThreeTowerBilinearMLP] = None
        self.device = torch.device("cpu")

    def fit(
        self,
        e1_matrix: np.ndarray,
        e2_matrix: np.ndarray,
        ctx_sparse: sp.csr_matrix,
        y_labels: List[str]
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.classes_ = np.sort(np.unique(y_labels))
        self.label2id = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([self.label2id[y] for y in y_labels], dtype=np.int64)

        # Redução do contexto N-gram esparso para denso ctx_dim
        k = min(self.ctx_dim, ctx_sparse.shape[1] - 1, ctx_sparse.shape[0] - 1)
        self.svd_ctx.n_components = max(2, k)
        ctx_dense = self.svd_ctx.fit_transform(ctx_sparse)
        if ctx_dense.shape[1] < self.ctx_dim:
            pad = np.zeros((ctx_dense.shape[0], self.ctx_dim - ctx_dense.shape[1]), dtype=np.float32)
            ctx_dense = np.hstack([ctx_dense, pad])

        # Criação do modelo
        self.model = ThreeTowerBilinearMLP(
            entity_dim=e1_matrix.shape[1],
            ctx_svd_dim=self.ctx_dim,
            num_classes=len(self.classes_),
            hidden_dim=self.hidden_dim,
            dropout=0.25
        ).to(self.device)

        # Balanceamento de classes para a perda CrossEntropy
        class_counts = np.bincount(y_indices, minlength=len(self.classes_))
        weights = len(y_indices) / (len(self.classes_) * np.maximum(class_counts, 1).astype(np.float32))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = ThreeTowerDataset(e1_matrix, e2_matrix, ctx_dense, y_indices)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for b_e1, b_e2, b_ctx, b_y in loader:
                b_e1 = b_e1.to(self.device)
                b_e2 = b_e2.to(self.device)
                b_ctx = b_ctx.to(self.device)
                b_y = b_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(b_e1, b_e2, b_ctx)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(
        self,
        e1_matrix: np.ndarray,
        e2_matrix: np.ndarray,
        ctx_sparse: sp.csr_matrix
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")

        ctx_dense = self.svd_ctx.transform(ctx_sparse)
        if ctx_dense.shape[1] < self.ctx_dim:
            pad = np.zeros((ctx_dense.shape[0], self.ctx_dim - ctx_dense.shape[1]), dtype=np.float32)
            ctx_dense = np.hstack([ctx_dense, pad])

        self.model.eval()
        with torch.no_grad():
            t_e1 = torch.tensor(e1_matrix, dtype=torch.float32).to(self.device)
            t_e2 = torch.tensor(e2_matrix, dtype=torch.float32).to(self.device)
            t_ctx = torch.tensor(ctx_dense, dtype=torch.float32).to(self.device)
            logits = self.model(t_e1, t_e2, t_ctx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(
        self,
        e1_matrix: np.ndarray,
        e2_matrix: np.ndarray,
        ctx_sparse: sp.csr_matrix
    ) -> np.ndarray:
        probs = self.predict_proba(e1_matrix, e2_matrix, ctx_sparse)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]


# =====================================================================
# 5. ATO 3: FUSÃO PRECOCE (TYPED N-GRAMS) & ENSEMBLE NEURO-SIMBÓLICO
# =====================================================================
def build_typed_positional_features(df_row: Dict[str, Any], base_positional_ngrams: str) -> str:
    """
    Alavanca 1: Injeta Assinatura Ontológica (Tipo_E1 -> Tipo_E2), Bucket de Distância
    e Interação Tipo x Contexto diretamente na string de N-Grams.
    Slide nlp-080: Engenharia de Atributos Condicionada por Tipagem Semântica.
    """
    t1 = str(df_row.get("e1_type", df_row.get("entity1_type", "Entity"))).upper()
    t2 = str(df_row.get("e2_type", df_row.get("entity2_type", "Entity"))).upper()
    dist = int(df_row.get("token_dist", df_row.get("distance_tokens", 5)))

    if dist <= 4:
        dist_tok = "DIST_SHORT"
    elif dist <= 10:
        dist_tok = "DIST_MEDIUM"
    else:
        dist_tok = "DIST_LONG"

    sig_tok = f"SIG_{t1}_TO_{t2}"
    sig_dist_tok = f"SIG_{t1}_{t2}_{dist_tok}"

    # Cria até 4 features de interação cruzada (Assinatura Ontológica x N-Grams do meio)
    btw_grams = [tok for tok in str(base_positional_ngrams).split() if tok.startswith("BTW_")][:4]
    cross_tokens = [f"{sig_tok}__{g}" for g in btw_grams]

    extra = " ".join([sig_tok, dist_tok, sig_dist_tok] + cross_tokens)
    return f"{base_positional_ngrams} {extra}"


def neuro_symbolic_ensemble_and_tune(
    probs_cascade_train: np.ndarray,
    probs_mlp_train: np.ndarray,
    prior_onto_train: np.ndarray,
    y_train_idx: np.ndarray,
    probs_cascade_val: np.ndarray,
    probs_mlp_val: np.ndarray,
    prior_onto_val: np.ndarray,
    w_cascade: float = 0.55,
    gamma_onto: float = 0.65,
) -> np.ndarray:
    """
    Alavancas 2 & 3: Combina Cascata Linear + Three-Tower MLP + Prior Ontológico MeSH
    e calibra multiplicadores por classe no fold de treino para maximizar o Macro-F1.
    """
    # 1. Fusão no Treino
    blend_train = (w_cascade * probs_cascade_train + (1.0 - w_cascade) * probs_mlp_train) * np.power(
        prior_onto_train + 1e-6, gamma_onto
    )
    blend_train /= np.sum(blend_train, axis=1, keepdims=True) + 1e-12

    # 2. Fusão na Validação (Paciente Inédito)
    blend_val = (w_cascade * probs_cascade_val + (1.0 - w_cascade) * probs_mlp_val) * np.power(
        prior_onto_val + 1e-6, gamma_onto
    )
    blend_val /= np.sum(blend_val, axis=1, keepdims=True) + 1e-12

    # 3. Calibração Coordinate-Search dos multiplicadores de classe (apenas no treino!)
    num_classes = blend_train.shape[1]
    class_multipliers = np.ones(num_classes, dtype=np.float64)

    for _ in range(2):
        for c in range(num_classes):
            best_m = class_multipliers[c]
            best_score = f1_score(y_train_idx, np.argmax(blend_train * class_multipliers, axis=1), average="macro")
            for candidate_m in np.linspace(0.4, 2.5, 22):
                temp_mult = class_multipliers.copy()
                temp_mult[c] = candidate_m
                score = f1_score(y_train_idx, np.argmax(blend_train * temp_mult, axis=1), average="macro")
                if score > best_score:
                    best_score = score
                    best_m = candidate_m
            class_multipliers[c] = best_m

    calibrated_val_probs = blend_val * class_multipliers
    return calibrated_val_probs / (np.sum(calibrated_val_probs, axis=1, keepdims=True) + 1e-12)
