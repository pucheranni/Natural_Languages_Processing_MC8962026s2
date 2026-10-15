"""
ngram_miner.py - Motor Estatístico de N-Grams, Collocations PPMI, SVD Denso e Morfologia OOV.
Fundamentado em:
- Jurafsky & Martin (2026) / Manning et al. (2009)
- Slide nlp-060: Espaço Vetorial, Hipótese Distributiva de Firth, Matriz de Coocorrência e PPMI
- Slide nlp-110: Representações Densas (Word2Vec / SVD de Levy & Goldberg)
"""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Stopwords clássicas para bordas de N-grams (não podem abrir nem fechar um termo composto)
BORDER_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
    'by', 'from', 'about', 'into', 'through', 'after', 'before', 'between',
    'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'as', 'it', 'its', 'he', 'she', 'they', 'this', 'that',
    'these', 'those', 'which', 'who', 'whom', 'what', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'should', 'now', 'there', 'patient', 'patients', 'case',
    'reported', 'noted', 'presented', 'day', 'days', 'year', 'years', 'old'
}


class CollocationMiner:
    """Minera bigramas e trigramas clinicamente significativos via PPMI e Student's t-test."""

    def __init__(self, min_freq: int = 3, border_stopwords: Optional[Set[str]] = None):
        self.min_freq = min_freq
        self.border_stopwords = border_stopwords if border_stopwords is not None else BORDER_STOPWORDS

    def _is_valid_collocation(self, n_gram_tokens: Tuple[str, ...]) -> bool:
        """Verifica se o N-gram é válido: não abre nem fecha com stopword, contém letras."""
        first, last = n_gram_tokens[0].lower(), n_gram_tokens[-1].lower()
        if first in self.border_stopwords or last in self.border_stopwords:
            return False
        # Precisa ter comprimento mínimo de caracteres e conter letras
        if any(len(tok) < 2 or not re.search(r'[a-zA-Z]', tok) for tok in n_gram_tokens):
            return False
        # Não pode ter dígitos puros
        if any(tok.isdigit() for tok in n_gram_tokens):
            return False
        return True

    def mine_collocations(
        self,
        tokenized_sentences: List[List[str]],
        max_n: int = 3,
        top_k: int = 250
    ) -> List[Dict[str, Any]]:
        """
        Extrai collocations de tamanho 2 e 3 calculando:
        - Frequência observada C(w1..wn)
        - PPMI (Positive Pointwise Mutual Information)
        - Student's t-score para significância estatística
        """
        unigram_counts = Counter()
        ngram_counts: Dict[int, Counter] = {2: Counter(), 3: Counter()}
        total_tokens = 0

        for sent in tokenized_sentences:
            clean_tokens = [w.lower() for w in sent if re.search(r'\w', w)]
            total_tokens += len(clean_tokens)
            unigram_counts.update(clean_tokens)

            # Bigramas
            for i in range(len(clean_tokens) - 1):
                bg = (clean_tokens[i], clean_tokens[i + 1])
                ngram_counts[2][bg] += 1

            # Trigramas
            if max_n >= 3:
                for i in range(len(clean_tokens) - 2):
                    tg = (clean_tokens[i], clean_tokens[i + 1], clean_tokens[i + 2])
                    ngram_counts[3][tg] += 1

        results = []
        N = float(total_tokens)
        if N == 0:
            return []

        # Processa Bigramas
        for (w1, w2), c12 in ngram_counts[2].items():
            if c12 < self.min_freq:
                continue
            if not self._is_valid_collocation((w1, w2)):
                continue

            c1 = unigram_counts[w1]
            c2 = unigram_counts[w2]

            # P(w1, w2), P(w1), P(w2)
            p12 = c12 / N
            p1 = c1 / N
            p2 = c2 / N

            # PPMI = max(0, log2( P(w1, w2) / (P(w1)*P(w2)) ))
            expected_p = p1 * p2
            pmi = math.log2(p12 / expected_p) if expected_p > 0 and p12 > 0 else 0.0
            ppmi = max(0.0, pmi)

            # Student's t-test: t = (P(w1, w2) - P(w1)*P(w2)) / sqrt(P(w1, w2) / N)
            variance = max(1e-12, p12 / N)
            t_score = (p12 - expected_p) / math.sqrt(variance)

            # Score composto: PPMI normalizado * sigmoid(t_score)
            composite_score = ppmi * (1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, t_score / 2.0)))))

            results.append({
                'ngram_text': f"{w1} {w2}",
                'n_size': 2,
                'tokens': (w1, w2),
                'freq': c12,
                'ppmi_score': round(ppmi, 4),
                't_score': round(t_score, 4),
                'composite_score': round(composite_score, 4)
            })

        # Processa Trigramas
        if max_n >= 3:
            for (w1, w2, w3), c123 in ngram_counts[3].items():
                if c123 < self.min_freq:
                    continue
                if not self._is_valid_collocation((w1, w2, w3)):
                    continue

                c1 = unigram_counts[w1]
                c2 = unigram_counts[w2]
                c3 = unigram_counts[w3]

                p123 = c123 / N
                expected_p = (c1 / N) * (c2 / N) * (c3 / N)
                pmi = math.log2(p123 / expected_p) if expected_p > 0 and p123 > 0 else 0.0
                ppmi = max(0.0, pmi)

                variance = max(1e-12, p123 / N)
                t_score = (p123 - expected_p) / math.sqrt(variance)
                composite_score = ppmi * (1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, t_score / 2.0)))))

                results.append({
                    'ngram_text': f"{w1} {w2} {w3}",
                    'n_size': 3,
                    'tokens': (w1, w2, w3),
                    'freq': c123,
                    'ppmi_score': round(ppmi, 4),
                    't_score': round(t_score, 4),
                    'composite_score': round(composite_score, 4)
                })

        # Ordena pelo score composto
        results.sort(key=lambda x: (x['composite_score'], x['freq']), reverse=True)
        return results[:top_k]


class MorphologyOOVClassifier:
    """
    Classificador de morfologia de subpalavras (3 a 5 caracteres) para tipar entidades Out-of-Vocabulary.
    Aprende a mapear prefixos e sufixos médicos greco-latinos (-itis, -ectomy, -oma, -scopy)
    para as 4 categorias ontológicas principais MeSH:
    - C: Disease / Finding
    - D: Chemical / Drug
    - E01: Exam / Procedure
    - A: Anatomy
    """

    def __init__(self, ngram_range: Tuple[int, int] = (3, 5)):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=2,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.is_fitted = False

    def fit(self, mesh_terms: List[str], mesh_categories: List[str]):
        """Treina nos termos MeSH curados do Projeto 1."""
        clean_terms = [t.lower().strip() for t in mesh_terms]
        X = self.vectorizer.fit_transform(clean_terms)
        self.classifier.fit(X, mesh_categories)
        self.is_fitted = True
        return self

    def predict_oov_types(
        self,
        candidate_terms: List[str]
    ) -> List[Dict[str, Any]]:
        """Prediz a categoria ontológica e probabilidade para cada termo OOV."""
        if not self.is_fitted:
            raise RuntimeError("MorphologyOOVClassifier must be fitted before predict.")

        clean_candidates = [t.lower().strip() for t in candidate_terms]
        X = self.vectorizer.transform(clean_candidates)
        probs = self.classifier.predict_proba(X)
        classes = self.classifier.classes_

        results = []
        for i, term in enumerate(candidate_terms):
            best_idx = int(np.argmax(probs[i]))
            best_cat = classes[best_idx]
            conf = float(probs[i, best_idx])
            dist = {cls_name: round(float(probs[i, j]), 4) for j, cls_name in enumerate(classes)}
            results.append({
                'term': term,
                'predicted_mesh_cat': best_cat,
                'cat_confidence': round(conf, 4),
                'prob_distribution': dist
            })
        return results


class DistributionalSVD:
    """
    Embeddings Densos Clínicos treinados 100% do zero no corpus via PPMI + Truncated SVD.
    Implementação do Teorema de Levy & Goldberg (2014):
    A fatoração por SVD da matriz de coocorrência ponderada por PPMI aproxima o espaço semântico
    do Word2Vec Skip-Gram sem necessitar de modelos externos proibidos.
    """

    def __init__(self, embedding_dim: int = 64, window_size: int = 5, min_count: int = 2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.matrix_64d: np.ndarray = np.array([], dtype=np.float32)

    def fit_transform(self, tokenized_sentences: List[List[str]]) -> np.ndarray:
        """Gera a matriz esparsa Termo x Contexto, calcula PPMI e aplica Truncated SVD."""
        word_counts = Counter()
        for sent in tokenized_sentences:
            word_counts.update(w.lower() for w in sent if re.search(r'\w', w))

        self.idx2word = [w for w, c in word_counts.items() if c >= self.min_count]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        V = len(self.idx2word)

        if V == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cooc_counts = Counter()
        total_coocs = 0

        for sent in tokenized_sentences:
            clean = [w.lower() for w in sent if w.lower() in self.word2idx]
            for i, target in enumerate(clean):
                t_idx = self.word2idx[target]
                start = max(0, i - self.window_size)
                end = min(len(clean), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        c_idx = self.word2idx[clean[j]]
                        cooc_counts[(t_idx, c_idx)] += 1
                        total_coocs += 1

        if total_coocs == 0:
            self.matrix_64d = np.zeros((V, self.embedding_dim), dtype=np.float32)
            return self.matrix_64d

        row_marginal = np.zeros(V, dtype=np.float64)
        col_marginal = np.zeros(V, dtype=np.float64)
        for (i, j), count in cooc_counts.items():
            row_marginal[i] += count
            col_marginal[j] += count

        rows, cols, ppmi_data = [], [], []
        N = float(total_coocs)
        for (i, j), count in cooc_counts.items():
            expected = (row_marginal[i] * col_marginal[j]) / N
            if expected > 0:
                pmi = math.log2(count / expected)
                ppmi = max(0.0, pmi)
                if ppmi > 0:
                    rows.append(i)
                    cols.append(j)
                    ppmi_data.append(ppmi)

        ppmi_matrix = sp.csr_matrix((ppmi_data, (rows, cols)), shape=(V, V), dtype=np.float32)

        # Truncated SVD: k dimensões
        k = min(self.embedding_dim, max(1, V - 1))
        svd = TruncatedSVD(n_components=k, random_state=42)
        u_sigma = svd.fit_transform(ppmi_matrix)

        # Se k < embedding_dim, preenche com zeros
        if k < self.embedding_dim:
            pad = np.zeros((V, self.embedding_dim - k), dtype=np.float32)
            u_sigma = np.hstack([u_sigma, pad])

        # Normalização L2 por linha
        norms = np.linalg.norm(u_sigma, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix_64d = (u_sigma / norms).astype(np.float32)

        return self.matrix_64d

    def get_entity_vector(self, entity_text: str) -> np.ndarray:
        """Retorna o vetor denso normalizado (64d) médio das palavras da entidade."""
        words = [w.lower() for w in re.findall(r'\b\w+\b', entity_text) if w.lower() in self.word2idx]
        if not words or len(self.matrix_64d) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vecs = [self.matrix_64d[self.word2idx[w]] for w in words]
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.astype(np.float32)

    def save(self, filepath: str):
        """Salva vocabulário e matriz comprimida em arquivo .npz."""
        np.savez_compressed(
            filepath,
            vocab=np.array(self.idx2word, dtype=object),
            matrix_64d=self.matrix_64d
        )

    def load(self, filepath: str):
        """Carrega vocabulário e matriz a partir de .npz."""
        data = np.load(filepath, allow_pickle=True)
        self.idx2word = data['vocab'].tolist()
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.matrix_64d = data['matrix_64d']
        self.embedding_dim = self.matrix_64d.shape[1] if len(self.matrix_64d) > 0 else 64
