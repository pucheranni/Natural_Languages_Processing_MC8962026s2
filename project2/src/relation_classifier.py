"""
relation_classifier.py - Torneio de Modelos Clássicos, Estudo de Ablação e Explicabilidade XAI.
Fundamentado em:
- Jurafsky & Martin (2026), caps. 4, 5
- Slide nlp-080: Naive Bayes (Laplace Smoothing), Regressão Logística, Espaço Vetorial TF-IDF,
  Métricas de Avaliação (Precision, Recall, F1 Macro/Micro, Matriz de Confusão)
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score


class ClinicalRelationClassifier:
    """
    Orquestrador de classificação de relações clínicas supervisionadas/fracamente supervisionadas.
    Suporta 4 famílias clássicas:
    - Naive Bayes Multinomial
    - Regressão Logística L2 (com explicabilidade direta via pesos beta * tfidf)
    - Linear SVM (Calibrado com Platt Scaling)
    - Random Forest
    """

    SUPPORTED_MODELS = ['naive_bayes', 'logistic_regression', 'linear_svm', 'random_forest', 'hierarchical_cascade']

    def __init__(
        self,
        model_name: str = 'logistic_regression',
        ngram_range: Tuple[int, int] = (1, 3),
        weighting: str = 'tfidf',
        max_features: int = 5000,
        min_df: int = 2,
        seed: int = 42
    ):
        self.model_name = model_name
        self.ngram_range = ngram_range
        self.weighting = weighting
        self.max_features = max_features
        self.min_df = min_df
        self.seed = seed

        # Vectorizer
        if weighting == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                sublinear_tf=True
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df
            )

        self.model = self._create_model(model_name)
        self.classes_: np.ndarray = np.array([])
        self.is_fitted = False

    def _create_model(self, name: str):
        if name == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        elif name == 'logistic_regression':
            return LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.seed
            )
        elif name == 'linear_svm':
            # LinearSVC direto: probabilidades calculadas via softmax sobre decision_function
            return LinearSVC(C=1.0, class_weight='balanced', random_state=self.seed, dual='auto')
        elif name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight='balanced',
                random_state=self.seed
            )
        elif name == 'hierarchical_cascade':
            from project2.src.genius_upgrades import HierarchicalCascadeClassifier
            return HierarchicalCascadeClassifier(
                none_label='NONE',
                c_stage1=1.5,
                c_stage2=2.5,
                link_threshold=0.42
            )
        else:
            raise ValueError(f"Modelo não suportado: {name}. Opções: {self.SUPPORTED_MODELS}")

    def fit(self, texts: List[str], labels: List[str]):
        """Ajusta vetorizador e classificador."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = np.array(self.model.classes_)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Gera distribuição de probabilidade sobre as classes."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado.")
        X = self.vectorizer.transform(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            if df.ndim == 1:
                prob1 = 1.0 / (1.0 + np.exp(-df))
                return np.column_stack([1.0 - prob1, prob1])
            # Softmax sobre decision function se não houver predict_proba
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            raise NotImplementedError("Modelo não suporta estimativa de probabilidades.")

    def predict(self, texts: List[str]) -> np.ndarray:
        """Prediz classe mais provável."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado.")
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def explain_prediction(
        self,
        instance_text: str,
        predicted_class: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Explicabilidade Local (XAI Linear):
        Calcula contribuição exata beta_i * x_i para cada N-gram presente no texto.
        Gera os Top-K N-grams positivos e negativos para inspeção no visualizador.
        """
        if self.model_name != 'logistic_regression' or not self.is_fitted:
            return {'top_positive': [], 'top_negative': [], 'formula': "XAI disponível para Regressão Logística."}

        X_row = self.vectorizer.transform([instance_text])
        probs = self.model.predict_proba(X_row)[0]

        if predicted_class is None:
            pred_idx = int(np.argmax(probs))
            predicted_class = str(self.classes_[pred_idx])
        else:
            pred_idx = int(np.where(self.classes_ == predicted_class)[0][0])

        feature_names = self.vectorizer.get_feature_names_out()
        weights = self.model.coef_[pred_idx]
        bias = float(self.model.intercept_[pred_idx])

        row_coo = X_row.tocoo()
        active_contributions = []

        total_linear_score = bias
        for feat_idx, val in zip(row_coo.col, row_coo.data):
            w = float(weights[feat_idx])
            contrib = w * float(val)
            total_linear_score += contrib
            active_contributions.append({
                'ngram': str(feature_names[feat_idx]),
                'weight': round(w, 4),
                'val': round(float(val), 4),
                'contribution': round(contrib, 4)
            })

        active_contributions.sort(key=lambda x: x['contribution'], reverse=True)
        top_pos = [item for item in active_contributions if item['contribution'] > 0][:top_k]
        top_neg = [item for item in reversed(active_contributions) if item['contribution'] < 0][:top_k]

        return {
            'predicted_class': predicted_class,
            'confidence': round(float(probs[pred_idx]), 4),
            'bias': round(bias, 4),
            'total_linear_score': round(total_linear_score, 4),
            'top_positive': top_pos,
            'top_negative': top_neg
        }


def run_cross_validation_ablation(
    df_dataset: pd.DataFrame,
    feature_col: str = 'masked_positional_tokens',
    target_col: str = 'silver_label',
    fold_col: str = 'fold',
    model_name: str = 'logistic_regression',
    ngram_range: Tuple[int, int] = (1, 3),
    weighting: str = 'tfidf',
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executa validação cruzada 5-Fold estrita usando os folds agrupados por paciente.
    Retorna métricas globais e por classe (Macro Precision, Recall, F1) e matriz de confusão.
    """
    folds = sorted(df_dataset[fold_col].unique())
    y_true_all = []
    y_pred_all = []
    y_probs_all = []

    all_classes = np.sort(df_dataset[target_col].unique())
    pos_classes = [c for c in all_classes if c != 'NONE']

    fold_f1_macros = []
    fold_f1_positives = []

    for val_fold in folds:
        train_df = df_dataset[df_dataset[fold_col] != val_fold]
        val_df = df_dataset[df_dataset[fold_col] == val_fold]

        clf = ClinicalRelationClassifier(
            model_name=model_name,
            ngram_range=ngram_range,
            weighting=weighting,
            seed=seed
        )

        clf.fit(train_df[feature_col].tolist(), train_df[target_col].tolist())
        preds = clf.predict(val_df[feature_col].tolist())
        probs = clf.predict_proba(val_df[feature_col].tolist())

        y_true_fold = val_df[target_col].tolist()
        y_true_all.extend(y_true_fold)
        y_pred_all.extend(preds)
        y_probs_all.extend(probs)

        _, _, f1_m_f, _ = precision_recall_fscore_support(
            y_true_fold, preds, labels=all_classes, average='macro', zero_division=0
        )
        _, _, f1_p_f, _ = precision_recall_fscore_support(
            y_true_fold, preds, labels=pos_classes, average='macro', zero_division=0
        )
        fold_f1_macros.append(float(f1_m_f))
        fold_f1_positives.append(float(f1_p_f))

    # Métricas agregadas (pooled OOF sobre todas as classes do dataset)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=all_classes, average='macro', zero_division=0
    )
    _, _, f1_positive, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=pos_classes, average='macro', zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average='micro', zero_division=0
    )
    acc = accuracy_score(y_true_all, y_pred_all)

    # Métricas por classe
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=all_classes, zero_division=0
    )

    per_class_results = {}
    for i, cls_name in enumerate(all_classes):
        per_class_results[cls_name] = {
            'precision': round(float(p_per[i]), 4),
            'recall': round(float(r_per[i]), 4),
            'f1': round(float(f1_per[i]), 4),
            'support': int(sup_per[i])
        }

    cm = confusion_matrix(y_true_all, y_pred_all, labels=all_classes)

    return {
        'model_name': model_name,
        'feature_col': feature_col,
        'target_col': target_col,
        'ngram_range': list(ngram_range),
        'weighting': weighting,
        'accuracy': round(float(acc), 4),
        'macro_precision': round(float(p_macro), 4),
        'macro_recall': round(float(r_macro), 4),
        'macro_f1': round(float(f1_macro), 4),
        'macro_f1_std': round(float(np.std(fold_f1_macros)), 4),
        'positive_f1': round(float(f1_positive), 4),
        'positive_f1_std': round(float(np.std(fold_f1_positives)), 4),
        'micro_f1': round(float(f1_micro), 4),
        'per_class': per_class_results,
        'confusion_matrix': cm.tolist(),
        'classes': all_classes.tolist()
    }
