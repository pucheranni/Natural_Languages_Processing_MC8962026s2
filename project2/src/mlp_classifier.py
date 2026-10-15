"""
mlp_classifier.py - Deep Learning Clássico: Multi-Layer Perceptron (MLP) de 2 Camadas.
Fundamentado em:
- Slide nlp-100 & nlp-105: Cross-Entropy Loss, Backpropagation, Gradient Descent, Learning Rate
- Slide nlp-110: Feedforward Neural Network (2-Layer com ReLU e Softmax, Jurafsky 2026),
  Dropout e Regularização L2 (Weight Decay)
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer


class TwoLayerClinicalMLP(nn.Module):
    """
    Rede Neural Feedforward de 2 Camadas Ocultas (Slide nlp-110):
    x (D) -> Linear(D, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> ReLU -> Dropout(0.2) -> Linear(64, K)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout1: float = 0.3,
        dropout2: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout2)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.drop1(self.relu1(self.fc1(x)))
        h2 = self.drop2(self.relu2(self.fc2(h1)))
        logits = self.fc3(h2)
        return logits


class ClinicalNeuralPipeline:
    """
    Orquestrador de Treinamento e Avaliação da Rede Neural MLP.
    Suporta dois modos de entrada:
    1. 'tfidf_sparse': Vetor TF-IDF de N-Grams Posicionais (dimensão ~2000)
    2. 'dense_hybrid_svd': Concatenação de embeddings densos PPMI-SVD 64d:
       [e(E1) || e(E2) || e(E1) * e(E2) || e(ctx)] (dimensão 256)
    """

    def __init__(
        self,
        input_mode: str = 'tfidf_sparse',
        max_features: int = 2000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 32,
        seed: int = 42
    ):
        self.input_mode = input_mode
        self.max_features = max_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=max_features,
            sublinear_tf=True
        )
        self.model: Optional[TwoLayerClinicalMLP] = None
        self.classes_: np.ndarray = np.array([])
        self.class2idx: Dict[str, int] = {}
        self.is_fitted = False
        self.train_history: List[Dict[str, float]] = []

    def _prepare_features(
        self,
        texts: List[str],
        svd_features: Optional[np.ndarray] = None,
        is_training: bool = True
    ) -> np.ndarray:
        if self.input_mode == 'tfidf_sparse':
            if is_training:
                X_mat = self.vectorizer.fit_transform(texts).toarray()
            else:
                X_mat = self.vectorizer.transform(texts).toarray()
            return X_mat.astype(np.float32)
        elif self.input_mode == 'dense_hybrid_svd':
            if svd_features is None:
                raise ValueError("svd_features obrigatório para modo 'dense_hybrid_svd'.")
            return svd_features.astype(np.float32)
        else:
            raise ValueError(f"Modo de entrada desconhecido: {self.input_mode}")

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        svd_features: Optional[np.ndarray] = None,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        val_svd_features: Optional[np.ndarray] = None
    ):
        """Treina a rede neural MLP com monitoramento de Cross-Entropy Loss."""
        self.classes_ = np.sort(np.unique(labels))
        self.class2idx = {c: i for i, c in enumerate(self.classes_)}
        num_classes = len(self.classes_)

        X_train = self._prepare_features(texts, svd_features, is_training=True)
        y_train = np.array([self.class2idx[l] for l in labels], dtype=np.int64)

        input_dim = X_train.shape[1]
        self.model = TwoLayerClinicalMLP(input_dim=input_dim, num_classes=num_classes)

        # Cálculo de pesos inversos de classe para balanceamento na perda
        class_counts = np.bincount(y_train, minlength=num_classes)
        total_samples = len(y_train)
        weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
        class_weights_t = torch.tensor(weights, dtype=torch.float32)

        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        X_val_t = None
        y_val_t = None
        if val_texts is not None and val_labels is not None:
            X_val = self._prepare_features(val_texts, val_svd_features, is_training=False)
            y_val = np.array([self.class2idx[l] for l in val_labels], dtype=np.int64)
            X_val_t = torch.tensor(X_val)
            y_val_t = torch.tensor(y_val)

        self.train_history = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_x)

            epoch_loss = running_loss / len(X_train)

            epoch_metrics = {'epoch': epoch + 1, 'train_loss': round(epoch_loss, 4)}

            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                    preds = torch.argmax(val_logits, dim=1).numpy()
                    acc = np.mean(preds == y_val_t.numpy())
                    epoch_metrics['val_loss'] = round(val_loss, 4)
                    epoch_metrics['val_accuracy'] = round(float(acc), 4)

            self.train_history.append(epoch_metrics)

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        texts: List[str],
        svd_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Gera probabilidades Softmax para cada instância."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("MLP não treinado.")

        self.model.eval()
        X = self._prepare_features(texts, svd_features, is_training=False)
        with torch.no_grad():
            logits = self.model(torch.tensor(X))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(
        self,
        texts: List[str],
        svd_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Prediz rótulo da classe com maior probabilidade."""
        probs = self.predict_proba(texts, svd_features)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
