"""
test_phase4_mlp.py - Protocolo de Testes para a Fase 4 (MLP 2-Camadas em PyTorch, Gradientes e Determinismo).
"""

import sys
import numpy as np
import torch
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from project2.src.mlp_classifier import TwoLayerClinicalMLP, ClinicalNeuralPipeline


def test_overfit_single_batch_gradient_check():
    """Teste Unitário 4.1: Overfit em Mini-Batch (Verificação do Grafo de Gradientes - Slide 100/105)."""
    torch.manual_seed(42)
    input_dim = 20
    num_classes = 3
    model = TwoLayerClinicalMLP(input_dim=input_dim, num_classes=num_classes, dropout1=0.0, dropout2=0.0)

    # Mini batch sintético fixo (16 exemplos)
    x = torch.randn(16, input_dim)
    y = torch.randint(0, num_classes, (16,))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    initial_loss = criterion(model(x), y).item()
    for _ in range(70):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 0.10, f"MLP falhou em convergir em mini-batch: initial={initial_loss}, final={final_loss}"


def test_inference_determinism_eval_mode():
    """Teste Unitário 4.2: Determinismo em Inferência (Dropout Desativado no Eval)."""
    model = TwoLayerClinicalMLP(input_dim=15, num_classes=4, dropout1=0.5, dropout2=0.5)
    model.eval()

    x = torch.randn(5, 15)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2), "Saída em eval() não foi determinística (Dropout ainda ativo!)."


def test_gradient_flow_non_zero():
    """Teste Unitário 4.3: Fluxo de Gradientes Não-Nulos em Todas as Camadas."""
    model = TwoLayerClinicalMLP(input_dim=10, num_classes=3)
    x = torch.randn(8, 10)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    criterion = torch.nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradiente nulo em {name}"
            assert not torch.isnan(param.grad).any(), f"NaN detectado no gradiente de {name}"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0.0, f"Gradiente zerado em {name}"


def test_pipeline_integration():
    """Teste de Integração 4.4: Pipeline Neural Completo (Fit + Predict Proba)."""
    texts = [
        "DIR_L2R BTW_revealed BTW_tumor POST_lung",
        "DIR_L2R BTW_consistent_with BTW_infection",
        "DIR_L2R BTW_treated_with BTW_antibiotic",
        "DIR_L2R BTW_unrelated_finding"
    ] * 5
    labels = ["REVEALS", "SUPPORTS", "TREATED_WITH", "NONE"] * 5

    pipeline = ClinicalNeuralPipeline(input_mode='tfidf_sparse', epochs=10, batch_size=8)
    pipeline.fit(texts, labels)

    probs = pipeline.predict_proba(texts[:4])
    assert probs.shape == (4, 4)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    preds = pipeline.predict(texts[:4])
    assert len(preds) == 4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
