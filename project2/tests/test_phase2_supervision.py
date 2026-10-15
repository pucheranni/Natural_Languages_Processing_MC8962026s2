"""
test_phase2_supervision.py - Protocolo de Testes para a Fase 2 (Supervisão Fraca, N-grams Posicionais e Trigger-Masking).
"""

import sys
import pandas as pd
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from project2.src.weak_supervisor import PositionalWeakSupervisor, WeakSupervisionDatasetGenerator


def test_trigger_masking_audit():
    """Teste Unitário 2.1: Auditoria de Mascaramento de Gatilhos (Trigger Blank-Out)."""
    supervisor = PositionalWeakSupervisor(mask_prob=1.0, seed=42)

    sentence = ["chest", "ct", "revealed", "severe", "pleural", "effusion"]
    e1_span = (0, 2)  # "chest ct"
    e2_span = (4, 6)  # "pleural effusion"

    # No modo treino com mask_prob=1.0, "revealed" DEVE ser mascarado para "TRIGGER_MASK"
    pos_str, masked = supervisor.extract_positional_tokens(
        sentence_tokens=sentence,
        e1_token_span=e1_span,
        e2_token_span=e2_span,
        is_training=True
    )

    assert "BTW_revealed" not in pos_str, "Vazamento crítico: 'BTW_revealed' encontrado nas features de treino!"
    assert "BTW_trigger_mask" in pos_str, "O token '[TRIGGER_MASK]' deveria estar presente no contexto intermediário."
    assert "revealed" in masked, "'revealed' deveria ter sido registrado na lista de gatilhos mascarados."

    # No modo inferência (is_training=False), o texto original não deve ser mascarado
    pos_str_eval, _ = supervisor.extract_positional_tokens(
        sentence_tokens=sentence,
        e1_token_span=e1_span,
        e2_token_span=e2_span,
        is_training=False
    )
    assert "BTW_revealed" in pos_str_eval, "No modo avaliação/inferência, 'revealed' deve ser preservado para leitura correta."


def test_directional_flag_and_zones():
    """Teste Unitário 2.2: Flags de Direcionalidade e Zonas Posicionais."""
    supervisor = PositionalWeakSupervisor(mask_prob=0.0, seed=42)

    # Caso 1: L2R (E1 antes de E2)
    tokens_l2r = ["fever", "and", "cough", "suggesting", "pneumonia"]
    pos_l2r, _ = supervisor.extract_positional_tokens(
        tokens_l2r, (0, 1), (4, 5), e1_type="Symptom", e2_type="Diagnosis", is_training=False
    )
    assert "DIR_L2R" in pos_l2r, "Deveria ser DIR_L2R quando E1 vem antes de E2."
    assert "TYPE_Symptom_Diagnosis" in pos_l2r

    # Caso 2: R2L (E2 antes de E1)
    tokens_r2l = ["pneumonia", "diagnosed", "after", "fever"]
    pos_r2l, _ = supervisor.extract_positional_tokens(
        tokens_r2l, (3, 4), (0, 1), e1_type="Symptom", e2_type="Diagnosis", is_training=False
    )
    assert "DIR_R2L" in pos_r2l, "Deveria ser DIR_R2L quando E1 vem depois de E2."


def test_patient_partitioning_disjoint():
    """Teste Unitário 2.3: Ausência de Data Leakage entre Pacientes nos Folds."""
    generator = WeakSupervisionDatasetGenerator(negative_ratio=1.0, seed=42)

    # Cria dados sintéticos com 10 pacientes
    cases_data = [
        {'case_id': f'CASE_{i:02d}', 'case_text': f'Patient {i} had abdominal pain. CT showed pancreatitis.'}
        for i in range(1, 11)
    ]
    cases_df = pd.DataFrame(cases_data)

    nodes_data = []
    edges_data = []
    for i in range(1, 11):
        cid = f'CASE_{i:02d}'
        nid1 = f'N_{cid}_01'
        nid2 = f'N_{cid}_02'
        nodes_data.append({'case_id': cid, 'node_id': nid1, 'label': 'CT', 'type': 'ExamInstance', 'start_char': 28, 'end_char': 30, 'sentence_idx': 0})
        nodes_data.append({'case_id': cid, 'node_id': nid2, 'label': 'pancreatitis', 'type': 'Diagnosis', 'start_char': 38, 'end_char': 50, 'sentence_idx': 0})
        edges_data.append({'case_id': cid, 'source_id': nid1, 'target_id': nid2, 'relation': 'REVEALS'})

    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)

    dataset_df = generator.build_dataset(cases_df, nodes_df, edges_df)
    assert not dataset_df.empty, "Dataset sintético gerado não pode ser vazio."
    assert 'fold' in dataset_df.columns, "Coluna 'fold' obrigatória."

    # Verifica para cada fold que os pacientes são 100% disjuntos
    for f in range(5):
        val_cases = set(dataset_df[dataset_df['fold'] == f]['case_id'])
        train_cases = set(dataset_df[dataset_df['fold'] != f]['case_id'])
        intersection = val_cases.intersection(train_cases)
        assert len(intersection) == 0, f"Data leakage detectado no Fold {f}: pacientes {intersection} no treino e teste!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
