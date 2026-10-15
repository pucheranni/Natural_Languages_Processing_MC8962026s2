"""
weak_supervisor.py - Supervisão Fraca, N-Grams Posicionais e Trigger-Masking (Anti-Vazamento).
Fundamentado em:
- Ratner et al. (2017) - Snorkel / Weak Supervision
- Brin (1998) / Agichtein & Gravano (2000) - DIPRE / Snowball bootstrapping
- Slide nlp-080: Engenharia de Atributos, Particionamento de Dados e Evitação de Data Leakage
"""

import re
import random
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Verbos literais usados pelas regras determinísticas de Hearst no Projeto 1
P1_HEARST_TRIGGERS = {
    "revealed", "reveals", "revealing", "demonstrated", "demonstrates", "demonstrating",
    "showed", "shows", "showing", "indicated", "indicates", "noted", "identified", "found",
    "confirmed", "confirms", "confirming", "consistent with", "compatible with",
    "excluded", "excludes", "excluding", "ruled out", "no evidence of", "free of", "negative for",
    "suggesting the diagnosis of", "suggesting", "suggestive of", "diagnosis of",
    "thought to have arisen from", "treated with", "underwent", "started on",
    "maintained on", "referred for", "scheduled for", "planned", "prescribed", "administered",
    "located in", "situated in", "arising from"
}


class PositionalWeakSupervisor:
    """
    Extrator de N-Grams Posicionais com Trigger-Masking estocástico.
    Evita que classificadores de ML decorem os verbos-gatilho do Projeto 1,
    forçando-os a aprender as estruturas sintáticas e contextuais circundantes.
    """

    def __init__(
        self,
        mask_prob: float = 0.85,
        window_size: int = 3,
        seed: int = 42,
        triggers: Optional[Set[str]] = None
    ):
        self.mask_prob = mask_prob
        self.window_size = window_size
        self.rng = random.Random(seed)
        all_triggers = triggers if triggers is not None else P1_HEARST_TRIGGERS
        # Ordena por comprimento decrescente para priorizar locuções compostas (longest match)
        sorted_triggers = sorted(all_triggers, key=len, reverse=True)
        pattern = r"\b(" + "|".join(re.escape(t) for t in sorted_triggers) + r")\b"
        self.trigger_regex = re.compile(pattern, flags=re.IGNORECASE)

    def mask_p1_triggers(self, text: str, is_training: bool = True) -> Tuple[str, List[str]]:
        """Substitui estocasticamente verbos exatos do P1 por TRIGGER_MASK no treino."""
        if not is_training:
            return text, []

        masked_words = []

        def _replacer(match):
            trigger_text = match.group(0)
            if self.rng.random() < self.mask_prob:
                masked_words.append(trigger_text.lower())
                return "TRIGGER_MASK"
            return trigger_text

        masked_text = self.trigger_regex.sub(_replacer, text)
        return masked_text, masked_words

    def extract_positional_tokens(
        self,
        sentence_tokens: List[str],
        e1_token_span: Tuple[int, int],
        e2_token_span: Tuple[int, int],
        e1_type: str = "Entity",
        e2_type: str = "Entity",
        ngram_range: Tuple[int, int] = (1, 3),
        is_training: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Gera tokens prefixados por zona posicional (PRE_, BTW_, POST_),
        direção de leitura (DIR_L2R / DIR_R2L) e tipo de par (TYPE_E1_E2).
        """
        # Determina ordem linear na sentença
        if e1_token_span[0] <= e2_token_span[0]:
            first_span, second_span = e1_token_span, e2_token_span
            dir_token = "DIR_L2R"
            first_type, second_type = e1_type, e2_type
        else:
            first_span, second_span = e2_token_span, e1_token_span
            dir_token = "DIR_R2L"
            first_type, second_type = e2_type, e1_type

        # Zonas de contexto
        pre_toks = sentence_tokens[max(0, first_span[0] - self.window_size) : first_span[0]]
        btw_toks = sentence_tokens[first_span[1] : second_span[0]]
        post_toks = sentence_tokens[second_span[1] : second_span[1] + self.window_size]

        # Trigger-Masking aplicado no contexto intermediário durante treino
        btw_text = " ".join(btw_toks)
        masked_btw_text, masked_triggers = self.mask_p1_triggers(btw_text, is_training=is_training)
        btw_toks_masked = masked_btw_text.split()

        features = [dir_token, f"TYPE_{first_type}_{second_type}"]

        for zone_prefix, zone_toks in [
            ("PRE", pre_toks),
            ("BTW", btw_toks_masked),
            ("POST", post_toks)
        ]:
            min_n, max_n = ngram_range
            clean_toks = [t.lower() for t in zone_toks if t]
            for n in range(min_n, max_n + 1):
                for i in range(len(clean_toks) - n + 1):
                    gram = "_".join(clean_toks[i : i + n])
                    features.append(f"{zone_prefix}_{gram}")

        return " ".join(features), masked_triggers


class WeakSupervisionDatasetGenerator:
    """
    Constrói o dataset Silver Standard (relation_dataset.csv) combinando:
    1. Arestas positivas extraídas pelas regras do Projeto 1.
    2. Amostras negativas legítimas (NONE) com controle estrito de desbalanceamento.
    3. Particionamento 5-Fold Stratified Group K-Fold por case_id.
    """

    def __init__(
        self,
        negative_ratio: float = 2.0,
        max_token_dist: int = 25,
        mask_prob: float = 0.85,
        seed: int = 42
    ):
        self.negative_ratio = negative_ratio
        self.max_token_dist = max_token_dist
        self.supervisor = PositionalWeakSupervisor(mask_prob=mask_prob, seed=seed)
        self.seed = seed

    def build_dataset(
        self,
        cases_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        oov_nodes_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Processa cada caso clínico e gera todas as instâncias rotuladas e negativas.
        """
        # Junta nós do P1 com eventuais nós OOV promovidos
        all_nodes = nodes_df.copy()
        if oov_nodes_df is not None and not oov_nodes_df.empty:
            all_nodes = pd.concat([all_nodes, oov_nodes_df], ignore_index=True)

        # Mapeamento rápido de arestas positivas do P1: (case_id, source_id, target_id) -> relation
        p1_edge_map = {}
        for _, row in edges_df.iterrows():
            cid = str(row['case_id'])
            s_id = str(row['source_id'])
            t_id = str(row['target_id'])
            rel = str(row['relation'])
            # Ignora arestas estruturais temporais ou ontológicas da classificação de relação oracional
            if rel not in ('PRECEDES', 'INSTANCE_OF', 'HAS_HISTORY', 'PRESENTS_WITH', 'DENIES_OR_ABSENT', 'UNDERWENT_TEST', 'DIAGNOSED_WITH'):
                p1_edge_map[(cid, s_id, t_id)] = rel
                # Relações bidirecionais conhecidas
                if rel in ('LOCATED_IN', 'TARGETS', 'SUPPORTS', 'REVEALS', 'CONFIRMS', 'EXCLUDES', 'TREATED_WITH'):
                    p1_edge_map[(cid, s_id, t_id)] = rel

        instances = []
        pair_counter = 1
        rng = random.Random(self.seed)

        # Agrupa nós por case_id
        nodes_by_case = all_nodes.groupby('case_id')

        for case_id, case_group in nodes_by_case:
            case_id_str = str(case_id)
            case_text_rows = cases_df[cases_df['case_id'] == case_id]
            if case_text_rows.empty:
                continue
            case_text = case_text_rows.iloc[0]['case_text']

            # Filtra nós episódicos (ignora paciente e conceitos MeSH globais)
            episodic_nodes = case_group[
                ~case_group['type'].isin(['Patient', 'CanonicalConcept', 'History'])
            ].to_dict('records')

            # Agrupa nós por sentença (ou aproximação por offsets)
            # Para cada par de nós no mesmo caso clínico:
            n_episodic = len(episodic_nodes)
            positive_pairs_case = []
            candidate_negatives_case = []

            for i in range(n_episodic):
                for j in range(i + 1, n_episodic):
                    n1 = episodic_nodes[i]
                    n2 = episodic_nodes[j]

                    s1_idx = n1.get('sentence_idx', 0)
                    s2_idx = n2.get('sentence_idx', 0)

                    # Considera apenas entidades na mesma sentença ou em sentenças contíguas
                    if abs(s1_idx - s2_idx) > 1:
                        continue

                    n1_id = str(n1['node_id'])
                    n2_id = str(n2['node_id'])

                    # Checa se há aresta P1 em qualquer direção
                    rel_p1 = p1_edge_map.get((case_id_str, n1_id, n2_id))
                    is_reversed = False
                    if not rel_p1:
                        rel_p1 = p1_edge_map.get((case_id_str, n2_id, n1_id))
                        if rel_p1:
                            is_reversed = True

                    pair_info = {
                        'n1': n1 if not is_reversed else n2,
                        'n2': n2 if not is_reversed else n1,
                        'rel': rel_p1,
                        'sentence_idx': s1_idx
                    }

                    if rel_p1:
                        positive_pairs_case.append(pair_info)
                    else:
                        candidate_negatives_case.append(pair_info)

            # Subamostragem de negativos para evitar desbalanceamento excessivo
            max_negatives = int(len(positive_pairs_case) * self.negative_ratio) + 5
            selected_negatives = rng.sample(
                candidate_negatives_case,
                min(len(candidate_negatives_case), max_negatives)
            )

            # Processa todos os pares selecionados
            all_selected = [(p, p['rel']) for p in positive_pairs_case] + \
                           [(n, "NONE") for n in selected_negatives]

            for pair_data, label in all_selected:
                n_src = pair_data['n1']
                n_tgt = pair_data['n2']
                sent_idx = pair_data['sentence_idx']

                src_start = int(n_src.get('start_char', 0))
                src_end = int(n_src.get('end_char', src_start + len(str(n_src.get('label', '')))))
                tgt_start = int(n_tgt.get('start_char', 0))
                tgt_end = int(n_tgt.get('end_char', tgt_start + len(str(n_tgt.get('label', '')))))

                # Trecho de contexto na sentença
                min_char = min(src_start, tgt_start)
                max_char = max(src_end, tgt_end)
                char_start_ctx = max(0, min_char - 40)
                char_end_ctx = min(len(case_text), max_char + 40)
                raw_ctx = case_text[char_start_ctx:char_end_ctx].replace('\n', ' ')

                # Tokenização simples de palavras para spans
                ctx_tokens = re.findall(r'\b[\w\-]+\b', raw_ctx)
                src_text = str(n_src.get('label', '')).lower()
                tgt_text = str(n_tgt.get('label', '')).lower()

                # Localiza índices dos tokens
                src_tok_idx = 0
                tgt_tok_idx = min(len(ctx_tokens) - 1, 1)
                for tidx, tok in enumerate(ctx_tokens):
                    if src_text.startswith(tok.lower()):
                        src_tok_idx = tidx
                    if tgt_text.startswith(tok.lower()):
                        tgt_tok_idx = tidx

                dist_tokens = abs(src_tok_idx - tgt_tok_idx)
                if dist_tokens > self.max_token_dist and label == "NONE":
                    continue

                e1_span = (src_tok_idx, src_tok_idx + 1)
                e2_span = (tgt_tok_idx, tgt_tok_idx + 1)

                masked_pos_str, masked_triggers = self.supervisor.extract_positional_tokens(
                    sentence_tokens=ctx_tokens,
                    e1_token_span=e1_span,
                    e2_token_span=e2_span,
                    e1_type=str(n_src.get('type', 'Entity')),
                    e2_type=str(n_tgt.get('type', 'Entity')),
                    is_training=True
                )

                instances.append({
                    'pair_id': f"PAIR_{case_id_str}_{pair_counter:05d}",
                    'case_id': case_id_str,
                    'sentence_idx': sent_idx,
                    'e1_id': str(n_src.get('node_id', '')),
                    'e1_text': str(n_src.get('label', '')),
                    'e1_type': str(n_src.get('type', '')),
                    'e2_id': str(n_tgt.get('node_id', '')),
                    'e2_text': str(n_tgt.get('label', '')),
                    'e2_type': str(n_tgt.get('type', '')),
                    'dir_flag': "DIR_L2R" if src_start <= tgt_start else "DIR_R2L",
                    'token_dist': dist_tokens,
                    'raw_context': raw_ctx[:200],
                    'masked_positional_tokens': masked_pos_str,
                    'silver_label': label,
                    'trigger_word_masked': ";".join(masked_triggers) if masked_triggers else "NONE"
                })
                pair_counter += 1

        df_result = pd.DataFrame(instances)
        if df_result.empty:
            return df_result

        # Adiciona coluna de Fold via StratifiedGroupKFold (evita data leakage por paciente)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
        df_result['fold'] = -1

        # Trata classes com contagem muito baixa para o StratifiedGroupKFold não quebrar
        y_strat = df_result['silver_label'].copy()
        class_counts = y_strat.value_counts()
        rare_classes = class_counts[class_counts < 5].index
        y_strat[y_strat.isin(rare_classes)] = "OTHER"

        for fold_idx, (_, val_idx) in enumerate(sgkf.split(df_result, y_strat, groups=df_result['case_id'])):
            df_result.loc[val_idx, 'fold'] = fold_idx

        return df_result
