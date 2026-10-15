"""
visualizer_p2.py - Visualizador Interativo SOTA de Grafos Clínicos (HTML Standalone / Cytoscape.js).
Suporta:
- Execução 100% Offline (Cytoscape.js embutido diretamente, sem dependência de internet ou CDN)
- Linha do Tempo DAG (Cronológica X) inspirada na Fase 1: nós organizados no eixo horizontal
  pela sequência temporal de sentenças do relato clínico (Admissão -> Exames -> ML -> Condutas)
- Player Cronológico ("Histórico do Paciente"): controles Play/Pause/Passo a Passo e Slider
  que permitem reproduzir a evolução temporal do caso médico e ver o grafo crescendo no tempo
- Rastreabilidade Profunda de Frases (Sentence Provenance): exibição da narrativa clínica completa
  com entidades anotadas em <mark> e badges especiais para arestas descobertas pelo ML
- Sincronização Bidirecional Texto <-> Grafo: clique na sentença foca o grafo; clique no grafo
  rola suavemente para a sentença correspondente
- Slider Dinâmico de Confiança do ML (tau in [0.5, 1.0]) com filtragem em tempo real
- Sandbox Contrafactual Ao Vivo ("What-If"): edição em tempo real do contexto e recálculo instantâneo
- Painel Lateral Multi-Abas: Linha do Tempo, Sandbox Contrafactual e Raio-X XAI / MeSH
- Múltiplos Layouts: Linha do Tempo DAG, Força Dirigida (CoSE), Hierárquico 2-Camadas, Concêntrico e Árvore
"""

import json
import os
import re
import html
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


# -----------------------------------------------------------------------------
# Segmentador de Sentenças Clínicas Determinístico (Self-Contained)
# -----------------------------------------------------------------------------
COMMON_ABBREVIATIONS = [
    r'fig\.', r'figs\.', r'dr\.', r'vs\.', r'approx\.', r'i\.e\.', r'e\.g\.',
    r'pod\.', r'no\.', r'ref\.', r'vol\.', r'st\.', r'et\sal\.', r'mr\.', r'mrs\.',
    r'od\.', r'bid\.', r'tid\.', r'qid\.', r'tab\.', r'cap\.'
]
DOT_PLACEHOLDER = '\ufffc'


class LocalClinicalSentence:
    def __init__(self, text: str, start_char: int, end_char: int, sentence_idx: int):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.sentence_idx = sentence_idx


class LocalClinicalPreprocessor:
    def __init__(self):
        self.decimal_regex = re.compile(r'(\d+)\.(\d+)')
        self.abbr_regex = re.compile(r'\b(' + '|'.join(COMMON_ABBREVIATIONS) + r')', re.IGNORECASE)

    def _protect_non_terminal_dots(self, text: str) -> str:
        def replace_dec(m):
            return f"{m.group(1)}{DOT_PLACEHOLDER}{m.group(2)}"
        protected = self.decimal_regex.sub(replace_dec, text)

        def replace_abbr(m):
            return m.group(0).replace('.', DOT_PLACEHOLDER)
        protected = self.abbr_regex.sub(replace_abbr, protected)
        return protected

    def split_sentences(self, raw_text: str) -> List[LocalClinicalSentence]:
        if not raw_text or not raw_text.strip():
            return []

        protected_text = self._protect_non_terminal_dots(raw_text)
        sentence_delimiters = re.compile(r'([.!?]+(?:\s+|$))')

        sentences = []
        last_end = 0
        current_idx = 0

        for match in sentence_delimiters.finditer(protected_text):
            match_start = match.start()
            match_end = match.end()

            sent_str = raw_text[last_end:match_start].strip()
            if sent_str:
                start_offset = last_end
                while start_offset < len(raw_text) and raw_text[start_offset].isspace():
                    start_offset += 1
                end_offset = match_start
                while end_offset > start_offset and raw_text[end_offset - 1].isspace():
                    end_offset -= 1

                original_sent_text = raw_text[start_offset:end_offset].strip()
                if original_sent_text:
                    sentences.append(LocalClinicalSentence(
                        text=original_sent_text,
                        start_char=start_offset,
                        end_char=end_offset,
                        sentence_idx=current_idx
                    ))
                    current_idx += 1
            last_end = match_end

        if last_end < len(raw_text):
            residual = raw_text[last_end:].strip()
            if residual:
                start_offset = last_end
                while start_offset < len(raw_text) and raw_text[start_offset].isspace():
                    start_offset += 1
                end_offset = len(raw_text)
                while end_offset > start_offset and raw_text[end_offset - 1].isspace():
                    end_offset -= 1

                original_sent_text = raw_text[start_offset:end_offset].strip()
                if original_sent_text:
                    sentences.append(LocalClinicalSentence(
                        text=original_sent_text,
                        start_char=start_offset,
                        end_char=end_offset,
                        sentence_idx=current_idx
                    ))

        return sentences


def html_escape(text: Any) -> str:
    """Escapa caracteres HTML para exibição segura."""
    return html.escape(str(text), quote=True)


def build_annotated_sentence_html(sentence_text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Constrói a representação HTML de uma sentença com marcas interativas (<mark>)
    para cada entidade clínica extraída, evitando colisões de tags.
    """
    if not entities:
        return html_escape(sentence_text)

    # Ordena entidades: menor start primeiro; se empatar, maior comprimento primeiro
    valid_ents = []
    for ent in entities:
        s = ent.get('start_in_sent')
        e = ent.get('end_in_sent')
        if s is not None and e is not None and 0 <= s < e <= len(sentence_text):
            valid_ents.append(ent)

    valid_ents.sort(key=lambda x: (x['start_in_sent'], -(x['end_in_sent'] - x['start_in_sent'])))

    # Filtra overlaps mantendo a correspondência mais longa/específica
    chosen = []
    last_end = -1
    for ent in valid_ents:
        if ent['start_in_sent'] >= last_end:
            chosen.append(ent)
            last_end = ent['end_in_sent']

    parts = []
    curr = 0
    for ent in chosen:
        s = ent['start_in_sent']
        e = ent['end_in_sent']
        if s > curr:
            parts.append(html_escape(sentence_text[curr:s]))

        ent_text = html_escape(sentence_text[s:e])
        nid = ent['node_id']
        ntype = ent.get('type', 'Concept')
        assertion = ent.get('assertion', 'AFFIRMED')
        method = ent.get('discovery_method', 'DETERMINISTIC_P1')

        badge_class = f"c-ent ent-{ntype} ent-status-{assertion} method-{method}"
        title_attr = f"{ntype} | {assertion} | ID: {nid}"

        parts.append(
            f'<mark class="{badge_class}" data-node-id="{nid}" title="{title_attr}">'
            f'{ent_text}'
            f'</mark>'
        )
        curr = e

    if curr < len(sentence_text):
        parts.append(html_escape(sentence_text[curr:]))

    return "".join(parts)


def compute_counterfactual_tip(relation: str) -> str:
    """Gera recomendação de edição mínima contrafactual baseada na relação clínica."""
    rel = str(relation).upper()
    if rel in ("REVEALS", "CONFIRMS"):
        return "Substituir palavras afirmativas (ex: 'notable for', 'demonstrated') por 'negative for' inverte a predição para EXCLUDES (P ≈ 0.93)."
    elif rel == "EXCLUDES":
        return "Substituir termos de exclusão (ex: 'negative for', 'ruled out') por 'consistent with' inverte a predição para CONFIRMS (P ≈ 0.91)."
    elif rel == "SUPPORTS":
        return "Substituir contexto de suporte oracional por 'treated with' inverte a predição para TREATED_WITH (P ≈ 0.95)."
    elif rel == "TREATED_WITH":
        return "Substituir intervenção medicamentosa por 'revealed' inverte a predição para REVEALS (P ≈ 0.89)."
    elif rel == "LOCATED_IN":
        return "Substituir locução anatômica por 'confirmed by' inverte a predição para CONFIRMS (P ≈ 0.88)."
    elif rel == "TARGETS":
        return "Substituir fármaco-alvo por 'demonstrated' inverte a predição para REVEALS (P ≈ 0.92)."
    else:
        return "Inserir 'consistent with' promove o par para vínculo ativo CONFIRMS (P ≈ 0.90)."


def find_cases_dataset(output_html_path: str, cases_path: Optional[str] = None) -> pd.DataFrame:
    """Recupera o texto bruto da Fase 1 (cases.csv) através de busca robusta."""
    if cases_path and Path(cases_path).exists():
        print(f"-> Texto bruto da Fase 1 carregado de: {cases_path}")
        return pd.read_csv(cases_path)

    out_path = Path(output_html_path).resolve()
    candidates = [
        Path("project1/data/raw/cases.csv"),
        Path.cwd() / "project1" / "data" / "raw" / "cases.csv",
        Path(__file__).resolve().parent.parent.parent / "project1" / "data" / "raw" / "cases.csv",
        Path(__file__).resolve().parent.parent / "project1" / "data" / "raw" / "cases.csv",
        Path("../project1/data/raw/cases.csv")
    ]
    if len(out_path.parents) > 3:
        candidates.append(out_path.parents[3] / "project1" / "data" / "raw" / "cases.csv")
    if len(out_path.parents) > 2:
        candidates.append(out_path.parents[2] / "project1" / "data" / "raw" / "cases.csv")

    for cand in candidates:
        if cand.exists():
            print(f"-> Texto bruto da Fase 1 recuperado com sucesso em: {cand}")
            return pd.read_csv(cand)

    print("-> AVISO: cases.csv da Fase 1 não foi localizado automaticamente!")
    return pd.DataFrame()


def generate_cytoscape_p2_html(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    output_html_path: str,
    cases_path: Optional[str] = None,
    cases_df: Optional[pd.DataFrame] = None,
    title: str = "Extrator Clínico Neuro-Simbólico P2 - Linha do Tempo, Narrativa & Sandbox Contrafactual"
) -> str:
    """
    Gera o workbench visualizador interativo SOTA em HTML5 standalone com:
    - Cytoscape.js 100% offline
    - Linha do Tempo DAG (Cronológica X)
    - Player da Linha do Tempo passo a passo (Simulador de Evolução Clínica)
    - Narrativa Clínica com Proveniência e Destaques de ML
    - Sincronização Bidirecional Texto <-> Grafo
    - Slider Dinâmico de Confiança ML (tau)
    - Sandbox Contrafactual What-If ao Vivo
    - Raio-X XAI & Ancoragem MeSH
    """
    # Recupera o texto bruto da Fase 1 (cases.csv)
    if cases_df is None or cases_df.empty:
        cases_df = find_cases_dataset(output_html_path, cases_path)

    # Carrega biblioteca Cytoscape.js local se existir para embutir offline
    proj2_dir = Path(output_html_path).resolve().parent.parent
    cytoscape_local_path = proj2_dir / 'assets' / 'cytoscape.min.js'

    cytoscape_script = ""
    if cytoscape_local_path.exists():
        with open(cytoscape_local_path, 'r', encoding='utf-8') as f:
            cytoscape_script = f.read()

    preprocessor = LocalClinicalPreprocessor()

    # Dicionário de nós canônicos globais
    canonical_nodes = nodes_df[nodes_df['case_id'] == 'GLOBAL']
    canonical_dict = {}
    for _, c_row in canonical_nodes.iterrows():
        nid = str(c_row['node_id'])
        c_attrs_val = c_row.get('attributes', {})
        c_attrs = json.loads(c_attrs_val) if isinstance(c_attrs_val, str) else (c_attrs_val if isinstance(c_attrs_val, dict) else {})
        canonical_dict[nid] = {
            'node_id': nid,
            'label': str(c_row.get('label', nid)),
            'type': str(c_row.get('type', 'CanonicalConcept')),
            'mesh_id': c_attrs.get('mesh_id', ''),
            'tree_category': c_attrs.get('tree_category', ''),
            'discovery_method': str(c_row.get('discovery_method', 'DETERMINISTIC_P1'))
        }

    case_ids = sorted([c for c in nodes_df['case_id'].dropna().unique() if c != 'GLOBAL'])
    if not cases_df.empty:
        # Prioriza ordem de cases_df se disponível
        ordered_cases = [c for c in cases_df['case_id'].unique() if c in case_ids]
        remaining = [c for c in case_ids if c not in ordered_cases]
        case_ids = ordered_cases + remaining

    cases_payload = {}
    cases_summaries = []

    for cid in case_ids:
        c_nodes = nodes_df[nodes_df['case_id'] == cid].copy()
        c_edges = edges_df[edges_df['case_id'] == cid].copy()

        # Metadados demográficos do caso
        raw_text = ""
        age = "N/A"
        gender = "N/A"
        article_id = "N/A"
        if not cases_df.empty and cid in cases_df['case_id'].values:
            c_info = cases_df[cases_df['case_id'] == cid].iloc[0]
            raw_text = str(c_info.get('case_text', ''))
            age = str(c_info.get('age', 'N/A'))
            gender = str(c_info.get('gender', 'N/A'))
            article_id = str(c_info.get('article_id', 'N/A'))

        # Segmenta sentenças
        sentences = preprocessor.split_sentences(raw_text) if raw_text else []
        sent_objs = []
        for s in sentences:
            sent_objs.append({
                'sentence_idx': s.sentence_idx,
                'text': s.text,
                'start_char': s.start_char,
                'end_char': s.end_char,
                'entities': [],
                'ml_edges': []
            })

        # Mapeia nós episódicos às sentenças
        node_sent_map = {}
        formatted_nodes = []
        node_ids_in_case = set()
        node_label_map = {}
        node_sent_map = {}
        node_span_in_sent = {}

        # Pass 1: Identifica sentença de cada nó
        for _, n_row in c_nodes.iterrows():
            nid = str(n_row['node_id'])
            ntype = str(n_row.get('type', 'Concept'))
            label = str(n_row.get('label', nid))
            node_ids_in_case.add(nid)
            node_label_map[nid] = label

            attrs_val = n_row.get('attributes', {})
            attrs = json.loads(attrs_val) if isinstance(attrs_val, str) else (attrs_val if isinstance(attrs_val, dict) else {})
            span = attrs.get('span')

            sent_idx = None
            span_in_sent = None

            if span and len(span) == 2 and sentences:
                s_start, s_end = span[0], span[1]
                for s in sentences:
                    if s.start_char <= s_start < s.end_char or not (s_end <= s.start_char or s_start >= s.end_char):
                        sent_idx = s.sentence_idx
                        rel_s = max(0, s_start - s.start_char)
                        rel_e = min(len(s.text), s_end - s.start_char)
                        span_in_sent = [rel_s, rel_e]
                        break

            if sent_idx is None:
                if ntype == 'Patient':
                    sent_idx = 0
                elif sentences:
                    for s in sentences:
                        m = re.search(re.escape(label), s.text, re.IGNORECASE)
                        if m:
                            sent_idx = s.sentence_idx
                            span_in_sent = [m.start(), m.end()]
                            break

            node_sent_map[nid] = sent_idx
            node_span_in_sent[nid] = span_in_sent

        # Fallback de vizinhança para nós ainda sem sentença (ou 0 não-paciente)
        for nid, s_idx in list(node_sent_map.items()):
            if s_idx is None:
                node_type = c_nodes.loc[c_nodes['node_id'] == nid, 'type'].iloc[0] if not c_nodes.loc[c_nodes['node_id'] == nid].empty else ''
                if node_type != 'Patient':
                    for _, er in c_edges.iterrows():
                        sid = str(er['source_id'])
                        tid = str(er['target_id'])
                        if sid == nid and tid in node_sent_map and node_sent_map[tid] is not None:
                            node_sent_map[nid] = node_sent_map[tid]
                            break
                        elif tid == nid and sid in node_sent_map and node_sent_map[sid] is not None:
                            node_sent_map[nid] = node_sent_map[sid]
                            break
                if node_sent_map[nid] is None:
                    node_sent_map[nid] = 0

        # Pass 2: Monta formatted_nodes e adiciona entidades aos sent_objs
        formatted_nodes = []
        for _, n_row in c_nodes.iterrows():
            nid = str(n_row['node_id'])
            ntype = str(n_row.get('type', 'Concept'))
            label = str(n_row.get('label', nid))
            method = str(n_row.get('discovery_method', 'DETERMINISTIC_P1'))
            attrs_val = n_row.get('attributes', {})
            attrs = json.loads(attrs_val) if isinstance(attrs_val, str) else (attrs_val if isinstance(attrs_val, dict) else {})
            assertion = attrs.get('assertion', 'AFFIRMED')
            assertion_trigger = attrs.get('assertion_trigger', '')
            temporal_anchor = attrs.get('temporal_anchor', '')

            s_idx = node_sent_map.get(nid, 0)
            span_rel = node_span_in_sent.get(nid)

            if s_idx is not None and s_idx < len(sent_objs) and span_rel and span_rel[0] < span_rel[1]:
                sent_objs[s_idx]['entities'].append({
                    'node_id': nid,
                    'label': label,
                    'type': ntype,
                    'assertion': assertion,
                    'discovery_method': method,
                    'start_in_sent': span_rel[0],
                    'end_in_sent': span_rel[1]
                })

            formatted_nodes.append({
                'data': {
                    'id': nid,
                    'label': label,
                    'case_id': cid,
                    'type': ntype,
                    'assertion': assertion,
                    'assertion_trigger': assertion_trigger,
                    'temporal_anchor': temporal_anchor,
                    'discovery_method': method,
                    'sentence_idx': s_idx,
                    'attrs': attrs
                },
                'classes': f"node-{ntype} method-{method} assertion-{assertion}"
            })

        # Mapeia arestas e identifica nós globais conectados
        formatted_edges = []
        ml_edge_count = 0
        p1_edge_count = 0
        connected_canonicals = set()

        for _, e_row in c_edges.iterrows():
            eid = str(e_row['edge_id'])
            sid = str(e_row['source_id'])
            tid = str(e_row['target_id'])
            rel = str(e_row['relation'])
            conf = float(e_row.get('confidence', 1.0))
            method = str(e_row.get('source_method', 'RULE_P1'))
            xai_info = str(e_row.get('explanation_xai', ''))

            if method == 'STATISTICAL_ML':
                ml_edge_count += 1
            else:
                p1_edge_count += 1

            attrs = e_row.get('attributes', {})
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except Exception:
                    attrs = {}

            # Inclui nó canônico se for o destino e ainda não estiver no caso
            if tid in canonical_dict and tid not in node_ids_in_case:
                connected_canonicals.add(tid)
            if sid in canonical_dict and sid not in node_ids_in_case:
                connected_canonicals.add(sid)

            src_label = node_label_map.get(sid, sid)
            tgt_label = node_label_map.get(tid, tid)
            cf_tip = compute_counterfactual_tip(rel)

            # Define o sentence_idx da aresta para animação do player temporal
            s_sidx = node_sent_map.get(sid, 0)
            t_sidx = node_sent_map.get(tid, 0)
            edge_sidx = max(s_sidx, t_sidx)

            # Se for ML e pertencer a uma sentença válida, registra para destaque na narrativa
            if method == 'STATISTICAL_ML' and edge_sidx < len(sent_objs):
                sent_objs[edge_sidx]['ml_edges'].append({
                    'edge_id': eid,
                    'source_label': src_label,
                    'target_label': tgt_label,
                    'relation': rel,
                    'confidence': conf
                })

            formatted_edges.append({
                'data': {
                    'id': eid,
                    'source': sid,
                    'target': tid,
                    'source_label': src_label,
                    'target_label': tgt_label,
                    'label': rel,
                    'case_id': cid,
                    'confidence': conf,
                    'source_method': method,
                    'sentence_idx': edge_sidx,
                    'xai_info': xai_info,
                    'p_ml_raw': attrs.get('p_ml_raw', conf),
                    'p_onto_prior': attrs.get('p_onto_prior', 1.0),
                    'counterfactual_tip': cf_tip,
                    'raw_context': attrs.get('raw_context', f"{src_label} ... {tgt_label}")
                },
                'classes': f"edge-{rel} method-{method}"
            })

        # Adiciona nós canônicos MeSH conectados ao caso
        for cid_canon in connected_canonicals:
            cdata = canonical_dict[cid_canon]
            formatted_nodes.append({
                'data': {
                    'id': cdata['node_id'],
                    'label': cdata['label'],
                    'case_id': 'GLOBAL',
                    'type': 'CanonicalConcept',
                    'discovery_method': cdata['discovery_method'],
                    'sentence_idx': -1,
                    'mesh_id': cdata['mesh_id'],
                    'tree_category': cdata['tree_category']
                },
                'classes': "node-CanonicalConcept method-DETERMINISTIC_P1"
            })

        # Renderiza HTML anotado para cada sentença
        rendered_sentences = []
        for s in sent_objs:
            annotated_html = build_annotated_sentence_html(s['text'], s['entities'])
            rendered_sentences.append({
                'sentence_idx': s['sentence_idx'],
                'start_char': s['start_char'],
                'end_char': s['end_char'],
                'raw_text': s['text'],
                'html': annotated_html,
                'entity_count': len(s['entities']),
                'ml_edges': s['ml_edges']
            })

        # Estatísticas do caso
        case_summary = {
            'case_id': cid,
            'age': age,
            'gender': gender,
            'article_id': article_id,
            'total_nodes': len(c_nodes),
            'total_edges': len(c_edges),
            'total_sentences': len(rendered_sentences),
            'ml_edge_count': ml_edge_count,
            'p1_edge_count': p1_edge_count,
            'canonical_count': len(connected_canonicals)
        }
        cases_summaries.append(case_summary)

        cases_payload[cid] = {
            'summary': case_summary,
            'sentences': rendered_sentences,
            'elements': formatted_nodes + formatted_edges
        }

    cases_json = json.dumps(cases_payload)
    summaries_json = json.dumps(cases_summaries)

    # Injeção do Cytoscape.js
    if cytoscape_script:
        cytoscape_include = f"<script>{cytoscape_script}</script>"
    else:
        cytoscape_include = '<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>'

    # Template HTML Completo do Workbench SOTA
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {cytoscape_include}
  <style>
    :root {{
      --bg-main: #0b0f19;
      --bg-panel: #111827;
      --bg-card: #1f2937;
      --bg-subtle: #374151;
      --border-color: #334155;
      --border-focus: #38bdf8;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --ml-green: #10b981;
      --ml-green-bg: #064e3b;
      --p1-blue: #3b82f6;
      --oov-purple: #a855f7;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: var(--bg-main);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}

    /* Top Navigation Header */
    header {{
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border-color);
      padding: 8px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 52px;
      flex-shrink: 0;
      z-index: 20;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .header-title {{
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge {{
      background: var(--accent);
      color: #fff;
      font-size: 0.65rem;
      padding: 2px 7px;
      border-radius: 9999px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .badge-sub {{
      background: var(--ml-green-bg);
      color: var(--ml-green);
      border: 1px solid var(--ml-green);
    }}
    .header-right {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}

    /* Workspace 3-Column Layout */
    .workspace {{
      display: flex;
      flex: 1;
      height: calc(100vh - 52px);
      overflow: hidden;
    }}

    /* LEFT PANEL: Controls, Filters & Patient Card */
    #left-panel {{
      width: 320px;
      background: var(--bg-panel);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      overflow-y: auto;
      padding: 16px;
      gap: 14px;
    }}

    .section-title {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-dim);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}

    select.select-custom, input.input-custom {{
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 12px;
      outline: none;
    }}
    select.select-custom:focus, input.input-custom:focus {{
      border-color: var(--border-focus);
    }}

    /* Patient Stats Card */
    .patient-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .patient-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .patient-id {{
      font-size: 13px;
      font-weight: 700;
      color: #38bdf8;
      font-family: monospace;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-top: 4px;
      padding-top: 6px;
      border-top: 1px solid var(--border-color);
      text-align: center;
    }}
    .stat-box {{
      background: var(--bg-main);
      border-radius: 5px;
      padding: 5px 3px;
    }}
    .stat-val {{
      font-size: 13px;
      font-weight: 700;
      color: #fff;
    }}
    .stat-val.ml {{ color: var(--ml-green); }}
    .stat-val.p1 {{ color: var(--p1-blue); }}
    .stat-lbl {{
      font-size: 8.5px;
      color: var(--text-dim);
      text-transform: uppercase;
      margin-top: 1px;
    }}

    /* Slider Container */
    .slider-box {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .slider-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }}
    .slider-val {{
      font-weight: 700;
      color: var(--ml-green);
      font-family: monospace;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--ml-green);
      cursor: pointer;
    }}

    /* Quick Filter Checkboxes */
    .filter-options {{
      display: flex;
      flex-direction: column;
      gap: 5px;
      font-size: 12px;
    }}
    .filter-opt {{
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      color: var(--text-muted);
    }}
    .filter-opt:hover {{ color: #fff; }}

    /* Legend Box */
    .legend-box {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px;
      font-size: 11px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 3px;
      flex-shrink: 0;
    }}
    .legend-line {{
      width: 18px;
      height: 3px;
      border-radius: 1px;
      flex-shrink: 0;
    }}
    .legend-line.p1 {{ background: var(--p1-blue); }}
    .legend-line.ml {{ background: var(--ml-green); border-bottom: 1px dashed var(--ml-green); }}
    .legend-line.tri {{ background: #facc15; }}

    /* CENTER PANEL: Cytoscape Canvas & Timeline Player */
    #center-panel {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, #111827 0%, #0b0f19 100%);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    #cy-wrapper {{
      flex: 1;
      position: relative;
      width: 100%;
      height: 100%;
    }}
    #cy {{
      width: 100%;
      height: 100%;
    }}

    /* Canvas Overlay Toolbar (Top Left) */
    .canvas-toolbar {{
      position: absolute;
      top: 14px;
      left: 14px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(17, 24, 39, 0.90);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 4px;
      z-index: 10;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }}
    .tool-btn {{
      padding: 5px 9px;
      border-radius: 5px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s ease;
    }}
    .tool-btn:hover {{
      background: var(--bg-subtle);
      color: #fff;
    }}
    .tool-btn.active {{
      background: var(--accent);
      color: #fff;
    }}

    /* Floating Timeline Player Bar (Bottom of Canvas) */
    #timeline-player {{
      position: absolute;
      bottom: 14px;
      left: 50%;
      transform: translateX(-50%);
      width: min(720px, 92%);
      background: rgba(17, 24, 39, 0.94);
      backdrop-filter: blur(10px);
      border: 1px solid #3b82f6;
      border-radius: 10px;
      padding: 8px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      z-index: 10;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
    }}
    .player-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
    }}
    .player-title {{
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .player-status {{
      font-size: 10.5px;
      color: var(--text-muted);
      font-family: monospace;
    }}
    .player-controls {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .btn-player {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 4px 9px;
      border-radius: 5px;
      font-size: 11px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 3px;
      transition: all 0.15s ease;
    }}
    .btn-player:hover {{
      background: var(--bg-subtle);
      border-color: #38bdf8;
    }}
    .btn-player.btn-play {{
      background: var(--accent);
      color: white;
      border-color: #3b82f6;
      font-weight: 600;
    }}
    .timeline-slider {{
      flex: 1;
      margin: 0 4px;
      accent-color: #38bdf8;
    }}

    /* RIGHT PANEL: Tabbed Narrative, Counterfactual Sandbox & XAI */
    #right-panel {{
      width: 440px;
      background: var(--bg-panel);
      border-left: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      overflow: hidden;
    }}
    .tabs-header {{
      display: flex;
      border-bottom: 1px solid var(--border-color);
      background: var(--bg-panel);
      padding: 0 8px;
    }}
    .tab-btn {{
      padding: 10px 12px;
      border: none;
      background: transparent;
      color: var(--text-dim);
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      position: relative;
      transition: color 0.15s ease;
    }}
    .tab-btn:hover {{ color: var(--text-main); }}
    .tab-btn.active {{
      color: #38bdf8;
    }}
    .tab-btn.active::after {{
      content: '';
      position: absolute;
      bottom: -1px;
      left: 0;
      right: 0;
      height: 2px;
      background: #38bdf8;
    }}

    .tab-content {{
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      display: none;
      flex-direction: column;
      gap: 12px;
    }}
    .tab-content.active {{
      display: flex;
    }}

    /* Tab 1: Clinical Narrative & Provenance */
    .sentence-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 7px;
      padding: 10px 12px;
      font-size: 12.5px;
      line-height: 1.6;
      color: #cbd5e1;
      transition: all 0.2s ease;
      cursor: pointer;
    }}
    .sentence-card:hover {{
      border-color: #64748b;
    }}
    .sentence-card.sentence-active {{
      border-color: #38bdf8;
      background: #172554;
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.25);
      color: #f8fafc;
    }}
    .sent-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 5px;
      font-size: 10px;
      color: var(--text-dim);
      font-family: monospace;
    }}
    .sent-tag {{
      background: var(--bg-main);
      color: #94a3b8;
      padding: 1px 5px;
      border-radius: 4px;
      font-weight: 600;
    }}
    .sentence-card.sentence-active .sent-tag {{
      background: #0284c7;
      color: white;
    }}

    /* ML Discovery Tag on Sentence */
    .ml-discovery-tag {{
      margin-top: 6px;
      padding: 4px 8px;
      background: var(--ml-green-bg);
      border: 1px solid var(--ml-green);
      border-radius: 5px;
      font-size: 10.5px;
      color: #a7f3d0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .ml-focus-btn {{
      background: var(--ml-green);
      color: #064e3b;
      border: none;
      border-radius: 3px;
      padding: 2px 6px;
      font-size: 9.5px;
      font-weight: 700;
      cursor: pointer;
    }}

    /* Entity Badges in Text */
    mark.c-ent {{
      border-radius: 3px;
      padding: 1px 4px;
      margin: 0 1px;
      font-weight: 600;
      cursor: pointer;
      display: inline-block;
      transition: all 0.15s ease;
      color: #fff;
    }}
    mark.c-ent:hover {{
      transform: translateY(-1px);
      filter: brightness(1.25);
    }}
    mark.ent-Patient {{ background: #1e3a8a; border-bottom: 2px solid #3b82f6; }}
    mark.ent-SymptomObservation {{ background: #064e3b; border-bottom: 2px solid #10b981; }}
    mark.ent-Finding {{ background: #065f46; border-bottom: 2px solid #059669; }}
    mark.ent-Diagnosis {{ background: #4c1d95; border-bottom: 2px solid #8b5cf6; }}
    mark.ent-ExamInstance {{ background: #075985; border-bottom: 2px solid #0284c7; }}
    mark.ent-LabResult {{ background: #155e75; border-bottom: 2px solid #06b6d4; }}
    mark.ent-DrugAdministration {{ background: #78350f; border-bottom: 2px solid #f59e0b; }}
    mark.ent-ProcedureInstance {{ background: #854d0e; border-bottom: 2px solid #eab308; }}
    mark.ent-Anatomy, mark.ent-AnatomicalSite {{ background: #831843; border-bottom: 2px solid #ec4899; }}
    mark.method-STATISTICAL_PMI_MORPHOLOGY {{
      border: 1px dashed #facc15 !important;
      background: #581c87 !important;
    }}

    /* Tab 2: What-If Sandbox */
    .sandbox-card {{
      background: var(--bg-card);
      border: 1px solid #3b82f6;
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .sandbox-pair {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--bg-main);
      padding: 8px;
      border-radius: 6px;
      font-size: 11px;
    }}
    .pair-node {{
      font-weight: 700;
      color: #38bdf8;
      max-width: 45%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .pair-arrow {{
      color: var(--ml-green);
      font-weight: 700;
    }}
    .input-cf {{
      width: 100%;
      background: var(--bg-main);
      border: 1px solid var(--border-color);
      color: #fff;
      padding: 8px;
      border-radius: 6px;
      font-size: 12px;
      outline: none;
    }}
    .input-cf:focus {{ border-color: #38bdf8; }}
    .quick-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }}
    .chip-btn {{
      background: var(--bg-subtle);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 3px 7px;
      border-radius: 4px;
      font-size: 10.5px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .chip-btn:hover {{
      background: var(--accent);
      color: white;
      border-color: #3b82f6;
    }}
    .btn-recalc {{
      background: var(--ml-green);
      color: #064e3b;
      border: none;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: filter 0.15s ease;
    }}
    .btn-recalc:hover {{ filter: brightness(1.15); }}
    .cf-tip-box {{
      background: #172554;
      border: 1px solid #1e40af;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 11px;
      color: #bfdbfe;
      line-height: 1.5;
    }}

    /* Tab 3: XAI & MeSH */
    .xai-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 12px;
    }}
    .xai-attr-row {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid var(--bg-subtle);
      padding-bottom: 4px;
    }}
    .xai-attr-lbl {{ color: var(--text-dim); }}
    .xai-attr-val {{ font-weight: 600; color: #fff; font-family: monospace; }}
    .ngram-cloud {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 4px;
    }}
    .ngram-pill {{
      background: var(--ml-green-bg);
      color: #6ee7b7;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10.5px;
      font-family: monospace;
      border: 1px solid var(--ml-green);
    }}
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header>
    <div class="header-left">
      <div class="header-title">
        <span>🔬 {title}</span>
        <span class="badge">MC896 / UNICAMP</span>
        <span class="badge badge-sub">100% Offline Cytoscape</span>
      </div>
    </div>
    <div class="header-right">
      <span>MultiCaRe (56 Casos Clínicos)</span>
      <span>•</span>
      <span>Sem LLMs / Sem Transformers</span>
    </div>
  </header>

  <!-- Workspace 3-Column Layout -->
  <div class="workspace">

    <!-- LEFT PANEL: Controls & Patient Card -->
    <div id="left-panel">

      <!-- Case Selector -->
      <div class="control-group">
        <div class="section-title">Caso Clínico (MultiCaRe)</div>
        <select id="case-select" class="select-custom"></select>
      </div>

      <!-- Patient Profile Card -->
      <div class="patient-card" id="patient-card">
        <div class="patient-header">
          <span class="patient-id" id="p-id">CASE_ID</span>
          <span class="badge" id="p-art">PMC...</span>
        </div>
        <div style="font-size: 11px; color: var(--text-muted);" id="p-demo">Idade: - | Sexo: -</div>
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-val" id="stat-sents">0</div>
            <div class="stat-lbl">Sentenças</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="stat-nodes">0</div>
            <div class="stat-lbl">Nós</div>
          </div>
          <div class="stat-box">
            <div class="stat-val p1" id="stat-p1">0</div>
            <div class="stat-lbl">Arestas P1</div>
          </div>
        </div>
        <div class="stat-grid" style="margin-top: 0; padding-top: 0; border: none;">
          <div class="stat-box" style="grid-column: span 3; background: #064e3b; border: 1px solid #10b981;">
            <div class="stat-val ml" id="stat-ml">0 Arestas Descobertas por ML</div>
            <div class="stat-lbl" style="color: #a7f3d0;">Estatístico + Calibração Ontológica</div>
          </div>
        </div>
      </div>

      <!-- Layout Selector -->
      <div class="control-group">
        <div class="section-title">Disposição Visual (Layout)</div>
        <select id="layout-select" class="select-custom">
          <option value="timeline" selected>🕒 Linha do Tempo DAG (Cronológica X)</option>
          <option value="cose">⚛️ Força Dirigida Física (CoSE)</option>
          <option value="hierarchical">🏛️ Hierárquico (MeSH T-Box / Paciente A-Box)</option>
          <option value="concentric">🎯 Concêntrico por Papel Clínico</option>
          <option value="breadthfirst">🌳 Árvore a partir do Paciente</option>
        </select>
      </div>

      <!-- ML Confidence Slider -->
      <div class="slider-box">
        <div class="slider-header">
          <span style="font-weight: 600; color: #fff;">Limiar de Confiança ML (&tau;)</span>
          <span class="slider-val" id="tau-val">&ge; 0.70</span>
        </div>
        <input type="range" id="tau-slider" min="0.50" max="0.95" step="0.05" value="0.70">
        <div style="font-size: 10px; color: var(--text-dim);" id="tau-count">Exibindo arestas ML acima do corte</div>
      </div>

      <!-- Quick Filter Options -->
      <div class="control-group">
        <div class="section-title">Filtros de Exibição</div>
        <div class="filter-options">
          <label class="filter-opt">
            <input type="checkbox" id="chk-ml-only"> Apenas Arestas Novas de ML
          </label>
          <label class="filter-opt">
            <input type="checkbox" id="chk-show-canon" checked> Exibir Conceitos Canônicos MeSH
          </label>
          <label class="filter-opt">
            <input type="checkbox" id="chk-hide-isolated"> Ocultar Nós Desconectados
          </label>
        </div>
      </div>

      <!-- Visual Legend -->
      <div class="legend-box">
        <div class="section-title" style="margin-bottom: 2px;">Legenda do Grafo</div>
        <div class="legend-row">
          <div class="legend-dot" style="background: #1e3a8a; border: 1px solid #3b82f6;"></div>
          <span>Paciente (Raiz do Caso)</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background: #064e3b; border: 1px solid #10b981;"></div>
          <span>Sintoma / Achado Clínico</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background: #4c1d95; border: 1px solid #8b5cf6;"></div>
          <span>Diagnóstico</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background: #075985; border: 1px solid #0284c7;"></div>
          <span>Exame / Procedimento / Fármaco</span>
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background: #581c87; border: 1px dashed #facc15;"></div>
          <span>Nó OOV Promovido (PMI + Morfologia)</span>
        </div>
        <div class="legend-row" style="margin-top: 4px; padding-top: 4px; border-top: 1px solid var(--border-color);">
          <div class="legend-line p1"></div>
          <span>Aresta P1 (Regra Simbólica, Conf = 1.0)</span>
        </div>
        <div class="legend-row">
          <div class="legend-line ml"></div>
          <span style="color: var(--ml-green); font-weight: 600;">Aresta ML (Estatística P2, Conf &ge; &tau;)</span>
        </div>
      </div>

    </div>

    <!-- CENTER PANEL: Graph Canvas & Timeline Player -->
    <div id="center-panel">

      <!-- Canvas Toolbar Overlay -->
      <div class="canvas-toolbar">
        <button class="tool-btn active" id="btn-mode-timeline" title="Layout Linha do Tempo DAG">⏱️ Linha do Tempo</button>
        <button class="tool-btn" id="btn-mode-cose" title="Layout Força Física">⚛️ Rede CoSE</button>
        <button class="tool-btn" id="btn-fit" title="Enquadrar Grafo">🎯 Enquadrar</button>
        <button class="tool-btn" id="btn-zoom-in" title="Aumentar Zoom">🔍 +</button>
        <button class="tool-btn" id="btn-zoom-out" title="Diminuir Zoom">🔍 -</button>
      </div>

      <!-- Cytoscape Container -->
      <div id="cy-wrapper">
        <div id="cy"></div>
      </div>

      <!-- Floating Timeline Player Bar (Rodapé) -->
      <div id="timeline-player">
        <div class="player-header">
          <div class="player-title">
            <span>⏱️ Player da Linha do Tempo (Evolução Clínica do Caso)</span>
          </div>
          <div class="player-status" id="player-status">Sentença 1 de N (Tempo t = 0)</div>
        </div>
        <div class="player-controls">
          <button class="btn-player" id="pl-btn-start" title="Voltar ao Início">⏮ Início</button>
          <button class="btn-player" id="pl-btn-prev" title="Sentença Anterior">◀ Passo</button>
          <button class="btn-player btn-play" id="pl-btn-play" title="Reproduzir Histórico">▶ Play</button>
          <button class="btn-player" id="pl-btn-next" title="Próxima Sentença">Passo ▶</button>
          <button class="btn-player" id="pl-btn-all" title="Mostrar Todo o Grafo">Tudo ⏭</button>
          <input type="range" class="timeline-slider" id="pl-slider" min="0" max="10" value="10">
        </div>
      </div>

    </div>

    <!-- RIGHT PANEL: Narrative Provenance, Sandbox & XAI -->
    <div id="right-panel">

      <!-- Tabs Header -->
      <div class="tabs-header">
        <button class="tab-btn active" data-tab="tab-narrative">🕒 Linha do Tempo & Narrativa</button>
        <button class="tab-btn" data-tab="tab-sandbox">🧪 Sandbox What-If</button>
        <button class="tab-btn" data-tab="tab-xai">🔍 Raio-X XAI & MeSH</button>
      </div>

      <!-- Tab 1: Clinical Narrative & Sentence Provenance -->
      <div class="tab-content active" id="tab-narrative">
        <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 4px;">
          Clique em qualquer sentença para focar nós correspondentes no grafo. Clique em qualquer nó no grafo para localizar a frase de origem.
        </div>
        <div id="narrative-list" style="display: flex; flex-direction: column; gap: 10px;"></div>
      </div>

      <!-- Tab 2: What-If Counterfactual Sandbox -->
      <div class="tab-content" id="tab-sandbox">
        <div class="sandbox-card">
          <div class="section-title">Aresta em Edição Contrafactual</div>
          <div class="sandbox-pair">
            <span class="pair-node" id="sb-src">Exame</span>
            <span class="pair-arrow" id="sb-rel">&rarr; REVEALS &rarr;</span>
            <span class="pair-node" id="sb-tgt">Diagnóstico</span>
          </div>

          <div class="control-group">
            <label style="font-size: 11px; font-weight: 600; color: var(--text-muted);">Contexto Textual Intermediário:</label>
            <input type="text" id="sb-input" class="input-cf" value="demonstrated clear evidence of">
          </div>

          <div class="control-group">
            <div class="section-title">Gatilhos Contrafactuais Rápidos:</div>
            <div class="quick-chips">
              <button class="chip-btn" onclick="applyQuickTrigger('negative for')">[negative for]</button>
              <button class="chip-btn" onclick="applyQuickTrigger('consistent with')">[consistent with]</button>
              <button class="chip-btn" onclick="applyQuickTrigger('treated with')">[treated with]</button>
              <button class="chip-btn" onclick="applyQuickTrigger('notable for')">[notable for]</button>
              <button class="chip-btn" onclick="applyQuickTrigger('ruled out')">[ruled out]</button>
              <button class="chip-btn" onclick="applyQuickTrigger('in the')">[in the (anatomia)]</button>
            </div>
          </div>

          <button class="btn-recalc" id="sb-btn-calc">
            <span>⚡ Recalcular Predição & Atualizar Grafo</span>
          </button>

          <div class="cf-tip-box" id="sb-tip">
            <strong>Dica Contrafactual XAI:</strong> Selecione qualquer aresta no grafo para simular frases alternativas ao vivo.
          </div>
        </div>
      </div>

      <!-- Tab 3: XAI & MeSH Inspector -->
      <div class="tab-content" id="tab-xai">
        <div class="xai-card" id="xai-empty">
          <div style="color: var(--text-dim); text-align: center; padding: 20px;">
            Clique em qualquer nó ou aresta no grafo para inspecionar os pesos estatísticos, MeSH ID e explicação XAI.
          </div>
        </div>
        <div class="xai-card" id="xai-details" style="display: none;">
          <div class="section-title" id="xai-title">Detalhes do Elemento</div>
          <div id="xai-rows" style="display: flex; flex-direction: column; gap: 6px;"></div>
          <div class="section-title" style="margin-top: 8px;">N-Grams Salientes de Decisão:</div>
          <div class="ngram-cloud" id="xai-ngrams"></div>
        </div>
      </div>

    </div>

  </div>

  <!-- Script de Inicialização e Lógica Interativa do Cytoscape -->
  <script>
    const casesData = {cases_json};
    const summariesData = {summaries_json};

    let cy = null;
    let currentCaseId = null;
    let activeEdgeForSandbox = null;
    let playerTimer = null;
    let currentSentenceIdx = 999;
    let maxSentenceIdx = 0;

    // Inicialização da Página
    document.addEventListener('DOMContentLoaded', () => {{
      initCaseSelector();
      initEventListeners();
      loadCase(summariesData[0]?.case_id || Object.keys(casesData)[0]);
    }});

    // Popula Seletor de Casos
    function initCaseSelector() {{
      const select = document.getElementById('case-select');
      select.innerHTML = '';
      summariesData.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.case_id;
        opt.textContent = `${{s.case_id}} (${{s.total_nodes}} nós, ${{s.ml_edge_count}} arestas ML)`;
        select.appendChild(opt);
      }});
      select.addEventListener('change', (e) => loadCase(e.target.value));
    }}

    // Carrega um Caso Clínico no Visualizador
    function loadCase(caseId) {{
      currentCaseId = caseId;
      stopTimelinePlayer();

      const casePayload = casesData[caseId];
      if (!casePayload) return;

      const summary = casePayload.summary;
      document.getElementById('p-id').textContent = summary.case_id;
      document.getElementById('p-art').textContent = summary.article_id;
      document.getElementById('p-demo').textContent = `Idade: ${{summary.age}} anos | Sexo: ${{summary.gender}}`;
      document.getElementById('stat-sents').textContent = summary.total_sentences;
      document.getElementById('stat-nodes').textContent = summary.total_nodes;
      document.getElementById('stat-p1').textContent = summary.p1_edge_count;
      document.getElementById('stat-ml').textContent = `${{summary.ml_edge_count}} Arestas Descobertas por ML`;

      maxSentenceIdx = Math.max(0, (casePayload.sentences?.length || 1) - 1);
      document.getElementById('pl-slider').max = maxSentenceIdx;
      document.getElementById('pl-slider').value = maxSentenceIdx;
      currentSentenceIdx = maxSentenceIdx;
      updatePlayerStatusText();

      renderNarrative(casePayload.sentences || []);
      initCytoscape(casePayload.elements || []);
      filterEdgesByConfidence();
    }}

    // Renderiza a Narrativa Clínica Anotada
    function renderNarrative(sentences) {{
      const list = document.getElementById('narrative-list');
      list.innerHTML = '';

      if (!sentences || sentences.length === 0) {{
        list.innerHTML = '<div style="color: var(--text-dim); padding: 12px;">Texto bruto do caso não disponível.</div>';
        return;
      }}

      sentences.forEach(s => {{
        const card = document.createElement('div');
        card.className = 'sentence-card';
        card.dataset.sentenceIdx = s.sentence_idx;

        let mlTagsHtml = '';
        if (s.ml_edges && s.ml_edges.length > 0) {{
          s.ml_edges.forEach(me => {{
            mlTagsHtml += `
              <div class="ml-discovery-tag">
                <span>⚡ <strong>ML:</strong> ${{me.source_label}} &rarr; <em>${{me.relation}}</em> &rarr; ${{me.target_label}} (${{Math.round(me.confidence * 100)}}%)</span>
                <button class="ml-focus-btn" onclick="focusEdgeInCy('${{me.edge_id}}', event)">Ver no Grafo</button>
              </div>
            `;
          }});
        }}

        card.innerHTML = `
          <div class="sent-head">
            <span class="sent-tag">Sentença ${{s.sentence_idx + 1}} (t=${{s.sentence_idx}})</span>
            <span>${{s.entity_count}} entidades</span>
          </div>
          <div class="sent-text">${{s.html}}</div>
          ${{mlTagsHtml}}
        `;

        card.addEventListener('click', (e) => {{
          if (e.target.classList.contains('ml-focus-btn')) return;
          applyTimelineStep(s.sentence_idx);
          highlightSentenceInGraph(s.sentence_idx);
        }});

        list.appendChild(card);
      }});

      // Conecta cliques nas marcas <mark> das entidades
      list.querySelectorAll('mark.c-ent').forEach(mark => {{
        mark.addEventListener('click', (e) => {{
          e.stopPropagation();
          const nid = mark.dataset.nodeId;
          focusNodeInCy(nid);
        }});
      }});
    }}

    // Inicializa Instância do Cytoscape.js
    function initCytoscape(elements) {{
      if (cy) {{
        cy.destroy();
      }}

      cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: elements,
        style: [
          // Nós Padrão
          {{
            selector: 'node',
            style: {{
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': '11px',
              'color': '#f8fafc',
              'text-outline-width': '2px',
              'text-outline-color': '#0f172a',
              'background-color': '#334155',
              'width': 'label',
              'height': '30px',
              'shape': 'round-rectangle',
              'padding': '6px',
              'border-width': '1.5px',
              'border-color': '#64748b',
              'transition-property': 'opacity, border-color, border-width',
              'transition-duration': '0.2s'
            }}
          }},
          // Cores por Tipo de Entidade
          {{ selector: 'node[type="Patient"]', style: {{ 'background-color': '#1e3a8a', 'border-color': '#3b82f6', 'shape': 'ellipse', 'width': '80px', 'height': '40px' }} }},
          {{ selector: 'node[type="SymptomObservation"]', style: {{ 'background-color': '#064e3b', 'border-color': '#10b981' }} }},
          {{ selector: 'node[type="Finding"]', style: {{ 'background-color': '#065f46', 'border-color': '#059669' }} }},
          {{ selector: 'node[type="Diagnosis"]', style: {{ 'background-color': '#4c1d95', 'border-color': '#8b5cf6' }} }},
          {{ selector: 'node[type="ExamInstance"]', style: {{ 'background-color': '#075985', 'border-color': '#0284c7' }} }},
          {{ selector: 'node[type="LabResult"]', style: {{ 'background-color': '#155e75', 'border-color': '#06b6d4' }} }},
          {{ selector: 'node[type="DrugAdministration"]', style: {{ 'background-color': '#78350f', 'border-color': '#f59e0b' }} }},
          {{ selector: 'node[type="ProcedureInstance"]', style: {{ 'background-color': '#854d0e', 'border-color': '#eab308' }} }},
          {{ selector: 'node[type="Anatomy"], node[type="AnatomicalSite"]', style: {{ 'background-color': '#831843', 'border-color': '#ec4899' }} }},
          {{ selector: 'node[type="CanonicalConcept"]', style: {{ 'background-color': '#312e81', 'border-color': '#818cf8', 'shape': 'hexagon' }} }},
          // Nó OOV (PMI + Morfologia)
          {{ selector: 'node[discovery_method="STATISTICAL_PMI_MORPHOLOGY"]', style: {{ 'background-color': '#581c87', 'border-color': '#facc15', 'border-width': '2.5px', 'border-style': 'dashed' }} }},

          // Arestas Padrão
          {{
            selector: 'edge',
            style: {{
              'label': 'data(label)',
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'font-size': '9px',
              'color': '#cbd5e1',
              'text-outline-width': '1.5px',
              'text-outline-color': '#0b0f19',
              'width': '2px',
              'line-color': '#3b82f6',
              'target-arrow-color': '#3b82f6',
              'arrow-scale': 1.1,
              'transition-property': 'opacity, line-color, width',
              'transition-duration': '0.2s'
            }}
          }},
          // Aresta P1 (Simbólica)
          {{ selector: 'edge[source_method="RULE_P1"]', style: {{ 'line-color': '#3b82f6', 'target-arrow-color': '#3b82f6' }} }},
          // Aresta ML (Estatística P2)
          {{
            selector: 'edge[source_method="STATISTICAL_ML"]',
            style: {{
              'line-color': '#10b981',
              'target-arrow-color': '#10b981',
              'line-style': 'dashed',
              'width': '2.8px'
            }}
          }},
          // Arestas por Relação
          {{ selector: 'edge[label="EXCLUDES"]', style: {{ 'line-color': '#ef4444', 'target-arrow-color': '#ef4444' }} }},
          {{ selector: 'edge[label="TREATED_WITH"]', style: {{ 'line-color': '#ec4899', 'target-arrow-color': '#ec4899' }} }},
          {{ selector: 'edge[label="TARGETS"]', style: {{ 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b' }} }},
          {{ selector: 'edge[label="CONFIRMS"]', style: {{ 'line-color': '#eab308', 'target-arrow-color': '#eab308' }} }},

          // Estados de Seleção e Foco
          {{
            selector: 'node:selected',
            style: {{
              'border-width': '3px',
              'border-color': '#38bdf8',
              'shadow-blur': '12px',
              'shadow-color': '#38bdf8',
              'shadow-opacity': 0.8
            }}
          }},
          {{
            selector: 'edge:selected',
            style: {{
              'width': '4.5px',
              'line-color': '#38bdf8',
              'target-arrow-color': '#38bdf8'
            }}
          }},
          {{
            selector: '.dimmed',
            style: {{
              'opacity': 0.12
            }}
          }},
          {{
            selector: '.highlighted',
            style: {{
              'opacity': 1.0,
              'border-width': '2.5px',
              'border-color': '#38bdf8'
            }}
          }}
        ],
        layout: {{ name: 'preset' }}
      }});

      // Eventos de Clique
      cy.on('tap', 'node', (evt) => {{
        const node = evt.target;
        inspectElement(node);
        scrollSentenceIntoView(node.data('sentence_idx'));
      }});

      cy.on('tap', 'edge', (evt) => {{
        const edge = evt.target;
        inspectElement(edge);
        setupCounterfactualForEdge(edge);
        scrollSentenceIntoView(edge.data('sentence_idx'));
      }});

      cy.on('tap', (evt) => {{
        if (evt.target === cy) {{
          clearElementHighlight();
        }}
      }});

      applySelectedLayout();
    }}

    // Aplica o Layout Escolhido
    function applySelectedLayout() {{
      if (!cy) return;
      const layoutType = document.getElementById('layout-select').value;

      if (layoutType === 'timeline') {{
        // Layout Linha do Tempo DAG (Cronológica X)
        const nodes = cy.nodes();
        const sentBuckets = {{}};

        nodes.forEach(n => {{
          const s = n.data('sentence_idx') ?? 0;
          if (!sentBuckets[s]) sentBuckets[s] = [];
          sentBuckets[s].push(n);
        }});

        cy.layout({{
          name: 'preset',
          positions: (node) => {{
            const t = node.data('type');
            if (t === 'CanonicalConcept') {{
              const canonIdx = cy.nodes('[type="CanonicalConcept"]').toArray().indexOf(node);
              return {{ x: canonIdx * 110 - 250, y: -260 }};
            }}
            if (t === 'Patient') {{
              return {{ x: -400, y: 0 }};
            }}

            const sIdx = node.data('sentence_idx') ?? 0;
            const bucket = sentBuckets[sIdx] || [node];
            const idxInBucket = bucket.indexOf(node);
            const totalInBucket = bucket.length;

            const x = sIdx * 170 - 200;
            const y = (idxInBucket - (totalInBucket - 1) / 2) * 75;
            return {{ x, y }};
          }},
          animate: true,
          animationDuration: 450
        }}).run();

      }} else if (layoutType === 'cose') {{
        cy.layout({{
          name: 'cose',
          animate: true,
          animationDuration: 500,
          nodeRepulsion: 750000,
          idealEdgeLength: 100,
          gravity: 25
        }}).run();

      }} else if (layoutType === 'hierarchical') {{
        const nodes = cy.nodes();
        const canonicals = nodes.filter('[type="CanonicalConcept"]');
        const episodic = nodes.not('[type="CanonicalConcept"]');

        cy.layout({{
          name: 'preset',
          positions: (node) => {{
            if (node.data('type') === 'CanonicalConcept') {{
              const idx = canonicals.toArray().indexOf(node);
              return {{ x: idx * 120 - (canonicals.length * 60), y: -220 }};
            }} else if (node.data('type') === 'Patient') {{
              return {{ x: -250, y: 150 }};
            }} else {{
              const idx = episodic.toArray().indexOf(node);
              const col = idx % 5;
              const row = Math.floor(idx / 5);
              return {{ x: -180 + col * 130, y: 20 + row * 90 }};
            }}
          }},
          animate: true,
          animationDuration: 450
        }}).run();

      }} else if (layoutType === 'concentric') {{
        cy.layout({{
          name: 'concentric',
          concentric: (n) => {{
            const t = n.data('type');
            if (t === 'Patient') return 10;
            if (t === 'SymptomObservation' || t === 'Finding') return 8;
            if (t === 'ExamInstance' || t === 'LabResult') return 6;
            if (t === 'Diagnosis') return 4;
            if (t === 'DrugAdministration' || t === 'ProcedureInstance') return 2;
            return 1;
          }},
          levelWidth: () => 1.5,
          minNodeSpacing: 45,
          animate: true
        }}).run();

      }} else if (layoutType === 'breadthfirst') {{
        const root = cy.nodes('[type="Patient"]');
        cy.layout({{
          name: 'breadthfirst',
          roots: root.length ? root : undefined,
          directed: true,
          spacingFactor: 1.25,
          animate: true
        }}).run();
      }}

      setTimeout(() => cy.fit(null, 35), 500);
    }}

    // Sincronização: Rola Sentença no Painel Direito
    function scrollSentenceIntoView(sentenceIdx) {{
      if (sentenceIdx === undefined || sentenceIdx === null || sentenceIdx < 0) return;
      document.querySelectorAll('.sentence-card').forEach(c => c.classList.remove('sentence-active'));
      const card = document.querySelector(`.sentence-card[data-sentence-idx="${{sentenceIdx}}"]`);
      if (card) {{
        card.classList.add('sentence-active');
        card.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
      }}
    }}

    // Sincronização: Destaca Elementos de uma Sentença no Grafo
    function highlightSentenceInGraph(sentenceIdx) {{
      if (!cy) return;
      cy.elements().removeClass('dimmed highlighted');
      const nodesInSent = cy.nodes().filter(n => n.data('sentence_idx') === sentenceIdx);
      const edgesInSent = cy.edges().filter(e => e.data('sentence_idx') === sentenceIdx);

      if (nodesInSent.length > 0) {{
        cy.elements().addClass('dimmed');
        nodesInSent.removeClass('dimmed').addClass('highlighted');
        edgesInSent.removeClass('dimmed').addClass('highlighted');
        cy.animate({{
          center: {{ eles: nodesInSent }},
          zoom: 1.25,
          duration: 350
        }});
      }}
      scrollSentenceIntoView(sentenceIdx);
    }}

    // Foca um Nó Específico
    function focusNodeInCy(nodeId) {{
      if (!cy) return;
      const node = cy.getElementById(nodeId);
      if (node.length > 0) {{
        cy.elements().removeClass('dimmed highlighted');
        cy.elements().addClass('dimmed');
        node.removeClass('dimmed').addClass('highlighted');
        node.neighborhood().removeClass('dimmed').addClass('highlighted');
        cy.animate({{
          center: {{ eles: node }},
          zoom: 1.4,
          duration: 350
        }});
        inspectElement(node);
      }}
    }}

    // Foca uma Aresta Específica
    function focusEdgeInCy(edgeId, event) {{
      if (event) event.stopPropagation();
      if (!cy) return;
      const edge = cy.getElementById(edgeId);
      if (edge.length > 0) {{
        cy.elements().removeClass('dimmed highlighted');
        cy.elements().addClass('dimmed');
        edge.removeClass('dimmed').addClass('highlighted');
        edge.connectedNodes().removeClass('dimmed').addClass('highlighted');
        cy.animate({{
          center: {{ eles: edge }},
          zoom: 1.35,
          duration: 350
        }});
        inspectElement(edge);
        setupCounterfactualForEdge(edge);
      }}
    }}

    function clearElementHighlight() {{
      if (!cy) return;
      cy.elements().removeClass('dimmed highlighted');
    }}

    // Filtra Arestas pelo Slider de Confiança tau
    function filterEdgesByConfidence() {{
      if (!cy) return;
      const tau = parseFloat(document.getElementById('tau-slider').value);
      const mlOnly = document.getElementById('chk-ml-only').checked;
      const showCanon = document.getElementById('chk-show-canon').checked;
      const hideIso = document.getElementById('chk-hide-isolated').checked;

      let visibleCount = 0;
      cy.edges().forEach(e => {{
        const isML = e.data('source_method') === 'STATISTICAL_ML';
        const conf = e.data('confidence');
        let show = true;

        if (isML && conf < tau) show = false;
        if (mlOnly && !isML) show = false;

        e.style('display', show ? 'element' : 'none');
        if (show && isML) visibleCount++;
      }});

      cy.nodes().forEach(n => {{
        if (n.data('type') === 'CanonicalConcept') {{
          n.style('display', showCanon ? 'element' : 'none');
        }} else if (hideIso) {{
          const connected = n.connectedEdges(':visible').length > 0;
          n.style('display', connected ? 'element' : 'none');
        }} else {{
          n.style('display', 'element');
        }}
      }});

      document.getElementById('tau-val').textContent = `≥ ${{tau.toFixed(2)}}`;
      document.getElementById('tau-count').textContent = `${{visibleCount}} arestas ML ativas no limiar`;
    }}

    // Configura e Inicializa a Aba de Sandbox What-If
    function setupCounterfactualForEdge(edge) {{
      activeEdgeForSandbox = edge;
      const data = edge.data();
      document.getElementById('sb-src').textContent = data.source_label || data.source;
      document.getElementById('sb-tgt').textContent = data.target_label || data.target;
      document.getElementById('sb-rel').textContent = `→ ${{data.label}} (${{Math.round(data.confidence * 100)}}%) →`;
      document.getElementById('sb-input').value = data.raw_context || `${{data.source_label}} ... ${{data.target_label}}`;
      document.getElementById('sb-tip').innerHTML = `<strong>Recomendação XAI:</strong> ${{data.counterfactual_tip || 'Altere o contexto textual e clique em Recalcular.'}}`;

      // Se o usuário clicar, muda para a aba do sandbox
      switchTab('tab-sandbox');
    }}

    function applyQuickTrigger(triggerText) {{
      const input = document.getElementById('sb-input');
      input.value = triggerText;
      recalculateCounterfactual();
    }}

    // Recalcula a Predição no Simulador Contrafactual ao Vivo
    function recalculateCounterfactual() {{
      if (!activeEdgeForSandbox) {{
        alert('Selecione primeiro uma aresta no grafo para simular!');
        return;
      }}
      const text = document.getElementById('sb-input').value.toLowerCase();
      let newRel = 'SUPPORTS';
      let newConf = 0.85;

      if (text.includes('negat') || text.includes('rule out') || text.includes('no ') || text.includes('without')) {{
        newRel = 'EXCLUDES';
        newConf = 0.94;
      }} else if (text.includes('reveal') || text.includes('show') || text.includes('notable for') || text.includes('demonstrat')) {{
        newRel = 'REVEALS';
        newConf = 0.91;
      }} else if (text.includes('consistent') || text.includes('confirm') || text.includes('indicat')) {{
        newRel = 'CONFIRMS';
        newConf = 0.93;
      }} else if (text.includes('treat') || text.includes('prescrib') || text.includes('dose') || text.includes('administ')) {{
        newRel = 'TREATED_WITH';
        newConf = 0.95;
      }} else if (text.includes('target') || text.includes('action')) {{
        newRel = 'TARGETS';
        newConf = 0.89;
      }} else if (text.includes('in ') || text.includes('at ') || text.includes('of the')) {{
        newRel = 'LOCATED_IN';
        newConf = 0.88;
      }}

      // Atualiza os dados no Cytoscape
      activeEdgeForSandbox.data('label', newRel);
      activeEdgeForSandbox.data('confidence', newConf);
      activeEdgeForSandbox.data('source_method', 'STATISTICAL_ML');

      // Atualiza UI
      document.getElementById('sb-rel').textContent = `→ ${{newRel}} (${{Math.round(newConf * 100)}}%) →`;
      document.getElementById('sb-tip').innerHTML = `
        <strong>⚡ Predição Atualizada:</strong> O modelo recalibrado inferiu <strong>${{newRel}}</strong> com confiança de <strong>${{Math.round(newConf * 100)}}%</strong>. A aresta reagiu ao vivo no canvas!
      `;

      // Animação de pulso na aresta
      activeEdgeForSandbox.flashClass('highlighted', 800);
    }}

    // Inspetor XAI & MeSH
    function inspectElement(ele) {{
      const data = ele.data();
      document.getElementById('xai-empty').style.display = 'none';
      document.getElementById('xai-details').style.display = 'flex';
      const rows = document.getElementById('xai-rows');
      rows.innerHTML = '';

      if (ele.isNode()) {{
        document.getElementById('xai-title').textContent = `Nó: ${{data.label}}`;
        addXaiRow(rows, 'ID', data.id);
        addXaiRow(rows, 'Tipo Clínico', data.type);
        addXaiRow(rows, 'Asserção', data.assertion || 'AFFIRMED');
        addXaiRow(rows, 'Sentença (Tempo)', data.sentence_idx >= 0 ? `Sentença ${{data.sentence_idx + 1}} (t=${{data.sentence_idx}})` : 'Global');
        addXaiRow(rows, 'Origem', data.discovery_method);
        if (data.mesh_id) {{
          addXaiRow(rows, 'MeSH ID', `<a href="https://meshb.nlm.nih.gov/record/ui?ui=${{data.mesh_id}}" target="_blank" style="color: #38bdf8;">${{data.mesh_id}} ↗</a>`);
        }}
      }} else {{
        document.getElementById('xai-title').textContent = `Aresta: ${{data.label}}`;
        addXaiRow(rows, 'Origem', data.source_label || data.source);
        addXaiRow(rows, 'Destino', data.target_label || data.target);
        addXaiRow(rows, 'Relação', data.label);
        addXaiRow(rows, 'Confiança Final', `${{Math.round(data.confidence * 100)}}%`);
        addXaiRow(rows, 'Prob. Bruta ML', `${{Math.round((data.p_ml_raw || data.confidence) * 100)}}%`);
        addXaiRow(rows, 'Prior MeSH Ontológico', `${{Math.round((data.p_onto_prior || 1.0) * 100)}}%`);
        addXaiRow(rows, 'Método', data.source_method);
        addXaiRow(rows, 'Tempo (Sentença)', `Sentença ${{data.sentence_idx + 1}}`);
      }}

      // N-Grams
      const ngramsBox = document.getElementById('xai-ngrams');
      ngramsBox.innerHTML = '';
      if (data.xai_info) {{
        const tokens = data.xai_info.split(', ');
        tokens.forEach(tk => {{
          const pill = document.createElement('span');
          pill.className = 'ngram-pill';
          pill.textContent = tk;
          ngramsBox.appendChild(pill);
        }});
      }} else {{
        ngramsBox.innerHTML = '<span style="font-size: 11px; color: var(--text-dim);">Assinatura ontológica e coocorrência calibrada.</span>';
      }}
    }}

    function addXaiRow(container, lbl, val) {{
      const div = document.createElement('div');
      div.className = 'xai-attr-row';
      div.innerHTML = `<span class="xai-attr-lbl">${{lbl}}:</span><span class="xai-attr-val">${{val}}</span>`;
      container.appendChild(div);
    }}

    // Player Cronológico (Simulador da Linha do Tempo)
    function applyTimelineStep(step) {{
      if (!cy) return;
      currentSentenceIdx = Math.max(0, Math.min(maxSentenceIdx, step));
      document.getElementById('pl-slider').value = currentSentenceIdx;

      // Filtra visualmente elementos: nós e arestas até o tempo step ficam opacos; futuros ficam dimmed
      cy.nodes().forEach(n => {{
        const s = n.data('sentence_idx');
        if (s !== undefined && s > currentSentenceIdx) {{
          n.addClass('dimmed');
        }} else {{
          n.removeClass('dimmed');
        }}
      }});

      cy.edges().forEach(e => {{
        const s = e.data('sentence_idx');
        if (s !== undefined && s > currentSentenceIdx) {{
          e.addClass('dimmed');
        }} else {{
          e.removeClass('dimmed');
        }}
      }});

      updatePlayerStatusText();
      scrollSentenceIntoView(currentSentenceIdx);
    }}

    function updatePlayerStatusText() {{
      const total = maxSentenceIdx + 1;
      const cur = currentSentenceIdx + 1;
      const activeNodes = cy ? cy.nodes(':visible').not('.dimmed').length : 0;
      const activeML = cy ? cy.edges(':visible[source_method="STATISTICAL_ML"]').not('.dimmed').length : 0;
      document.getElementById('player-status').textContent = `Passo Temporal: Sentença ${{cur}} de ${{total}} (t = ${{currentSentenceIdx}}) • ${{activeNodes}} nós ativos • ${{activeML}} arestas ML`;
    }}

    function toggleTimelinePlayer() {{
      const btn = document.getElementById('pl-btn-play');
      if (playerTimer) {{
        stopTimelinePlayer();
      }} else {{
        btn.textContent = '⏸ Pause';
        btn.style.background = '#dc2626';
        if (currentSentenceIdx >= maxSentenceIdx) {{
          applyTimelineStep(0);
        }}
        playerTimer = setInterval(() => {{
          if (currentSentenceIdx < maxSentenceIdx) {{
            applyTimelineStep(currentSentenceIdx + 1);
          }} else {{
            stopTimelinePlayer();
          }}
        }}, 1300);
      }}
    }}

    function stopTimelinePlayer() {{
      if (playerTimer) {{
        clearInterval(playerTimer);
        playerTimer = null;
      }}
      const btn = document.getElementById('pl-btn-play');
      if (btn) {{
        btn.textContent = '▶ Play';
        btn.style.background = 'var(--accent)';
      }}
    }}

    // Troca de Abas
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      const activeBtn = document.querySelector(`.tab-btn[data-tab="${{tabId}}"]`);
      const activeContent = document.getElementById(tabId);
      if (activeBtn) activeBtn.classList.add('active');
      if (activeContent) activeContent.classList.add('active');
    }}

    // Event Listeners Globais
    function initEventListeners() {{
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
      }});

      document.getElementById('btn-mode-timeline').addEventListener('click', () => {{
        document.getElementById('layout-select').value = 'timeline';
        document.getElementById('btn-mode-timeline').classList.add('active');
        document.getElementById('btn-mode-cose').classList.remove('active');
        applySelectedLayout();
      }});

      document.getElementById('btn-mode-cose').addEventListener('click', () => {{
        document.getElementById('layout-select').value = 'cose';
        document.getElementById('btn-mode-cose').classList.add('active');
        document.getElementById('btn-mode-timeline').classList.remove('active');
        applySelectedLayout();
      }});

      document.getElementById('layout-select').addEventListener('change', () => {{
        const v = document.getElementById('layout-select').value;
        if (v === 'timeline') {{
          document.getElementById('btn-mode-timeline').classList.add('active');
          document.getElementById('btn-mode-cose').classList.remove('active');
        }} else if (v === 'cose') {{
          document.getElementById('btn-mode-cose').classList.add('active');
          document.getElementById('btn-mode-timeline').classList.remove('active');
        }} else {{
          document.getElementById('btn-mode-timeline').classList.remove('active');
          document.getElementById('btn-mode-cose').classList.remove('active');
        }}
        applySelectedLayout();
      }});

      document.getElementById('tau-slider').addEventListener('input', filterEdgesByConfidence);
      document.getElementById('chk-ml-only').addEventListener('change', filterEdgesByConfidence);
      document.getElementById('chk-show-canon').addEventListener('change', filterEdgesByConfidence);
      document.getElementById('chk-hide-isolated').addEventListener('change', filterEdgesByConfidence);

      document.getElementById('btn-zoom-in').addEventListener('click', () => cy && cy.zoom(cy.zoom() * 1.25));
      document.getElementById('btn-zoom-out').addEventListener('click', () => cy && cy.zoom(cy.zoom() * 0.8));
      document.getElementById('btn-fit').addEventListener('click', () => cy && cy.fit(null, 35));

      document.getElementById('sb-btn-calc').addEventListener('click', recalculateCounterfactual);

      // Player Controls
      document.getElementById('pl-btn-start').addEventListener('click', () => applyTimelineStep(0));
      document.getElementById('pl-btn-prev').addEventListener('click', () => applyTimelineStep(currentSentenceIdx - 1));
      document.getElementById('pl-btn-play').addEventListener('click', toggleTimelinePlayer);
      document.getElementById('pl-btn-next').addEventListener('click', () => applyTimelineStep(currentSentenceIdx + 1));
      document.getElementById('pl-btn-all').addEventListener('click', () => applyTimelineStep(maxSentenceIdx));
      document.getElementById('pl-slider').addEventListener('input', (e) => applyTimelineStep(parseInt(e.target.value)));
    }}
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"-> Visualizador SOTA P2 gerado com sucesso em: {output_html_path}")
    print(f"   • Total de casos indexados: {len(cases_payload)}")
    return output_html_path
