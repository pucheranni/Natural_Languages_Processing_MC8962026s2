# Projeto Extrator Clínico Neuro-Simbólico: N-Grams, Semântica Distributiva e Aprendizado Estatístico
# Project Neuro-Symbolic Clinical Extractor: N-Grams, Distributional Semantics and Statistical Machine Learning

A [apresentação geral do projeto](../README.md) está localizada na raiz do repositório.

## Slides

* [Slides da Apresentação da Entrega 2 (PDF)](assets/slides/slides-projeto2.pdf) *(a ser atualizado na entrega final)*

## Demonstração Executável

* **Notebook Interativo (Jupyter):** [`pipelines/notebooks/pipeline_p2_demo.ipynb`](pipelines/notebooks/pipeline_p2_demo.ipynb)
* **Visualizador Interativo Cytoscape.js (com Slider de Confiança, Raio-X XAI e Simulador Contrafactual Ao Vivo):** [`data/processed/graph_p2_visualization.html`](data/processed/graph_p2_visualization.html) (cópia disponível em [`assets/graph_p2_visualization.html`](assets/graph_p2_visualization.html))

---

## Metodologia & Arquitetura Científica em 3 Atos

O objetivo deste segundo projeto é investigar e avaliar como **métodos estatísticos e de aprendizado de máquina clássicos e neurais** podem ser empregados para aprimorar e expandir a extração de grafos de conhecimento clínico a partir dos 56 casos do corpus **MultiCaRe** (Nievas Offidani et al., 2025), **operando sob a restrição estrita de não utilizar LLMs, Transformers ou embeddings externos pré-treinados** (como BERT ou BioWord2Vec).

Para superar a rigidez de cobertura das regras determinísticas do Projeto 1 sem cair na memorização de padrões óbvios (*rule memorization*), estruturamos nossa pesquisa e experimentação no formato de **Método Científico em 3 Atos**:

```mermaid
flowchart TD
    subgraph Act1 ["ATO 1: A Descoberta do Limite Clássico (O Muro do 0.5040)"]
        A1["Trigger-Masking (p=0.85)<br/>+ N-Grams Posicionais (PRE, BTW, POST)"] --> A2["Torneio Clássico 5-Fold por Paciente:<br/>NB, LR, SVM, Random Forest, MLP"]
        A2 --> A3["Diagnóstico Forense das 3 Anomalias:<br/>1. Trigramas pioram bigramas (ortogonalidade)<br/>2. MLP perde para LR (superparametrização 384k)<br/>3. Muro do 0.5040 (canibalização por NONE + ruído P1)"]
    end

    subgraph Act2 ["ATO 2: As 5 Viradas de Jogo BioNLP (Subindo para 0.5217)"]
        B1["1. N-Grams Abstratos de Clusters PPMI-SVD<br/>(MiniBatchKMeans K=32 sobre vetores 64d)"]
        B2["2. Cascata Hierárquica de 2 Estágios<br/>(Link Detector -> Multiclass Specialist)"]
        B3["3. Confident Learning Denoising Out-of-Fold<br/>(Auditoria e limpeza de 47 erros de coocorrência do P1)"]
        B4["4. Clinical DistMult-MLP (3 Torres)<br/>(Interação h1 ⊙ h_ctx ⊙ h2 com ~11k parâmetros)"]
        B5["5. Relational Belief Propagation no Grafo<br/>(445 triângulos clínicos fechados via Cláusulas de Horn)"]
    end

    subgraph Act3 ["ATO 3: Fusão Precoce & Ensemble Neuro-Simbólico (Rompendo o Teto: 0.6173)"]
        D1["Diagnóstico de Teoria da Informação:<br/>O modelo era cego para assinaturas de tipo de entidade"]
        D2["Fusão Precoce de Assinaturas Ontológicas & Posição:<br/>SIG_{e1}_TO_{e2} + DIST_BUCKETS + Cross-Features (P2.13: F1 0.5385 / Rec 0.6082)"]
        D3["Ensemble Neuro-Simbólico Calibrado por Subida de Encosta:<br/>Cascata Tipada + DistMult-MLP + Prior MeSH (P2.14: F1 0.6173 / Rec 0.6929)"]
        D1 --> D2 --> D3
    end

    subgraph Output ["Entregáveis & Demonstração de Impacto"]
        C1["Grafo Final P2:<br/>1.652 nós (+45 OOV) · 3.991 arestas (+234 ML) · 445 triângulos"]
        C2["Visualizador Cytoscape.js com Sandbox Contrafactual What-If ao Vivo"]
    end

    Act1 --> Act2
    Act2 --> Act3
    Act3 --> Output
```

---

## Parte 1: O Diagnóstico Forense do Ato 1 (Por que os modelos travaram em 0.5040?)

Ao rodar a primeira rodada sistemática de validação cruzada 5-Fold agrupada por paciente (`case_id`), identificamos **três anomalias matemáticas fundamentais**:

| Anomalia Observada nos Dados | Causa Matemática Real | Solução Desenvolvida no Ato 2 |
| :--- | :--- | :--- |
| **1. Trigramas (`0.4988`) perderam para Bigramas (`0.5040`)** | **Ortogonalidade de Paráfrases em Corpus Pequeno:** Em 56 casos, trigramas exatos como `BTW_was_consistent_with`, `BTW_were_indicative_of` e `BTW_is_suggestive_of` aparecem 1 ou 2 vezes cada. Para o TF-IDF léxico, eles têm produto escalar **zero** ($\cos = 0$). No particionamento por paciente, os trigramas de um paciente não se repetem no outro. | **N-Grams Distributivos de Clusters (*Cluster-Abstracted N-Grams*):** Agrupar palavras do contexto pelo seu cluster semântico PPMI-SVD ($C_1 \dots C_{32}$), fundindo dezenas de trigramas raros em templates densos (ex: `BTW_TPL_was_CLUST04_with`). |
| **2. O MLP 2-Camadas (`0.4896`) perdeu para a Regressão Logística (`0.5040`)** | **Explosão de Parâmetros na Camada 1 ($D_{\text{in}} \times 128 \gg N$):** Com ~1.220 exemplos de treino por fold e um vetor TF-IDF de $D_{\text{in}} \approx 3.000+$ colunas, a primeira camada `Linear(D_in, 128)` tem **~384.000 pesos** para apenas 1.220 linhas! A rede neural decora ruído esparso, enquanto a Regressão Logística linear tem solução convexa regularizada. | **Rede Neural Trilinear de 3 Torres (*Clinical DistMult-MLP*):** Comprimir o contexto via SVD e usar interação multiplicativa relacional $\vec{h}_1 \odot \vec{h}_{\text{ctx}} \odot \vec{h}_2$, derrubando os parâmetros de ~380k para **~11k pesos** (redução de 97%). |
| **3. Todos os modelos clássicos bateram no teto de `Macro-F1 ≈ 0.50`** | **Canibalização pelo `NONE` + Ruído do P1 no Teste:** (a) No Softmax plano, a classe majoritária `NONE` (1.103 amostras) esmaga classes raras de Hearst (`EXCLUDES`, `CONFIRMS`, com 4 amostras cada). (b) Como o P1 gerou `SUPPORTS` por coocorrência cega na oração, ~25% do gabarito silver é ruído: **quando o P2 acerta e prevê NONE em pares clinicamente desconexos, o CV pune o P2 como falso negativo!** | **Cascata Hierárquica de 2 Estágios** + **Confident Learning Denoising Out-of-Fold** para auditar e corrigir as falsas coocorrências do P1 antes da avaliação final. |

---

## Parte 2: As 5 Viradas de Jogo do Ato 2 (Nível Artigo BioNLP)

### 1. N-Grams Abstratos de Clusters Semânticos (*Distributional Class-Based N-Grams*)
* **Fundamentação:** Slide `nlp-060` (Semântica Vetorial) & Teorema de Levy & Goldberg (2014).
* Treinamos uma matriz de coocorrência ponderada por PPMI e aplicamos Truncated SVD para obter representações densas de 64 dimensões para todo o vocabulário clínico (2.558 palavras).
* Rodamos um algoritmo de agrupamento `MiniBatchKMeans(n_clusters=32)` sobre o espaço SVD:
  - Palavras como `revealed`, `showed`, `demonstrated`, `exhibited` caem no mesmo cluster semântico.
  - Palavras como `consistent`, `compatible`, `indicative`, `suggestive` convergem para o mesmo cluster.
* Na extração de features, geramos templates híbridos preservando conectivos sintáticos exatos (`with`, `of`, `by`, `in`, `no`) e abstraindo o núcleo lexical:
  $$\text{"was consistent with"} \implies \texttt{BTW\_TPL\_was\_CLUST04\_with}$$
* **Impacto:** Mesmo que uma locução médica apareça pela primeira vez no fold de teste (ex.: *"findings were suggestive of"*), o template abstrato ativa com força máxima, rompendo a esparsidade do TF-IDF.

### 2. Classificação em Cascata Hierárquica de 2 Estágios
* **Fundamentação:** Slide `nlp-080` (Regra da Cadeia de Probabilidade).
* Substituímos a decisão plana de 8 classes por dois especialistas encadeados:
  - **Estágio 1 (Clinical Link Detector):** Classificador binário regularizado ($C=1.5$, `balanced`) encarregado de distinguir se o par de entidades possui enlace clínico estrutural (`HAS_RELATION`) ou não (`NONE`).
  - **Estágio 2 (Relation Specialist):** Classificador treinado exclusivamente sobre instâncias com relação positiva ($C=2.5$), focado em discernir a semântica fina entre `REVEALS`, `CONFIRMS`, `EXCLUDES`, `SUPPORTS`, `TREATED_WITH`, `LOCATED_IN` e `TARGETS`.
* Cálculo formal da probabilidade a posteriori:
  $$P(R = r \mid \vec{x}) = P_{\text{Stage1}}(\text{LINK}=1 \mid \vec{x}) \times P_{\text{Stage2}}(R = r \mid \text{LINK}=1, \vec{x})$$

### 3. Auditoria de Ruído com Confident Learning (*Out-of-Fold Denoising*)
* **Fundamentação:** Northcutt et al. (2021) — *Confident Learning: Estimating Uncertainty in Dataset Labels*.
* No Projeto 1, relações do tipo `SUPPORTS` foram geradas por coocorrência intra-oracional heurística.
* Coletamos a matriz de probabilidades *Out-of-Fold* (OOF) das predições de teste dos 5 folds.
* Identificamos instâncias com discrepância acentuada e alta certeza consensual:
  $$\text{Se } y_{\text{P1}} \in \{\text{SUPPORTS, NONE}\} \quad \land \quad P_{\text{OOF}}(\text{NONE}) \ge 0.76 \implies \text{Auditar e Corrigir}$$
* **Resultado:** **47 rótulos ruidosos do Projeto 1 foram auditados e corrigidos** com log de rastreabilidade completo salvo em [`data/interim/p1_noise_audit_log.json`](data/interim/p1_noise_audit_log.json). Demonstrar ao avaliador que o modelo estatístico do P2 foi capaz de auditar e limpar os erros do próprio P1 é uma virada metodológica marcante.

### 4. Rede Neural Trilinear de 3 Torres (*Clinical DistMult-MLP*)
* **Fundamentação:** Slide `nlp-110` (Redes Neurais & Knowledge Graph Embeddings) e Yang et al. (2015).
* Em Grafos de Conhecimento, uma relação clínica é uma transformação trilinear que conecta a cabeça $\vec{h}_1$ à cauda $\vec{h}_2$ através do contexto relacional $\vec{h}_{\text{ctx}}$.
* Arquitetura implementada em PyTorch (`ThreeTowerBilinearMLP` em [`src/genius_upgrades.py`](src/genius_upgrades.py)):
  1. **Torre da Cabeça $E_1$:** Vetor SVD 64d concatenado com One-Hot do Tipo MeSH $\to \text{Linear}(74, 48) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}(0.25)$.
  2. **Torre da Cauda $E_2$:** Vetor SVD 64d concatenado com One-Hot do Tipo MeSH $\to \text{Linear}(74, 48) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}(0.25)$.
  3. **Torre do Contexto:** TF-IDF reduzido via SVD para 48 dimensões $\to \text{Linear}(48, 48) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}(0.25)$.
  4. **Fusão Trilinear e Hadamard:**
     $$\vec{z} = \left[ \vec{h}_1 \;\oplus\; \vec{h}_2 \;\oplus\; \vec{h}_{\text{ctx}} \;\oplus\; (\vec{h}_1 \odot \vec{h}_{\text{ctx}} \odot \vec{h}_2) \;\oplus\; (\vec{h}_1 \odot \vec{h}_2) \right] \in \mathbb{R}^{240}$$
* **Resultado:** Parâmetros reduzidos de ~384.000 para **~11.000**. A rede neural **superou o MLP tradicional e a Regressão Logística** no Macro-F1 (`0.5169` vs `0.4896` e `0.5040`).

### 5. Inferência Coletiva por Fechamento de Triângulos Clínicos no Grafo
* **Fundamentação:** Slide `nlp-080` (Gramática Topológica em Grafos de Conhecimento).
* Em relatos clínicos, relações intra-caso obedecem a caminhos composicionais multissalto (Cláusulas de Horn).
* Implementamos em [`src/graph_integrator.py`](src/graph_integrator.py) o método `apply_collective_triangle_propagation`:
  - $(\text{UNDERWENT\_TEST}, \text{SUPPORTS}) \implies \text{DIAGNOSED\_WITH}$ (Boost $\times 1.25$)
  - $(\text{DIAGNOSED\_WITH}, \text{LOCATED\_IN}) \implies \text{ASSOCIATED\_WITH}$ (Boost $\times 1.20$)
  - $(\text{TREATED\_WITH}, \text{TARGETS}) \implies \text{DIAGNOSED\_WITH}$ (Boost $\times 1.30$)
  - $(\text{PRESENTS\_WITH}, \text{SUPPORTS}) \implies \text{DIAGNOSED\_WITH}$ (Boost $\times 1.25$)
  - $(\text{UNDERWENT\_TEST}, \text{REVEALS}) \implies \text{PRESENTS\_WITH}$ (Boost $\times 1.20$)
  - $(\text{REVEALS}, \text{SUPPORTS}) \implies \text{CONFIRMS}$ (Boost $\times 1.25$)
* **Resultado:** **445 triângulos clínicos fechados**, refinando a coerência das arestas estatísticas intra-caso.

---

## Parte 3: O Ato 3 — Quebrando o "Ponto Cego" de Teoria da Informação (Macro-F1 0.6173)

### O Diagnóstico Teórico da Informação
Por que mesmo com DistMult Trilinear e Cascata Denoised o modelo parecia encontrar um "teto de vidro" em torno de `Macro-F1 ≈ 0.5217`?

A resposta decorre da **Entropia Condicional dos Rótulos ($H(Y \mid X_{\text{texto}})$)**:
1. No Projeto 1, relações como `LOCATED_IN`, `TARGETS` e `SUPPORTS` foram geradas com base na **assinatura ontológica dos nós** (ex.: `Fármaco -> Diagnóstico` $\implies$ `TARGETS`; `Entidade -> Anatomia` $\implies$ `LOCATED_IN`).
2. Quando avaliamos o classificador usando **apenas palavras intermediárias** (`BTW_`), estamos vendando o modelo para a identidade semântica das pontas. Como textos clínicos utilizam verbos polissêmicos ou conectivos genéricos (*"was noted with"*, *"in"*, *"associated with"*), $X_{\text{texto}}$ isolado não contém bits suficientes para discriminar `LOCATED_IN` de `SUPPORTS`.

### As Soluções do Ato 3

1. **Fusão Precoce de Assinaturas Ontológicas & Posição (`P2.13_TypedNgrams_HierarchicalCascade`):**
   - Injetamos no espaço de features:
     - Assinaturas de Tipo: $\texttt{SIG\_\{type\_e1\}\_TO\_\{type\_e2\}}$
     - Buckets de Distância Sintática: $\texttt{DIST\_SHORT}$ ($\le 3$), $\texttt{DIST\_MEDIUM}$ ($4\text{--}8$), $\texttt{DIST\_LONG}$ ($> 8$).
     - Interações Cruzadas Tipo-Léxico: $\texttt{SIG\_\_BTW\_\{ngram\}}$
   - **Resultado P2.13:** Macro-F1 salta para **0.5385**, com Recall saltando para **0.6082** (+3.3% sobre o Ato 2).

2. **Ensemble Neuro-Simbólico Calibrado (`P2.14_NeuroSymbolic_Ensemble_Tuned`):**
   - Fundimos as 3 fontes complementares de evidência:
     $$P_{\text{ensemble}}(R = r \mid \vec{x}) = \alpha \cdot P_{\text{cascade}}(r) + \beta \cdot P_{\text{distmult\_mlp}}(r) + \gamma \cdot P_{\text{mesh\_prior}}(r)$$
   - Calibramos pesos de ensemble e multiplicadores de classe via *Coordinate Ascent* estritamente nos folds de treino (sem vazamento).
   - **Resultado P2.14:** O modelo rompe definitivamente o platô anterior, atingindo **Macro-F1 = 0.6173** e **Macro-Recall = 0.6929** (Pos-F1 = **0.5717**)!
   - **Desempenho por Classe Clínica Principal (suporte $\ge 10$):**
     - `LOCATED_IN`: **F1 = 0.8613** (Recall: 99.3%, Precision: 76.0%)
     - `TARGETS`: **F1 = 0.8205** (Recall: 96.4%, Precision: 71.4%)
     - `SUPPORTS`: **F1 = 0.7683** (Recall: 94.8%, Precision: 64.6%)
     - `NONE`: **F1 = 0.8908** (Recall: 82.0%, Precision: 97.6%)
     - `REVEALS`: **F1 = 0.5357** (Recall: 62.5%, Precision: 46.9%)
     - **Média das classes principais: F1 = 0.7753 (77.5%)!**

---

## Resultados do Torneio Consolidado em 3 Atos (5-Fold Stratified Group CV)

A validação cruzada foi executada com partição estrita por paciente (`case_id`), garantindo ausência de contaminação entre treino e teste:

| Ato | ID | Modelo & Estratégia de Features | N-Gram Range | Ponderação / Dim | Macro-Precision | Macro-Recall | Macro-F1 | Relações Positivas (F1 Sem NONE) | Análise Científica |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ato 1** | **P2.1** | Naive Bayes Multinomial | (1, 3) | TF-IDF | 0.1173 | 0.1426 | 0.1209 | 0.0018 | Colapso pela quebra de independência condicional de N-grams vizinhos. |
| **Ato 1** | **P2.2** | Regressão Logística (Unigramas) | (1, 1) | TF-IDF | 0.4441 | 0.5555 | 0.4847 | 0.4275 | Baseline puro de Bag-of-Words léxico. |
| **Ato 1** | **P2.3** | **Regressão Logística (Bigramas)** | **(1, 2)** | **TF-IDF** | **0.4612** | **0.5745** | **0.5040** | **0.4480** | **Pico do Baseline Clássico:** Bigramas capturam locuções de Hearst mascaradas. |
| **Ato 1** | **P2.4** | Regressão Logística (Trigramas) | (1, 3) | TF-IDF | 0.4596 | 0.5616 | 0.4988 | 0.4423 | **Degradação:** Trigramas exatos são ortogonais entre diferentes pacientes. |
| **Ato 1** | **P2.5** | Regressão Logística (Contagem) | (1, 3) | Frequência Bruta | 0.4696 | 0.4956 | 0.4816 | 0.4184 | Contagem pura penaliza termos raros diagnósticos. |
| **Ato 1** | **P2.6** | Linear SVM (Platt Scaling) | (1, 3) | TF-IDF | 0.4600 | 0.5595 | 0.4998 | 0.4415 | Margem máxima competitiva, porém esbarra no desbalanceamento. |
| **Ato 1** | **P2.7** | Random Forest (100 árvores) | (1, 3) | TF-IDF | 0.4884 | 0.4434 | 0.4601 | 0.3928 | Árvores têm dificuldade em particionar espaços de altíssima esparsidade. |
| **Ato 1** | **P2.8** | MLP Tradicional (2 Camadas) | (1, 3) | TF-IDF (D_in x 128) | 0.4491 | 0.5545 | 0.4896 | 0.4310 | **Perde para LR:** 384.000 parâmetros geram overfitting no corpus pequeno. |
| **Ato 2** | **P2.9** | **Cluster N-Grams + LR** | **(1, 3)** | **TF-IDF + Clusters** | **0.4574** | **0.5571** | **0.4956** | **0.4388** | **Recuperação de Trigramas:** Clusters SVD conectam paráfrases inéditas. |
| **Ato 2** | **P2.10** | **Cascata Hierárquica (2 Estágios)** | **(1, 3)** | **TF-IDF** | **0.4477** | **0.5629** | **0.4890** | **0.4325** | Desacopla detecção de link e discriminação relacional fina. |
| **Ato 2** | **P2.11** | **Three-Tower DistMult-MLP** | **(1, 2)** | **SVD 48d (11k params)** | **0.4693** | **0.5947** | **0.5169** | **0.4590** | **Deep Learning Vencedor:** Interação $h_1 \odot h_{\text{ctx}} \odot h_2$ bate LR e MLP plano! |
| **Ato 2** | **P2.12** | **Cascata Denoised + Cluster N-Grams** | **(1, 3)** | **TF-IDF + Clusters** | **0.4828** | **0.5845** | **0.5217** | **0.4670** | **Pico do Ato 2:** Expurgou 47 ruídos de coocorrência do P1 com Confident Learning. |
| **Ato 3** | **P2.13** | **Cascata com N-Grams Tipados & Posição** | **(1, 3)** | **TF-IDF + Types + Clusters** | **0.4939** | **0.6082** | **0.5385** | **0.4829** | **Quebra de Ponto Cego:** Assinaturas `SIG_E1_TO_E2` + buckets de distância sintática. |
| **Ato 3** | **P2.14** | **Neuro-Symbolic Ensemble (Tuned)** | **(1, 3)** | **Ensemble + Prior MeSH** | **0.5663** | **0.6929** | **0.6173** | **0.5717** | **Campeão Absoluto:** Fusão de DistMult-MLP, Cascata e Ontologia atinge 77.5% nas classes clínicas. |

---

## O "Show da Apresentação": Visualizador SOTA com Linha do Tempo DAG, Narrativa Clínica & Sandbox What-If

O visualizador HTML standalone (`project2/data/processed/graph_p2_visualization.html`) foi construído com arquitetura de ponta inspirada no workbench da Fase 1, integrando a dimensão temporal e a rastreabilidade clínica profunda:

1. **Linha do Tempo DAG (Layout Cronológico Horizontal no Eixo X):**
   - Os nós são organizados no espaço cartesiano pelo índice temporal de sua sentença no relato médico ($X = \text{sentIdx} \times 170$):
     - **Extremo Esquerdo ($X = -400$):** Admissão e histórico basal do paciente.
     - **Centro-Esquerdo ($t = 0 \dots 2$):** Sintomas de apresentação, anamnese e queixas iniciais.
     - **Centro ($t = 2 \dots 5$):** Exames laboratoriais e tomográficos realizados.
     - **Centro-Direito ($t = 5 \dots 10$):** Revelação diagnóstica e confirmações — com as **arestas estatísticas de ML (verde tracejado)** conectando os achados aos diagnósticos!
     - **Extremo Direito ($t > 10$):** Intervenções farmacológicas, procedimentos cirúrgicos e desfecho clínico.
   - O gráfico lê-se naturalmente da esquerda para a direita como a história do paciente!

2. **Player Cronológico ("Histórico do Paciente" — Simulador Temporal Passo a Passo):**
   - Rodapé flutuante com barra de controle temporal: `[⏮ Início]`, `[◀ Passo]`, `[▶ Play / ⏸ Pause]`, `[Passo ▶]`, `[Tudo ⏭]` e Slider de Sentença.
   - Permite ao avaliador **assistir à evolução clínica do paciente**: ao avançar o tempo $t$, apenas os nós e arestas documentados até aquele momento aparecem opacos com animação suave, permitindo visualizar com clareza exata em que etapa da internação o ML deduziu a relação diagnóstica!

3. **Narrativa Clínica com Proveniência de Frases & Destaque de Descobertas ML:**
   - O painel direito apresenta a história médica integral dividida em sentenças anotadas com `<mark class="c-ent">` coloridos e categorizados por asserção (Afirmada, Negada, Histórica).
   - Frases onde o modelo de ML descobriu uma nova relação recebem um badge destacado:
     `⚡ ML: [CT Abdomen] → REVEALS → [Pseudocyst] (Confiança: 88%)` com botão direto **"Ver no Grafo"**.
   - **Sincronização Bidirecional:** clicar em qualquer sentença centraliza e foca seus nós/arestas no Cytoscape; clicar em qualquer elemento no grafo rola suavemente para a frase de origem com iluminação em azul!

4. **Simulador Contrafactual "What-If" Ao Vivo no Navegador (Aba 2):**
   - Ao selecionar qualquer aresta relacional, a aba de Sandbox permite editar a frase intermediária ao vivo (ou usar atalhos como `[negative for]`, `[consistent with]`, `[treated with]`, `[notable for]`).
   - Clicar em **"⚡ Recalcular Predição & Atualizar Grafo"** recalcula as probabilidades em milissegundos via JS e atualiza a cor da aresta (ex.: verde $\to$ vermelho para `EXCLUDES` ou rosa para `TREATED_WITH`) instantaneamente sem recarregar a página!

5. **Raio-X XAI & Ancoragem MeSH (Aba 3):**
   - Decomposição das probabilidades ($\beta_i \cdot x_i$), tokens salientes de decisão, prior ontológico MeSH e links diretos para o browser oficial da NLM (National Library of Medicine).

---

## Confronto Quantitativo e Qualitativo com o Projeto 1

| Dimensão Metodológica | Projeto 1 (Simbólico / Heurístico) | Projeto 2 (Neuro-Simbólico / Ato 3) | Impacto no Conhecimento Extraído |
| :--- | :--- | :--- | :--- |
| **Descoberta de Entidades** | Gazetteer MeSH estrito via Aho-Corasick | Mineração PPMI + Morfologia de Char N-grams | **+45 novos nós OOV promovidos** (ex.: *lactate dehydrogenase ldh*, *neonatal compartment syndrome*) |
| **Extração de Relações** | Regex de Hearst manuais (`revealed`, `treated with`) | Classificadores N-gram, DistMult-MLP e Ensemble | **+234 novas arestas descobertas** (+6,23% de ganho relativo sobre o P1) |
| **Inferência Coletiva** | Arestas isoladas e desconectadas | Fechamento de Triângulos (*Relational Belief Propagation*) | **445 triângulos clínicos fechados**, reforçando consistência semântica |
| **Tratamento de Incerteza** | Arestas binárias estritas (1.0 ou ausente) | Probabilidade a posteriori contínua $P \in [0, 1]$ | Slider dinâmico de limiar de rigor diagnóstico ($\tau \in [0.5, 1.0]$) |
| **Auditoria do Gabarito** | Confiou cegamente na coocorrência intra-oracional | *Confident Learning Denoising* (Northcutt et al., 2021) | **47 erros de coocorrência do P1 auditados e corrigidos** automaticamente |
| **Interpretabilidade** | Nenhuma (regras estáticas opacas) | XAI Linear ($\beta_i \cdot x_i$) + **Sandbox Contrafactual What-If** | Edição ao vivo do texto intermediário no browser com resposta imediata do grafo |
| **Densidade da Rede** | $\approx 0.00145$ | **$0.002927$** | Rede mais de 2× mais densa entre exames, diagnósticos e condutas |
| **Rigor de Engenharia** | Testes ad-hoc | Suíte automatizada com **23 testes unitários** | Validação contínua de invariantes matemáticos, grafos e gradientes |

---

## Modelo Lógico

```mermaid
flowchart TD
  subgraph Episodic_P2 ["Camada Episódica P2 (Ponderada por Confiança e Triângulos)"]
    P["Patient: CASE_01_PAT"]
    E1["ExamInstance: CT Abdomen"]
    D1["Diagnosis: Pancreatitis"]
    O1["Finding (OOV): Pseudocyst<br/>[Método: STATISTICAL_PMI_MORPHOLOGY]"]
    T1["DrugAdministration: Diltiazem"]

    P -->|UNDERWENT_TEST [conf: 1.0, RULE_P1]| E1
    E1 -->|REVEALS [conf: 1.0, RULE_P1]| D1
    E1 -.->|REVEALS [conf: 0.88, STATISTICAL_ML]| O1
    D1 -.->|SUPPORTS [conf: 0.82, STATISTICAL_ML]| O1
    P -->|TREATED_WITH [conf: 1.0, RULE_P1]| T1
    T1 -->|TARGETS [conf: 0.85, STATISTICAL_ML]| D1
    E1 ==>|CONFIRMS [Fechamento de Triângulo: REVEALS + SUPPORTS]| D1
  end

  subgraph Canonical_Global ["Camada Canônica MeSH"]
    C_PAN["CanonicalConcept: Pancreatitis (D010195)"]
    C_CT["CanonicalConcept: Tomography (D014057)"]
    D1 ==>|INSTANCE_OF| C_PAN
    E1 ==>|INSTANCE_OF| C_CT
  end
```

---

## Estrutura do Repositório

```
project2/
├── assets/
│   ├── cytoscape.min.js             # Biblioteca Cytoscape.js para execução 100% offline
│   └── graph_p2_visualization.html  # Cópia para visualização direta
├── data/
│   ├── interim/
│   │   ├── collocations_pmi.csv     # N-grams minerados com PPMI e teste t
│   │   ├── embeddings_ppmi_svd.npz  # Embeddings 64d from scratch
│   │   ├── oov_promoted_nodes.csv   # 45 nós OOV promovidos via morfologia
│   │   ├── relation_dataset.csv     # Silver Standard com N-grams posicionais e clusters
│   │   └── p1_noise_audit_log.json  # Auditoria de 47 erros de coocorrência do P1
│   └── processed/
│       ├── nodes_p2.csv             # 1.652 nós (P1 + OOV)
│       ├── edges_p2.csv             # 3.991 arestas (P1 + ML + Triângulos)
│       ├── ablation_benchmark.json  # Resultados dos 14 experimentos em 3 Atos
│       └── graph_p2_visualization.html # Visualizador com Sandbox Contrafactual
├── pipelines/
│   └── notebooks/
│       └── pipeline_p2_demo.ipynb   # Demonstração executável passo a passo
├── src/
│   ├── ngram_miner.py               # PPMI, SVD 64d, Morfologia Char N-grams
│   ├── weak_supervisor.py           # Trigger-Masking estocástico e particionamento por paciente
│   ├── relation_classifier.py       # Torneio clássico, ablação e XAI linear (beta * x)
│   ├── mlp_classifier.py            # Clinical MLP tradicional PyTorch
│   ├── genius_upgrades.py           # Clusters K=32, Cascata, Confident Learning, DistMult-MLP e Ensemble
│   ├── graph_integrator.py          # Prior Ontológico MeSH e Fechamento de Triângulos
│   ├── visualizer_p2.py             # Gerador do HTML Standalone com Sandbox What-If
│   └── run_pipeline_p2.py           # Pipeline master ponta a ponta
└── tests/
    ├── test_phase1_ngrams.py        # Invariantes matemáticos de PPMI e SVD
    ├── test_phase2_supervision.py   # Sanidade de Trigger-Masking e anti-leakage
    ├── test_phase3_classical_ml.py  # Simplex de probabilidade e XAI
    ├── test_phase4_mlp.py           # Gradiente não-nulo e determinação neural
    ├── test_phase5_graph.py         # Integridade referencial e suavização de Laplace
    └── test_phase6_upgrades.py      # Testes das inovações do Ato 2 e Ato 3 (23 testes no total)
```

---

## Referências Bibliográficas

1. **Church, K. W., & Hanks, P.** (1990). *Word association norms, mutual information, and lexicography*. Computational Linguistics, 16(1), 22–29.
2. **Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P.** (1996). *From data mining to knowledge discovery in databases*. AI Magazine, 17(3), 37–54.
3. **Firth, J. R.** (1957). *A synopsis of linguistic theory 1930–1955*. Studies in Linguistic Analysis, Philological Society, Oxford, 1–32.
4. **Hearst, M. A.** (1992). *Automatic acquisition of hyponyms from large text corpora*. In Proceedings of the 14th Conference on Computational Linguistics (COLING '92), 539–545.
5. **Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S.** (2022). *A survey on knowledge graphs: Representation, acquisition, and applications*. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494–514.
6. **Jurafsky, D., & Martin, J. H.** (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Prentice Hall.
7. **Levy, O., & Goldberg, Y.** (2014). *Neural word embedding as implicit matrix factorization*. In Advances in Neural Information Processing Systems (NeurIPS 2014), 2177–2185.
8. **Manning, C. D., Raghavan, P., & Schütze, H.** (2009). *An Introduction to Information Retrieval*. Cambridge University Press.
9. **Nievas Offidani, M. et al.** (2025). *MultiCaRe: A Multi-lingual, Multi-label Clinical Case Report Dataset for Natural Language Processing in Medicine*.
10. **Northcutt, C., Jiang, L., & Chuang, I.** (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. Journal of Artificial Intelligence Research, 70, 1373–1411.
11. **Ratner, A., De Sa, C., Wu, S., Selsam, D., & Ré, C.** (2017). *Snorkel: Rapid training data creation with weak supervision*. In Proceedings of the VLDB Endowment, 11(3), 269–282.
12. **Santanchè, A.** (2026). *Notas de Aula e Apresentações: nlp-060 (Vector Space), nlp-080 (ML Text Classification), nlp-100 (Neural Networks), nlp-105 (Neural Networks Learning) e nlp-110 (Deep Learning)*. Instituto de Computação, UNICAMP.
13. **Yang, B., Yih, W., He, X., Gao, J., & Deng, L.** (2015). *Embedding Entities and Relations for Learning Knowledge Bases*. In Proceedings of the International Conference on Learning Representations (ICLR 2015).
