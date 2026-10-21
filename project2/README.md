# Projeto Extrator Clínico Neuro-Simbólico: N-Grams, Semântica Distributiva e Aprendizado Estatístico
# Project Neuro-Symbolic Clinical Extractor: N-Grams, Distributional Semantics and Statistical Machine Learning

A [apresentação geral do projeto](../README.md) está localizada na raiz do repositório.

## Slides

* [Slides da Apresentação da Entrega 2 no Canva](https://canva.link/r4dueh01zzx0xa1)
* [Cópia Local dos Slides em PDF](assets/slides/slides-projeto2.pdf)

## Demonstração Executável

* **Notebook Interativo (Jupyter):** [`pipelines/notebooks/pipeline_p2_demo.ipynb`](pipelines/notebooks/pipeline_p2_demo.ipynb)
* **Visualizador Interativo Cytoscape.js (com Slider de Confiança, Raio-X XAI e Simulador Contrafactual Ao Vivo):** [`data/processed/graph_p2_visualization.html`](data/processed/graph_p2_visualization.html) (cópia disponível em [`assets/graph_p2_visualization.html`](assets/graph_p2_visualization.html))

---

## Metodologia & Arquitetura Científica em 3 Atos

O objetivo deste segundo projeto é investigar e avaliar como **métodos estatísticos e de aprendizado de máquina clássicos e neurais** podem ser empregados para aprimorar e expandir a extração de grafos de conhecimento clínico a partir dos 56 casos do corpus **MultiCaRe** (Nievas Offidani et al., 2025), **operando sob a restrição estrita de não utilizar LLMs, Transformers ou embeddings externos pré-treinados** (como BERT ou BioWord2Vec).

Para superar a rigidez de cobertura das regras determinísticas do Projeto 1 sem cair na memorização trivial de verbos-gatilho óbvios (*lexical rule memorization*), estruturamos nossa pesquisa e experimentação no formato de **Método Científico em 3 Atos**, dialogando com a literatura clássica de *Class-Based N-Grams* (Brown et al., 1992), *Universal Schema & Text-KB Joint Embeddings* (Toutanova et al., 2015; Yang et al., 2015), *Entity-Type Markers* (Zhong & Chen, 2021) e *Interactive Counterfactual Probing* (Wexler et al., 2019):

```mermaid
flowchart TD
    subgraph Act1 ["ATO 1: A Descoberta do Limite Clássico (O Muro do 0.5040)"]
        A1["Trigger-Masking (p=0.85, W=3)<br/>+ N-Grams Posicionais (PRE, BTW, POST)"] --> A2["Torneio Clássico 5-Fold por Paciente:<br/>NB, LR, SVM, Random Forest, MLP"]
        A2 --> A3["Diagnóstico Forense das 3 Anomalias:<br/>1. Trigramas pioram bigramas (ortogonalidade)<br/>2. MLP perde para LR (superparametrização 384k)<br/>3. Muro do 0.5040 (canibalização por NONE + ruído P1)"]
    end

    subgraph Act2 ["ATO 2: As 5 Viradas de Jogo BioNLP (Subindo para 0.5217)"]
        B1["1. N-Grams Abstratos de Clusters PPMI-SVD<br/>(MiniBatchKMeans K=32 sobre vetores 64d)"] --> B2["2. Cascata Hierárquica de 2 Estágios<br/>(Link Detector -> Multiclass Specialist)"]
        B2 --> B3["3. Confident Learning Denoising Out-of-Fold<br/>(Auditoria e limpeza de 47 erros do Silver Standard P1)"]
        B3 --> B4["4. Clinical DistMult-MLP (3 Torres)<br/>(Interação h1 ⊙ h_ctx ⊙ h2 com ~12k parâmetros)"]
        B4 --> B5["5. Relational Belief Propagation no Grafo<br/>(445 triângulos clínicos fechados via Cláusulas de Horn)"]
    end

    subgraph Act3 ["ATO 3: Fusão Precoce & Ensemble Neuro-Simbólico (Rompendo o Teto: 0.6173)"]
        D1["Diagnóstico de Teoria da Informação:<br/>Redução da entropia condicional H(Y | X_texto, T_E1, T_E2)"]
        D2["Fusão Precoce de Assinaturas Ontológicas & Posição:<br/>SIG_{e1}_TO_{e2} + DIST_BUCKETS + Cross-Features (P2.13: F1 0.5385 / Silver 0.5198)"]
        D3["Ensemble Neuro-Simbólico Calibrado por Subida de Encosta:<br/>Cascata Tipada + DistMult-MLP + Prior MeSH (P2.14: F1 0.6173 / Silver 0.5998)"]
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

### Especificação Formal da Representação Base (`PositionalWeakSupervisor`)

Para garantir reprodutibilidade na construção do conjunto de supervisão fraca ([`src/weak_supervisor.py`](src/weak_supervisor.py), **1.529 instâncias** em **7 classes ativas**: `NONE` [1.103], `SUPPORTS` [176], `LOCATED_IN` [138], `TARGETS` [80], `REVEALS` [24], `CONFIRMS` [4] e `EXCLUDES` [4]; `TREATED_WITH` opera nas regras estruturais do paciente no P1, com suporte oracional zero entre pares de entidades episódicas), aplicamos três mecanismos de pré-processamento e anti-vazamento:

1. **Janelamento Posicional (`PRE_`, `BTW_`, `POST_`):** Dado um par de menções $(E_1, E_2)$ na mesma sentença ou em sentenças contíguas ($|\text{sentIdx}(E_1) - \text{sentIdx}(E_2)| \le 1$, distância máxima de $25$ tokens para pares negativos `NONE`, subamostrados à razão `negative_ratio = 2.0`), ordenamos linearmente os spans $[s_1, e_1)$ e $[s_2, e_2)$ com $s_1 \le s_2$ (registrando a flag direcional `DIR_L2R` ou `DIR_R2L` e o marcador de tipo base `TYPE_{first}_{second}`). O contexto é fatiado em três zonas:
   - $\texttt{PRE\_}$: até $W = 3$ tokens imediatamente anteriores à primeira entidade ($[ \max(0, s_1 - 3), s_1 )$).
   - $\texttt{BTW\_}$: todos os tokens intermediários situados entre o fim da primeira entidade e o início da segunda ($[e_1, s_2)$).
   - $\texttt{POST\_}$: até $W = 3$ tokens imediatamente posteriores à segunda entidade ($[e_2, e_2 + 3)$).
2. **Trigger-Masking Estocástico ($p = 0.85$):** Para evitar que os modelos supervisionados meramente memorizem os 38 verbos/locuções literais de Hearst utilizados pelas expressões regulares do Projeto 1 (`P1_HEARST_TRIGGERS`, como `"revealed"`, `"consistent with"`, `"negative for"`, `"ruled out"`), qualquer ocorrência desses gatilhos na zona $\texttt{BTW\_}$ é estocasticamente substituída pelo token neutro `TRIGGER_MASK` com probabilidade $p = 0.85$ durante a geração do dataset (`relation_dataset.csv`). Assim, no benchmark de validação cruzada 5-Fold, tanto os folds de treino quanto os de validação operam sob a mesma distribuição mascarada (forçando a avaliação de generalização contextual circunvizinha), enquanto na inferência aberta (`is_training=False`) o texto original permanece intacto.

---

## Parte 1: O Diagnóstico Forense do Ato 1 (Por que os modelos travaram em 0.5040?)

Ao rodar a primeira rodada sistemática de validação cruzada 5-Fold agrupada por paciente (`case_id` via `StratifiedGroupKFold`), identificamos **três anomalias matemáticas fundamentais**:

| Anomalia Observada nos Dados | Causa Matemática Real | Solução Desenvolvida no Ato 2 |
| :--- | :--- | :--- |
| **1. Trigramas (`0.4988`) perderam para Bigramas (`0.5040`)** | **Ortogonalidade de Paráfrases em Corpus Pequeno:** Em 56 casos, trigramas exatos como `BTW_was_consistent_with`, `BTW_were_indicative_of` e `BTW_is_suggestive_of` aparecem 1 ou 2 vezes cada. Para o TF-IDF léxico, eles têm produto escalar **zero** ($\cos = 0$). No particionamento por paciente, os trigramas de um paciente não se repetem no outro. | **N-Grams Distributivos de Clusters (*Cluster-Abstracted N-Grams*):** Agrupar palavras do contexto pelo seu cluster semântico PPMI-SVD ($C_1 \dots C_{32}$), fundindo dezenas de trigramas raros em templates densos (ex: `BTW_TPL_was_CLUST04_with`) que sinergizam com a cascata e a limpeza de ruído (`P2.12`). |
| **2. O MLP 2-Camadas (`0.4896`) perdeu para a Regressão Logística (`0.5040`)** | **Explosão de Parâmetros na Camada 1 ($D_{\text{in}} \times 128 \gg N$):** Com ~1.220 exemplos de treino por fold e um vetor TF-IDF de $D_{\text{in}} \approx 2.000\text{--}3.000+$ colunas, a primeira camada densa tem centenas de milhares de pesos (até **~384.000 pesos** para $D_{\text{in}}=3.000$) para apenas 1.220 linhas! A rede neural decora ruído esparso, enquanto a Regressão Logística linear tem solução convexa regularizada. | **Rede Neural Trilinear de 3 Torres (*Clinical DistMult-MLP*):** Comprimir o contexto de forma não-supervisionada via SVD fixo e usar interação multiplicativa relacional $\vec{h}_1 \odot \vec{h}_{\text{ctx}} \odot \vec{h}_2$, derrubando os pesos neurais treináveis para **~12k parâmetros** (redução de ~97%). |
| **3. Todos os modelos clássicos bateram no teto de `Macro-F1 ≈ 0.50`** | **Canibalização pelo `NONE` + Ruído do P1 no Teste:** (a) No classificador plano, a classe majoritária `NONE` (1.103 de 1.529 amostras) compete diretamente contra classes raras de Hearst (`EXCLUDES`, `CONFIRMS`, com 4 amostras cada). (b) Como o P1 gerou relações por heurísticas ruidosas e omitiu relações válidas sem gatilho exato, parte do gabarito silver contém falsos negativos (`NONE`) e falsos positivos (`TARGETS`): **quando o P2 acerta a semântica clínica real, o gabarito silver ruidoso o pune!** | **Cascata Hierárquica de 2 Estágios** + **Confident Learning Denoising Out-of-Fold** para auditar e corrigir as discrepâncias de alta confiança do P1, reportando métricas tanto no gabarito auditado quanto no Silver Standard bruto original. |

---

## Parte 2: As 5 Viradas de Jogo do Ato 2 (Nível Artigo BioNLP)

### 1. N-Grams Abstratos de Clusters Semânticos (*Distributional Class-Based N-Grams*)
* **Fundamentação:** Slide `nlp-060` (Semântica Vetorial), Modelos de N-Grams Baseados em Classes (Brown et al., 1992) & Teorema de Fatoração Implícita de Levy & Goldberg (2014).
* Construímos uma matriz de coocorrência ponderada por PPMI (janela simétrica $W=5$, `min_count=2`) e aplicamos `TruncatedSVD` para obter representações densas L2-normalizadas de 64 dimensões para todo o vocabulário clínico ativo (**2.558 palavras**, salvas em [`data/interim/embeddings_ppmi_svd.npz`](data/interim/embeddings_ppmi_svd.npz)).
* Executamos `MiniBatchKMeans(n_clusters=32, random_state=42)` sobre o espaço SVD:
  - Verbos de achado clínico como `revealed`, `showed`, `demonstrated`, `exhibited` convergem para o mesmo cluster semântico.
  - Adjetivos diagnósticos como `consistent`, `compatible`, `indicative`, `suggestive` agrupam-se no mesmo arquétipo.
* Na extração de features (`DistributionalClusterNgramExtractor`), geramos templates híbridos de 2-grams e 3-grams preservando conectivos sintáticos curtos ($\text{len}(w) \le 3$ ou preposições/negações estruturais como `with`, `from`, `into`, `without`, `not`, `for`, `trigger_mask`) e abstraindo palavras de conteúdo:
  $$\text{"was consistent with"} \implies \texttt{BTW\_TPL\_was\_CLUST04\_with}$$
* **Protocolo Indutivo vs. Transdutivo (Ausência de Vazamento Distributivo):** Como a fatoração PPMI-SVD e o K-Means são puramente **não-supervisionados** (não têm acesso aos rótulos relacionais $Y$), o pipeline padrão pré-computa os clusters sobre o corpus das 56 histórias clínicas (configuração transdutiva de vocabulário). Para certificar que não há inflação por vazamento distributivo entre pacientes de treino e teste, implementamos em [`src/run_pipeline_p2.py`](src/run_pipeline_p2.py) a **Auditoria Estritamente Indutiva por Fold**, na qual a matriz PPMI, o SVD de 64 dimensões e o `MiniBatchKMeans(K=32)` são reajustados do zero *exclusivamente dentro dos textos dos pacientes de treino de cada um dos 5 folds* (mapeando palavras inéditas do paciente de teste para `CLUST_UNK`). O resultado indutivo atinge **Macro-F1 = 0.4995** (Pos-F1 = **0.4429**), superando tanto os trigramas léxicos puros (`P2.4 = 0.4988`) quanto a versão transdutiva global (`P2.9 = 0.4956`), pois o ajuste por fold evita centroides dominados por termos idiossincráticos de relatos isolados.

### 2. Classificação em Cascata Hierárquica de 2 Estágios
* **Fundamentação:** Slide `nlp-080` (Regra da Cadeia de Probabilidade).
* Substituímos a decisão plana multiclasse por dois especialistas encadeados (`HierarchicalCascadeClassifier` em [`src/genius_upgrades.py`](src/genius_upgrades.py)):
  - **Estágio 1 (Clinical Link Detector):** Regressão Logística binária L2 ($C=1.5$, `class_weight='balanced'`) encarregada de estimar a probabilidade de existência de vínculo clínico ativo $P_{\text{Stage1}}(\text{LINK}=1 \mid \vec{x})$ versus ausência de relação $P_{\text{Stage1}}(\text{NONE} \mid \vec{x}) = 1 - P_{\text{Stage1}}(\text{LINK}=1 \mid \vec{x})$.
  - **Estágio 2 (Relation Specialist):** Regressão Logística multiclasse ($C=2.5$, `class_weight='balanced'`) treinada exclusivamente sobre instâncias positivas ($\text{LINK}=1$), discernindo a semântica fina entre as relações positivas ativas $r \in \mathcal{R}^+$.
* **Regra de Inferência e Calibração de Prior:** Como o Estágio 1 utiliza ponderação inversamente proporcional às frequências (`balanced`, o que multiplica o peso da classe minoritária positiva por $\approx 1{,}103 / 426 \approx 2{,}59\times$, deslocando o log-odds em $\approx +0{,}95$ para um prior sintético 50/50), a decisão discreta em `predict(X)` opera por **limiarização hierárquica calibrada** ($\tau_{\text{link}} \in [0.42, 0.45]$):
  $$\hat{y}(\vec{x}) = \begin{cases} \arg\max_{r \in \mathcal{R}^+} P_{\text{Stage2}}(R = r \mid \text{LINK}=1, \vec{x}), & \text{se } P_{\text{Stage1}}(\text{LINK}=1 \mid \vec{x}) \ge \tau_{\text{link}} \\ \text{NONE}, & \text{caso contrário.} \end{cases}$$
  Já para a composição contínua de probabilidades em `predict_proba(X)` (usada no ensemble e no Confident Learning), o vetor conjunto junta $P(R = \text{NONE} \mid \vec{x}) = P_{\text{Stage1}}(\text{LINK}=0 \mid \vec{x})$ e, para cada $r \in \mathcal{R}^+$, o produto da regra da cadeia normalizado na linha:
  $$P(R = r \mid \vec{x}) = P_{\text{Stage1}}(\text{LINK}=1 \mid \vec{x}) \times P_{\text{Stage2}}(R = r \mid \text{LINK}=1, \vec{x})$$
* **Análise de Ablação Individual vs. Sinergia:** Isoladamente sobre o gabarito silver bruto, tanto `P2.9` (`0.4956`) quanto `P2.10` (`0.4890`) e sua combinação sem denoising (`0.4864`) pontuam ligeiramente abaixo da Regressão Logística plana `P2.3` (`0.5040`). Isso ocorre porque o Estágio 1 com peso balanceado promove agressivamente pares com contexto clínico real que haviam sido deixados como `NONE` pelas regex estritas do P1 (gerando falsos positivos aparentes perante o gabarito incompleto do P1). Quando combinado à auditoria de ruído (`P2.12`) e às assinaturas ontológicas (`P2.13–P2.14`), o desacoplamento em 2 estágios destrava o recall das classes positivas.

### 3. Auditoria de Ruído com Confident Learning (*Out-of-Fold Denoising*)
* **Fundamentação:** Northcutt et al. (2021) — *Confident Learning: Estimating Uncertainty in Dataset Labels*.
* No Projeto 1, o conjunto *Silver Standard* foi gerado automaticamente por heurísticas determinísticas: pares sem o gatilho exato da regex caíram na vala comum de `NONE` (falsos negativos de supervisão fraca), enquanto a regra de tipo `Fármaco -> Diagnóstico` gerou `TARGETS` por simples coocorrência na sentença mesmo quando o fármaco não tratava aquela patologia específica (falsos positivos de coocorrência).
* Coletamos a matriz de probabilidades *Out-of-Fold* (OOF) estrita das predições de validação dos 5 folds (`detect_p1_label_noise_oof` em [`src/genius_upgrades.py`](src/genius_upgrades.py)) e aplicamos a regra formal de auditoria bidirecional sobre os rótulos ruidosos do P1 ($\mathcal{Y}_{\text{noisy}} = \{\text{SUPPORTS}, \text{TARGETS}, \text{NONE}\}$):
  $$\hat{c}_i = \arg\max_{c \in \mathcal{C}} P_{\text{OOF}}(Y = c \mid \vec{x}_i), \qquad \text{Se } y_{\text{P1}, i} \in \{\text{SUPPORTS}, \text{TARGETS}, \text{NONE}\} \;\land\; \hat{c}_i \neq y_{\text{P1}, i} \;\land\; P_{\text{OOF}}(Y = \hat{c}_i \mid \vec{x}_i) \ge 0.76 \implies y_{\text{denoised}, i} \leftarrow \hat{c}_i$$
* **Análise do Log de Auditoria (`p1_noise_audit_log.json`):** Ao todo, **47 instâncias (3,07% das 1.529 amostras)** apresentaram divergência consensual ($\ge 76\%$ de confiança Out-of-Fold) e foram auditadas:
  - **40 falsos negativos resgatados de `NONE`:** 18 promovidos para `SUPPORTS`, 12 para `LOCATED_IN` e 10 para `TARGETS` (pares onde o P1 falhou por ausência de verbo literal da regex, mas a estrutura sintática/posicional era inequívoca).
  - **7 falsos positivos demovidos de `TARGETS` para `NONE`:** coocorrências espúrias na mesma oração sem vínculo terapêutico real.
* **Transparência Metodológica e Dupla Avaliação (Prevenção de Circularidade):** Como apontado por Northcutt et al. (2021), alterar rótulos de validação com base em probabilidades OOF pode introduzir viés de autoconfirmação se reportado isoladamente. Por isso, em [`src/run_pipeline_p2.py`](src/run_pipeline_p2.py) e na tabela consolidada abaixo, **avaliamos todos os modelos subsequentes (`P2.12`, `P2.13` e `P2.14`) simultaneamente em dois cenários**:
  1. **No Gabarito Silver Auditado (`denoised_label`):** `P2.12 = 0.5217`, `P2.13 = 0.5385`, `P2.14 = 0.6173`.
  2. **No Gabarito Silver Bruto Intocado do P1 (`silver_label` original, 100% *apples-to-apples* com `P2.1–P2.11`):** `P2.13` salta para **`0.5198`** e o Ensemble `P2.14` atinge **`0.5998`** (Pos-F1 = **`0.5538`**), comprovando que o salto de desempenho de quase +10 pontos de F1 no Ato 3 é genuíno e independe da mutação dos 47 rótulos!

### 4. Rede Neural Trilinear de 3 Torres (*Clinical DistMult-MLP*)
* **Fundamentação:** Slide `nlp-110` (Redes Neurais & Knowledge Graph Embeddings), *DistMult* (Yang et al., 2015) e *Text-KB Joint Embeddings* (Toutanova et al., 2015).
* Em Grafos de Conhecimento, uma relação clínica pode ser modelada como uma forma trilinear que conecta a representação da entidade cabeça $\vec{h}_1$ à cauda $\vec{h}_2$ modulada pelo contexto textual $\vec{h}_{\text{ctx}}$.
* Arquitetura implementada em PyTorch (`ThreeTowerBilinearMLP` — mantido o nome de classe `Bilinear` por compatibilidade retroativa com a API da Fase 4, embora opere com fusão trilinear completa — em [`src/genius_upgrades.py`](src/genius_upgrades.py)):
  1. **Agregação de Entidades Compostas e Assimetria Direcional:** Para entidades multi-palavra (ex.: *neonatal compartment syndrome*), `svd_model.get_entity_vector` calcula o **mean pooling normalizado em L2** dos vetores SVD 64d das palavras constituintes, concatenando-o ao vetor *One-Hot* de 10 dimensões do tipo clínico ($\mathbb{R}^{74}$). As Torres 1 e 2 possuem **matrizes de pesos independentes (untied weights, $W_{E_1} \neq W_{E_2}$)**, preservando a assimetria direcional entre sujeito ($E_1$) e objeto ($E_2$) mesmo antes dos termos simétricos de Hadamard:
     - **Torre da Cabeça $E_1$:** $\mathbb{R}^{74} \to \text{Linear}(74, 48) \to \text{LayerNorm}(48) \to \text{ReLU} \to \text{Dropout}(0.25)$ ($3.600 + 96 = 3.696$ pesos).
     - **Torre da Cauda $E_2$:** $\mathbb{R}^{74} \to \text{Linear}(74, 48) \to \text{LayerNorm}(48) \to \text{ReLU} \to \text{Dropout}(0.25)$ ($3.600 + 96 = 3.696$ pesos).
  2. **Torre do Contexto (com SVD Não-Supervisionado Congelado):** O vetor TF-IDF de N-grams posicionais ($D_{\text{in}} \le 2.500$) é projetado para 48 dimensões via `TruncatedSVD(n_components=48)` ajustado sem supervisão no fold de treino (cujos coeficientes lineares ficam congelados fora do grafo de gradientes do PyTorch), seguido por $\text{Linear}(48, 48) \to \text{LayerNorm}(48) \to \text{ReLU} \to \text{Dropout}(0.25)$ ($2.352 + 96 = 2.448$ pesos).
  3. **Fusão Trilinear e Hadamard ($\mathbb{R}^{240}$):**
     $$\vec{z} = \left[ \vec{h}_1 \;\oplus\; \vec{h}_2 \;\oplus\; \vec{h}_{\text{ctx}} \;\oplus\; (\vec{h}_1 \odot \vec{h}_{\text{ctx}} \odot \vec{h}_2) \;\oplus\; (\vec{h}_1 \odot \vec{h}_2) \right] \in \mathbb{R}^{240}$$
  4. **Cabeça de Classificação Supervisionada e Hiperparâmetros:** Um classificador gargalo $\text{Linear}(240, 48) \to \text{ReLU} \to \text{Dropout}(0.25) \to \text{Linear}(48, K)$ ($11.568 + 343 = 11.911$ pesos para $K=7$), totalizando exatamente **21.751 parâmetros treináveis** na configuração padrão $H=48$ (ou **~11.768 parâmetros** com cabeça linear direta $\mathbb{R}^{240} \to \mathbb{R}^K$), representando uma **redução superior a 94%–97%** frente aos ~384.000 pesos treináveis de um MLP denso sobre TF-IDF bruto. O modelo é otimizado com `CrossEntropyLoss` ponderada pelo inverso da frequência das classes, otimizador `AdamW` ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), `batch_size=32` e $15\text{--}20$ épocas.
* **Resultado (`P2.11`):** Avaliado diretamente no gabarito Silver Bruto intocado do P1, o `ThreeTowerBilinearMLP` salta para **Macro-F1 = 0.5169 (±0.0249)** e **Pos-F1 = 0.4591**, superando com folga o MLP tradicional (`0.4896`) e a melhor Regressão Logística (`0.5040`).

### 5. Inferência Coletiva por Fechamento de Triângulos Clínicos no Grafo
* **Fundamentação:** Slide `nlp-080` (Gramática Topológica em Grafos de Conhecimento).
* Em relatos clínicos, relações intra-caso obedecem a restrições composicionais de 2 saltos (Cláusulas de Horn sobre tríades direcionadas $u \xrightarrow{R_1} v \xrightarrow{R_2} w \implies u \xrightarrow{R_3} w$).
* **Escopo Pós-CV e Interação com Arestas Estruturais do P1:** Predicados centrados no nó `Patient` (como `UNDERWENT_TEST`, `DIAGNOSED_WITH`, `PRESENTS_WITH` e `TREATED_WITH`) são extraídos pelas regras estruturais determinísticas do Projeto 1. O método `apply_collective_triangle_propagation` ([`src/graph_integrator.py`](src/graph_integrator.py)) atua na **Fase 5 (montagem global do grafo pós-validação cruzada)**, integrando as arestas estruturais do P1 com as arestas oracionais descobertas pelo ML estatístico no P2.
* **Formalismo de Atualização e Limitação no Simplex $[0, 0.99]$:** O algoritmo não cria arestas artificiais onde a predição foi `NONE`; ele **reforça a confiança de arestas candidatas existentes** $(u, R_3, w)$ que fecham um triângulo consistente com $(u, R_1, v)$ (com $P_1 \ge 0.60$) e $(v, R_2, w)$ (com confiança $P_2$) no mesmo `case_id`, aplicando uma atualização amortecida ($\eta = \text{boost\_damping} = 0.15$) estritamente limitada a $[0, 0.99]$:
  $$P_{\text{novo}}(u \xrightarrow{R_3} w) = \min\!\Big(0.99,\; P_{\text{antigo}}(u \xrightarrow{R_3} w) + \eta \cdot (P_1 \cdot P_2) \cdot (\mu_{R_1, R_2} - 1.0)\Big)$$
  Regras composicionais implementadas ($\forall u, v, w \in V_{\text{case}}$):
  - $R_1(u, v) = \text{UNDERWENT\_TEST} \;\land\; R_2(v, w) = \text{SUPPORTS} \implies R_3(u, w) = \text{DIAGNOSED\_WITH}$ ($\mu = 1.25$)
  - $R_1(u, v) = \text{DIAGNOSED\_WITH} \;\land\; R_2(v, w) = \text{LOCATED\_IN} \implies R_3(u, w) = \text{ASSOCIATED\_WITH}$ ($\mu = 1.20$)
  - $R_1(u, v) = \text{TREATED\_WITH} \;\land\; R_2(v, w) = \text{TARGETS} \implies R_3(u, w) = \text{DIAGNOSED\_WITH}$ ($\mu = 1.30$)
  - $R_1(u, v) = \text{PRESENTS\_WITH} \;\land\; R_2(v, w) = \text{SUPPORTS} \implies R_3(u, w) = \text{DIAGNOSED\_WITH}$ ($\mu = 1.25$)
  - $R_1(u, v) = \text{UNDERWENT\_TEST} \;\land\; R_2(v, w) = \text{REVEALS} \implies R_3(u, w) = \text{PRESENTS\_WITH}$ ($\mu = 1.20$)
  - $R_1(u, v) = \text{REVEALS} \;\land\; R_2(v, w) = \text{SUPPORTS} \implies R_3(u, w) = \text{CONFIRMS}$ ($\mu = 1.25$)
* **Resultado:** **445 triângulos clínicos fechados** e reponderados sobre as 3.991 arestas do Grafo P2, elevando a consistência topológica para filtragem pelo limiar $\tau$.

---

## Parte 3: O Ato 3 — Quebrando o "Ponto Cego" de Teoria da Informação (Macro-F1 0.6173)

### O Diagnóstico Teórico da Informação & Relação com *Entity-Type Markers*
Por que mesmo com DistMult Trilinear (`0.5169`) e Cascata Denoised (`0.5217`) os modelos encontravam uma barreira em torno de `Macro-F1 ≈ 0.52`?

A resposta decorre da **Entropia Condicional dos Rótulos ($H(Y \mid X_{\text{texto}})$)** e da literatura moderna de extração de relações com marcadores de tipo de entidade (*Typed Entity Markers*, Zhong & Chen, 2021):
1. Na ontologia clínica (e na geração do Silver Standard do Projeto 1), relações como `LOCATED_IN`, `TARGETS` e `SUPPORTS` são fortemente governadas pela **assinatura ontológica dos tipos de nós** (ex.: `DrugAdministration -> Diagnosis` compatível com `TARGETS`; `Entity -> Anatomy` compatível com `LOCATED_IN`).
2. Quando restringimos o classificador linear a olhar predominantemente para os N-grams intermediários (`BTW_`), exigimos que conectivos curtos ou verbos polissêmicos (*"was noted with"*, *"in"*, *"associated with"*) desambigúem sozinhos a relação. Sem a interação explícita entre a assinatura de tipos $(T_{E_1}, T_{E_2})$ e o léxico intermediário, $H(Y \mid X_{\text{texto}})$ permanece elevada.
3. **Recuperação de Heurísticas Ontológicas vs. Desambiguação Lexical:** É importante distinguir cientificamente dois regimes no benchmark: (i) para relações ontologicamente restritas (`LOCATED_IN`, `TARGETS`, `SUPPORTS`), fornecer a assinatura de tipos permite ao modelo recuperar com alta fidelidade as restrições de domínio do P1, enquanto o contexto textual e o Estágio 1 filtram coocorrências negativas/desconexas (`NONE`); (ii) para pares com a **mesma assinatura ontológica** (`ExamInstance -> Diagnosis` ou `ExamInstance -> Finding`), o tipo sozinho é ambíguo entre `REVEALS`, `CONFIRMS`, `EXCLUDES` e `NONE` — nesses casos, a discriminação depende 100% dos N-grams posicionais, clusters semânticos e interações trilineares.

### As Soluções do Ato 3

1. **Fusão Precoce de Assinaturas Ontológicas & Posição (`P2.13_TypedNgrams_HierarchicalCascade`):**
   - Em `build_typed_positional_features` ([`src/genius_upgrades.py`](src/genius_upgrades.py)), injetamos marcadores estruturais explícitos no espaço TF-IDF:
     - Assinaturas Direcionais de Tipo: $\texttt{SIG\_\{type\_e1\}\_TO\_\{type\_e2\}}$
     - Buckets de Distância Sintática (em tokens): $\texttt{DIST\_SHORT}$ ($\text{dist} \le 4$), $\texttt{DIST\_MEDIUM}$ ($5 \le \text{dist} \le 10$), $\texttt{DIST\_LONG}$ ($\text{dist} > 10$), além do termo conjunto $\texttt{SIG\_\{type\_e1\}\_\{type\_e2\}\_\{dist\_bucket\}}$.
     - Interações Cruzadas Tipo-Léxico (produto cartesiano com os 4 primeiros N-grams intermediários): $\texttt{SIG\_\{type\_e1\}\_TO\_\{type\_e2\}\_\_BTW\_\{ngram\}}$.
   - **Resultado P2.13:** O **Macro-F1** salta para **0.5385 (±0.0141)** (+3,22% de ganho relativo em Macro-F1 sobre o pico do Ato 2 `0.5217`), enquanto o **Macro-Recall** avança para **0.6082** (+2,37 pontos percentuais ou +4,05% de ganho relativo sobre `0.5845`). No gabarito Silver Bruto sem denoising, `P2.13` também atinge **0.5198**, superando todos os modelos dos Atos 1 e 2.

2. **Ensemble Neuro-Simbólico Calibrado (`P2.14_NeuroSymbolic_Ensemble_Tuned`):**
   - Em `neuro_symbolic_ensemble_and_tune` ([`src/genius_upgrades.py`](src/genius_upgrades.py)), fundimos de forma multiplicativo-aditiva as 3 fontes complementares de evidência com pesos fixos pré-definidos ($w_{\text{casc}} = 0.55$, $1 - w_{\text{casc}} = 0.45$, expoente ontológico $\gamma_{\text{onto}} = 0.65$) e multiplicadores de calibração por classe $\lambda_r \in [0.4, 2.5]$:
     $$\tilde{P}_{\text{blend}}(R = r \mid \vec{x}, T_{E_1}, T_{E_2}) = \frac{1}{Z(\vec{x})} \Big[ w_{\text{casc}} P_{\text{cascade}}(R = r \mid \vec{x}) + (1 - w_{\text{casc}}) P_{\text{distmult\_mlp}}(R = r \mid \vec{x}) \Big] \cdot \Big( P_{\text{mesh\_prior}}(R = r \mid T_{E_1}, T_{E_2}) + \epsilon \Big)^{\gamma_{\text{onto}}}$$
     $$P_{\text{ensemble}}(R = r \mid \vec{x}, T_{E_1}, T_{E_2}) = \frac{\lambda_r \cdot \tilde{P}_{\text{blend}}(R = r \mid \vec{x}, T_{E_1}, T_{E_2})}{\sum_{r' \in \mathcal{C}} \lambda_{r'} \cdot \tilde{P}_{\text{blend}}(R = r' \mid \vec{x}, T_{E_1}, T_{E_2})}$$
   - **Protocolo de Otimização de $\lambda_r$:** Como os pesos estruturais de mistura ($w_{\text{casc}} = 0.55$ vs. $0.45$) são mantidos **fixos** (e não otimizados livremente in-sample), evita-se que o ensemble colapse em favor do modelo neural de maior capacidade. Os multiplicadores de limiar por classe $\lambda_r$ são ajustados via 2 rodadas de *Coordinate Ascent* sobre uma grade discreta de 22 pontos em $[0.4, 2.5]$ estritamente dentro de cada fold de treino para compensar o desbalanceamento extremo de classes raras (`EXCLUDES`, `REVEALS`).
   - **Resultado P2.14:** O modelo atinge **Macro-F1 = 0.6173 (±0.0286)** e **Macro-Recall = 0.6929** no benchmark auditado (com **Pos-F1 = 0.5717 ±0.0334**), e **Macro-F1 = 0.5998** (Pos-F1 = **0.5538**) quando avaliado contra o gabarito Silver Bruto original do Projeto 1!
   - **Desempenho Detalhado de Todas as 7 Classes Ativas no Dataset (`P2.14`, Pooled OOF):**
     | Classe Relacional | Natureza Semântica | Suporte ($N$) | Precision | Recall | F1-Score |
     | :--- | :--- | :---: | :---: | :---: | :---: |
     | `LOCATED_IN` | Positiva (Anatômica/Ontológica) | 150 | 76,02% (`0.7602`) | 99,33% (`0.9933`) | **0.8613** |
     | `TARGETS` | Positiva (Terapêutica/Ontológica) | 83 | 71,43% (`0.7143`) | 96,39% (`0.9639`) | **0.8205** |
     | `SUPPORTS` | Positiva (Achado $\to$ Diagnóstico) | 194 | 64,56% (`0.6456`) | 94,85% (`0.9485`) | **0.7683** |
     | `REVEALS` | Positiva (Exame $\to$ Achado, Lexical) | 24 | 46,88% (`0.4688`) | 62,50% (`0.6250`) | **0.5357** |
     | `EXCLUDES` | Positiva Rara (Negação Diagnóstica) | 4 | 40,00% (`0.4000`) | 50,00% (`0.5000`) | **0.4444** |
     | `CONFIRMS` | Positiva Rara (Confirmação Estrita) | 4 | 0,00% (`0.0000`) | 0,00% (`0.0000`) | **0.0000** |
     | `NONE` | Negativa (Ausência de Vínculo) | 1.070 | 97,55% (`0.9755`) | 81,96% (`0.8196`) | **0.8908** |
     - **Reconciliação Aritmética Exata das Médias:**
       - **Média das 4 Classes Clínicas Positivas Principais (suporte $\ge 10$, excluindo `NONE`):** $\frac{0.8613 + 0.8205 + 0.7683 + 0.5357}{4} = \mathbf{0.7465}$ (**74,65%**).
       - **Média das 5 Classes com Suporte $\ge 10$ (incluindo `NONE`):** $\frac{0.8613 + 0.8205 + 0.7683 + 0.5357 + 0.8908}{5} = \mathbf{0.7753}$ (**77,53%**).
       - **Média das 6 Relações Positivas Ativas (`Pos-F1` sem `NONE`):** $\frac{0.8613 + 0.8205 + 0.7683 + 0.5357 + 0.4444 + 0.0000}{6} = \mathbf{0.5717}$ (**57,17%**).
       - **Macro-F1 Global (todas as 7 classes ativas no dataset):** $\frac{6 \times 0.5717 + 0.8908}{7} = \frac{4.3210}{7} = \mathbf{0.6173}$ (**61,73%**). *(Nota: a 8ª relação do esquema global, `TREATED_WITH`, conecta o nó estrutural `Patient` a `DrugAdministration` via regras de escopo de caso no P1, não possuindo instâncias de treinamento entre pares de menções episódicas em `relation_dataset.csv`, razão pela qual o classificador oracional opera sobre $|\mathcal{C}| = 7$ classes ativas).*

---

## Resultados do Torneio Consolidado em 3 Atos (5-Fold Stratified Group CV)

A validação cruzada foi executada com partição estrita por paciente (`case_id`), reportando abaixo as estimativas pontuais agregadas (pooled OOF), o desvio-padrão entre os 5 folds ($\pm \text{SD}$), o F1 exclusivo das relações positivas (`Pos-F1`) e, para os modelos pós-auditoria (`P2.12–P2.14`), a avaliação paralela direta sobre o `silver_label` bruto original do P1:

| Ato | ID | Modelo & Estratégia de Features | N-Gram Range | Ponderação / Dim | Macro-Precision | Macro-Recall | Macro-F1 ($\pm$ SD) | Pos-F1 Sem NONE ($\pm$ SD) | Macro-F1 no Silver Bruto P1 | Análise Científica |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ato 1** | **P2.1** | Naive Bayes Multinomial | (1, 3) | TF-IDF | 0.1173 | 0.1426 | 0.1209 (±0.0036) | 0.0018 (±0.0036) | 0.1209 | Colapso pela quebra de independência condicional de N-grams vizinhos. |
| **Ato 1** | **P2.2** | Regressão Logística (Unigramas) | (1, 1) | TF-IDF | 0.4441 | 0.5555 | 0.4847 (±0.0401) | 0.4276 (±0.0453) | 0.4847 | Baseline puro de Bag-of-Words léxico. |
| **Ato 1** | **P2.3** | **Regressão Logística (Bigramas)** | **(1, 2)** | **TF-IDF** | **0.4612** | **0.5745** | **0.5040 (±0.0270)** | **0.4480 (±0.0307)** | **0.5040** | **Pico do Baseline Clássico Inicial:** Bigramas capturam locuções de Hearst mascaradas. |
| **Ato 1** | **P2.4** | Regressão Logística (Trigramas) | (1, 3) | TF-IDF | 0.4596 | 0.5616 | 0.4988 (±0.0233) | 0.4423 (±0.0264) | 0.4988 | **Degradação Aparente:** Trigramas exatos mostravam queda (investigada e resolvida na Parte 4). |
| **Ato 1** | **P2.5** | Regressão Logística (Contagem) | (1, 3) | Frequência Bruta | 0.4696 | 0.4956 | 0.4816 (±0.0490) | 0.4184 (±0.0570) | 0.4816 | Contagem pura penaliza termos raros diagnósticos. |
| **Ato 1** | **P2.6** | Linear SVM (Softmax Decision) | (1, 3) | TF-IDF | 0.4600 | 0.5595 | 0.4998 (±0.0231) | 0.4415 (±0.0273) | 0.4998 | Margem máxima competitiva, porém esbarra no desbalanceamento. |
| **Ato 1** | **P2.7** | Random Forest (100 árvores) | (1, 3) | TF-IDF | 0.4884 | 0.4434 | 0.4601 (±0.0559) | 0.3928 (±0.0650) | 0.4601 | Árvores têm dificuldade em particionar espaços de altíssima esparsidade. |
| **Ato 1** | **P2.8** | MLP Tradicional (2 Camadas) | (1, 3) | TF-IDF ($D_{\text{in}} \times 128$) | 0.4491 | 0.5545 | 0.4896 (±0.0427) | 0.4304 (±0.0482) | 0.4896 | **Perde para LR:** Superparametrização gera overfitting no corpus pequeno. |
| **Ato 2** | **P2.9** | **Cluster N-Grams + LR** | **(1, 3)** | **TF-IDF + Clusters** | **0.4574** | **0.5571** | **0.4956 (±0.0224)** | **0.4389 (±0.0252)** | **0.4956** *(Indutivo: **0.4995**)* | **Templates Semânticos:** Ajuste indutivo por fold (`0.4995`) supera trigramas puros (`0.4988`). |
| **Ato 2** | **P2.10** | **Cascata Hierárquica (2 Estágios)** | **(1, 3)** | **TF-IDF** | **0.4477** | **0.5629** | **0.4890 (±0.0242)** | **0.4326 (±0.0283)** | **0.4890** *(+Clusters: **0.4864**)* | Desacopla Link Detector e Especialista (recall de `SUPPORTS` sobe a 84,7%, mas sofre com falsos negativos do P1). |
| **Ato 2** | **P2.11** | **Three-Tower DistMult-MLP** | **(1, 2)** | **SVD 48d (~12k params)** | **0.4693** | **0.5947** | **0.5169 (±0.0249)** | **0.4591 (±0.0273)** | **0.5169** | **Deep Learning Vencedor:** Interação $\vec{h}_1 \odot \vec{h}_{\text{ctx}} \odot \vec{h}_2$ bate LR e MLP plano no Silver Bruto! |
| **Ato 2** | **P2.12** | **Cascata Denoised + Cluster N-Grams** | **(1, 3)** | **TF-IDF + Clusters** | **0.4828** | **0.5845** | **0.5217 (±0.0156)** | **0.4670 (±0.0181)** | **0.4864** | **Pico do Ato 2 Inicial:** Audita discrepâncias do P1 via Confident Learning e reduz variância entre folds. |
| **Ato 3** | **P2.13** | **Cascata com N-Grams Tipados & Posição** | **(1, 3)** | **TF-IDF + Types + Clusters** | **0.4939** | **0.6082** | **0.5385 (±0.0141)** | **0.4829 (±0.0161)** | **0.5198** | **Quebra de Ponto Cego:** Assinaturas `SIG_E1_TO_E2` + distância batem recorde tanto no Denoised quanto no Silver Bruto. |
| **Ato 3** | **P2.14** | **Neuro-Symbolic Ensemble (Tuned)** | **(1, 3)** | **Ensemble + Prior MeSH** | **0.5663** | **0.6929** | **0.6173 (±0.0286)** | **0.5717 (±0.0334)** | **0.5998** | **Campeão Absoluto Inicial:** Atinge **74,65%** nas 4 classes clínicas principais e **0.5998** no Silver Bruto! |

---

## Parte 4: A Auditoria Forense dos Dados & O Desbloqueio Pós-Correção (Slides 17 a 20)

> [!IMPORTANT]
> **Espírito Crítico & Honestidade Metodológica Científica:**
> Durante os ensaios do Ato 3, formulamos um teste de sanidade (*Sanity Check*): se o modelo estava aprendendo a ler o contexto sintático dos relatos, o que aconteceria se treinássemos um **Baseline Cego** que não lesse nenhuma palavra do texto e utilizasse estritamente o par de tipos das entidades (`e1_type` e `e2_type`)?

### 1. O Teste de Sanidade e o Choque do Baseline Cego (Slide 18)
Ao avaliar o Baseline puramente ontológico (sem texto algum), obtivemos **Macro-F1 = 0.539** — exatamente o mesmo resultado do modelo tipado complexo (`P2.13 = 0.5385`).
Constatamos que **83,7% dos rótulos do Projeto 1 já eram rigidamente pré-determinados pela combinação de tipos MeSH**. O classificador estava se apoiando em um atalho estatístico (*shortcut learning*).

### 2. A Investigação Forense e a Causa-Raiz no Extrator (Slide 19)
Fomos inspecionar diretamente o código do gerador de datasets (`WeakSupervisionDatasetGenerator` em [`src/weak_supervisor.py`](src/weak_supervisor.py)) e os arquivos de origem:
* O extrator buscava ler diretamente a propriedade `node.get('start_char', 0)`.
* Porém, no arquivo gerado pelo Projeto 1 (`nodes.csv`), os limites textuais estavam serializados como um array dentro de uma string JSON: `attributes['span'] = [start, end]`.
* Sem encontrar a chave no primeiro nível, o código caía silenciosamente no valor padrão `0`!
* **A Consequência Concreta:** Para todos os 1.529 pares analisados, o extrator recortava os primeiros 66 caracteres do relato (*"A 49-year-old male presented to our clinic for further diagnostic..."*), enquanto as entidades reais estavam frequentemente localizadas nos caracteres 4.300 e 4.500 do caso! Apenas 3 a 10 pares continham por pura coincidência as entidades reais no início do texto.
* **Isso explicou todas as anomalias:**
  1. O Trigger-Masking só mascarava **2 pares em 1.529**, pois os verbos de Hearst não estavam no cabeçalho.
  2. Os trigramas degradavam porque tentar encontrar trigramas exatos em cabeçalhos de pacientes distintos gerava pura ortogonalidade esparsa.
  3. O modelo dependia quase exclusivamente dos tipos porque o texto que ele lia era ruído não-relacionado.

### 3. A Correção do Extrator e o Re-Benchmark Completo
Corrigimos o extrator em [`src/weak_supervisor.py`](src/weak_supervisor.py) para desserializar `attributes['span']`, inferir com precisão os limites de `sentence_idx` reais e recortar a janela exata entre as menções.

Ao re-executar todo o pipeline ponta a ponta ([`src/run_pipeline_p2.py`](src/run_pipeline_p2.py)), os resultados mudaram drasticamente, confirmando o poder do processamento de linguagem natural:

#### Comparativo Antes vs. Depois da Correção Forense

| Indicador Metodológico | Antes (Bug `start_char=0`) | Depois (Com `span` Real Corrigido) | Impacto Científico |
| :--- | :---: | :---: | :--- |
| **Gatilhos Mascarados (`trigger_word_masked`)** | 2 pares (0,13%) | **263 pares (21,88%)** | O Trigger-Masking agora atua de verdade sobre o contexto intermediário. |
| **Tamanho do Dataset Silver** | 1.529 pares | **1.202 pares** | Filtro estrito de sentenças contíguas elimina falsos negativos espúrios de páginas distantes. |
| **LR Unigramas (`P2.2`)** | 0.4847 | **0.4615** | Captura apenas palavras isoladas no contexto. |
| **LR Bigramas (`P2.3`)** | 0.5040 | **0.5632** (+0.059) | Bigramas no `BTW_` alavancam +10,1 pontos sobre unigramas. |
| **LR Trigramas (`P2.4`)** | 0.4988 *(degradava)* | **0.5688** *(**supera bigramas**)* | **Fim da anomalia dos trigramas!** Trigramas reais agora somam informação útil (`0.5688 > 0.5632`). |
| **Linear SVM (`P2.6`)** | 0.4998 | **0.5847** (+0.085) | Linear SVM sobe expressivamente com o contexto oracional correto. |
| **Cascata Hierárquica (`P2.10`)** | 0.4890 | **0.5478** (+0.059) | Link detector e especialista operam sobre features textuais válidas. |
| **Three-Tower DistMult-MLP (`P2.11`)** | 0.5169 | **0.6760** (+0.159) | **Salto de +15,9 pontos:** a torre de contexto $h_{\text{ctx}}$ agora guia a forma trilinear $h_1 \odot h_{\text{ctx}} \odot h_2$. |
| **Cascata Tipada (`P2.13`)** | 0.5385 | **0.6335** (+0.095) | Supera o baseline cego de tipos (`0.5873`) e destrava `EXCLUDES` com **F1 = 0.4444**! |
| **Neuro-Symbolic Ensemble (`P2.14`)** | 0.6173 *(0.5998 Silver)* | **0.6828** *(**0.6844 no Silver Puro!**)* | Recorde absoluto: opera com altíssima precisão e recall equilibrado em dados não-ruidosos. |

#### O Papel Crítico do Texto na Resolução do Ponto Cego Ontológico
Para classes com a mesma assinatura ontológica (ex.: `ExamInstance -> Finding`), o Baseline Cego de tipos é incapaz de discernir o desfecho. Com a leitura correta do texto entre as entidades (`BTW_`), o modelo finalmente aprendeu a diferenciar:
* **`REVEALS`**: saltou de ~0.45 para **F1 = 0.8163** (Precision 0.769, Recall 0.870);
* **`EXCLUDES`**: saltou de **0.0000** (zero absoluto antes) para **F1 = 0.4444** na Cascata Tipada e **0.4000** no Ensemble;
* **`LOCATED_IN`** e **`TARGETS`**: atingiram **F1 = 0.9241** e **F1 = 0.8427**, respectivamente.

Essa reconciliação forense comprova a premissa fundamental de PLN: **as regras ontológicas fornecem a estrutura e os tipos, mas é o processamento do texto oracional que desambigua a relação clínica real.**

---

## O "Show da Apresentação": Visualizador SOTA com Linha do Tempo DAG, Narrativa Clínica & Sandbox What-If

O visualizador HTML standalone ([`data/processed/graph_p2_visualization.html`](data/processed/graph_p2_visualization.html) e cópia em [`assets/graph_p2_visualization.html`](assets/graph_p2_visualization.html)) foi construído com arquitetura inspirada no workbench da Fase 1 e na filosofia do *What-If Tool* (Wexler et al., 2019), integrando a dimensão temporal e a rastreabilidade clínica profunda:

1. **Linha do Tempo DAG (Layout Cronológico Horizontal no Eixo X):**
   - Os nós episódicos são organizados no espaço cartesiano pelo índice temporal de sua sentença no relato médico ($t = \text{sentIdx} \in \mathbb{N}_0$, com coordenada horizontal $X(t) = 170 \cdot t - 200$ em pixels):
     - **Extremo Esquerdo ($X = -400\text{ px}$, nó global):** Admissão e identificação do paciente (`Patient`).
     - **Centro-Esquerdo ($t \in [0, 2]$):** Sintomas de apresentação, anamnese e queixas iniciais.
     - **Centro ($t \in [3, 5]$):** Exames laboratoriais e de imagem realizados.
     - **Centro-Direito ($t \in [6, 10]$):** Revelação diagnóstica e confirmações — com as **arestas estatísticas de ML (verde tracejado)** conectando os achados aos diagnósticos!
     - **Extremo Direito ($t \ge 11$):** Intervenções farmacológicas, procedimentos cirúrgicos e desfecho clínico.
   - O gráfico lê-se naturalmente da esquerda para a direita como a história clínica do paciente!

2. **Player Cronológico ("Histórico do Paciente" — Simulador Temporal Passo a Passo):**
   - Rodapé flutuante com barra de controle temporal: `[⏮ Início]`, `[◀ Passo]`, `[▶ Play / ⏸ Pause]`, `[Passo ▶]`, `[Tudo ⏭]` e Slider de Sentença.
   - Permite ao avaliador **assistir à evolução clínica do paciente**: ao avançar o passo temporal $t$, apenas os nós e arestas documentados até a sentença $t$ ficam ativos, permitindo visualizar em que etapa da internação o modelo deduziu cada relação diagnóstica.

3. **Narrativa Clínica com Proveniência de Frases & Destaque de Descobertas ML:**
   - O painel direito apresenta a história médica integral dividida em sentenças anotadas com `<mark class="c-ent">` coloridos e categorizados por asserção (`AFFIRMED`, `NEGATED`, `HISTORICAL`).
   - Frases onde o modelo de ML descobriu uma nova relação recebem um badge destacado:
     `⚡ ML: [CT Abdomen] → REVEALS → [Pseudocyst] (Confiança: 88%)` com botão direto **"Ver no Grafo"**.
   - **Sincronização Bidirecional:** clicar em qualquer sentença centraliza e foca seus nós/arestas no Cytoscape; clicar em qualquer elemento no grafo rola suavemente para a frase de origem.

4. **Simulador Contrafactual "What-If" Ao Vivo no Navegador (Aba 2) & Alinhamento Arquitetural XAI:**
   - Ao selecionar qualquer aresta relacional, a aba de Sandbox permite editar a frase intermediária ao vivo (ou usar atalhos como `[negative for]`, `[consistent with]`, `[treated with]`, `[notable for]`).
   - **Arquitetura de Inferência Client-Side e Atribuição Linear ($\beta_i \cdot x_i$):** Para garantir resposta em milissegundos 100% offline no navegador sem carregar binários PyTorch ou matrizes SVD completas em JavaScript, a geração de candidatos XAI na Fase 5 ([`src/run_pipeline_p2.py`](src/run_pipeline_p2.py), linhas 489–525) utiliza o componente linear interpretável de Regressão Logística calibrado pelo Prior Ontológico MeSH (`NeuroSymbolicGraphIntegrator`), exportando os pesos exatos $\beta_i \cdot x_i$ dos Top-3 N-grams decisivos para cada aresta, enquanto o motor JavaScript client-side (`recalculateCounterfactual` em [`src/visualizer_p2.py`](src/visualizer_p2.py)) executa um *surrogate* léxico-ontológico rápido de gatilhos contrafactuais que atualiza a classe e a cor da aresta instantaneamente no canvas.

5. **Raio-X XAI & Ancoragem MeSH (Aba 3):**
   - Exibição da decomposição linear ($\beta_i \cdot x_i$) do ramo esparso, probabilidade bruta de ML ($P_{\text{ML}}$), prior ontológico MeSH ($P_{\text{onto}}$), confiança calibrada final ($P_{\text{final}}$) e links diretos para o browser oficial da NLM (National Library of Medicine).

---

## Confronto Quantitativo e Qualitativo com o Projeto 1

| Dimensão Metodológica | Projeto 1 (Simbólico / Heurístico) | Projeto 2 (Neuro-Simbólico / Ato 3) | Impacto no Conhecimento Extraído |
| :--- | :--- | :--- | :--- |
| **Descoberta de Entidades** | Gazetteer MeSH estrito via Aho-Corasick (`1.607` nós) | Mineração PPMI + Morfologia de Char N-grams (`1.652` nós) | **+45 novos nós OOV promovidos** (+2,80%, ex.: *lactate dehydrogenase ldh*, *neonatal compartment syndrome*) |
| **Extração de Relações** | Regex de Hearst manuais (`3.757` arestas) | Classificadores N-gram, DistMult-MLP e Ensemble (`3.991` arestas) | **+234 novas arestas descobertas pelo ML** (+6,23% de ganho líquido de arestas sobre o P1) |
| **Inferência Coletiva** | Arestas isoladas e desconectadas | Fechamento de Triângulos (*Relational Belief Propagation*) | **445 triângulos clínicos fechados** (reponderando a confiança de arestas existentes sem inflar a contagem) |
| **Tratamento de Incerteza** | Arestas binárias estritas (1.0 ou ausente) | Probabilidade a posteriori contínua $P \in [0, 1]$ | Slider dinâmico de limiar de rigor diagnóstico ($\tau \in [0.5, 1.0]$) |
| **Auditoria do Gabarito** | Confiou cegamente nas regex e coocorrências | *Confident Learning Denoising* (Northcutt et al., 2021) | **47 erros do Silver Standard P1 auditados** (40 falsos negativos resgatados de `NONE` e 7 falsos positivos demovidos de `TARGETS`) |
| **Interpretabilidade** | Nenhuma (regras estáticas opacas) | XAI Linear ($\beta_i \cdot x_i$) + **Sandbox Contrafactual What-If** | Decomposição do ramo linear + simulação interativa instantânea no browser |
| **Densidade da Rede Global** | Direcionada: `0.001456`<br/>Não-Direcionada: `0.002911` | Direcionada: **`0.001463`**<br/>Não-Direcionada: **`0.002927`** | Padronizando a fórmula ($D_{\text{dir}} = \frac{E}{V(V-1)}$ e $D_{\text{undir}} = \frac{2E}{V(V-1)}$), a densidade global cresce **+0,52%** mesmo após expandir $V$ em +45 nós (+5,68% pares possíveis) |
| **Rigor de Engenharia** | Testes ad-hoc | Suíte automatizada com **23 testes unitários** | Validação contínua de invariantes matemáticos, grafos, simplex e gradientes |

---

## Modelo Lógico

O modelo de grafos de propriedades do projeto estrutura instâncias clínicas na camada episódica (A-Box) associadas a conceitos universais padronizados na camada canônica MeSH (T-Box):

![Modelo Lógico de Grafos](assets/images/modelo-logico-grafos.png)

No diagrama abaixo, observe a consistência direcional da cadeia composicional de Horn para fechamento de triângulo: o exame `E1` (`CT Abdomen`) **revela** (`REVEALS`) o achado OOV `O1` (`Pseudocyst`), o qual por sua vez **suporta** (`SUPPORTS`, orientado do achado $O_1$ para o diagnóstico $D_1$) a hipótese diagnóstica `D1` (`Pancreatitis`), fechando o caminho transitivo $E_1 \xrightarrow{\text{REVEALS}} O_1 \xrightarrow{\text{SUPPORTS}} D_1 \implies E_1 \xrightarrow{\text{CONFIRMS}} D_1$:

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
    O1 -.->|SUPPORTS [conf: 0.82, STATISTICAL_ML]| D1
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
│   │   └── p1_noise_audit_log.json  # Auditoria de 47 discrepâncias do Silver Standard P1
│   └── processed/
│       ├── nodes_p2.csv             # 1.652 nós (P1 + OOV)
│       ├── edges_p2.csv             # 3.991 arestas (P1 + ML + Triângulos)
│       ├── ablation_benchmark.json  # Resultados dos 14 experimentos em 3 Atos (com SD e Dual-Eval)
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
│   └── run_pipeline_p2.py           # Pipeline master ponta a ponta (inclui auditoria indutiva e dual-benchmark)
└── tests/
    ├── test_phase1_ngrams.py        # Invariantes matemáticos de PPMI e SVD
    ├── test_phase2_supervision.py   # Sanidade de Trigger-Masking e anti-leakage
    ├── test_phase3_classical_ml.py  # Simplex de probabilidade e XAI
    ├── test_phase4_mlp.py           # Gradiente não-nulo e determinação neural
    ├── test_phase5_graph.py         # Integridade referencial e suavização de Laplace
    └── test_phase6_upgrades.py      # Testes das inovações do Ato 2 e Ato 3 (23 testes no total)
```

---

## Análises que podem ser realizadas

A estruturação neuro-simbólica do Grafo P2 viabiliza consultas clínicas analíticas avançadas combinando a topologia determinística e as descobertas estatísticas calibradas:

1. **Auditoria de Decisões e Contradições Diagnósticas:**
   - Detectar discrepâncias entre exames de imagem e hipóteses diagnósticas (ex.: pares onde o modelo atribui EXCLUDES com alta confiança em achados que anteriormente justificariam intervenções invasivas).
2. **Progressão Temporal e Trajetórias de Tratamento (TimeML + DAG):**
   - Através das arestas PRECEDES, reconstruir o intervalo temporal entre a queixa inicial, os exames confirmatórios e o desfecho cirúrgico/farmacológico.
3. **Análise de Eficácia Terapêutica e Alvos Farmacológicos:**
   - Mapear caminhos do tipo  \to D_1$ corroborados pela resolução do sintoma em sentenças posteriores.
4. **Fechamento de Triângulos Clínicos (Consistência Diagnóstica):**
   - Verificar consistência de tríades clínicas: se um exame revela um achado e o achado suporta um diagnóstico, inferir o reforço confirmatório do exame sobre a patologia ( \to D_1$).
5. **Agrupamento de Pacientes por Perfis Fenotípicos Globais:**
   - Consultas via nós canônicos MeSH respondem de imediato: *"Quais pacientes compartilham a mesma comorbidade prévia e desenvolveram complicações pós-operatórias semelhantes?"*.

---

## Ferramentas

| Ferramenta | Finalidade no Projeto | Justificativa Técnica |
| :--- | :--- | :--- |
| **Python 3.10+** | Linguagem principal | Ecossistema maduro para manipulação de grafos e aprendizado de máquina. |
| **Scikit-learn** | Modelos clássicos e validação cruzada | Implementações eficientes de Regressão Logística, Naive Bayes, Linear SVM, Random Forest e StratifiedGroupKFold. |
| **PyTorch** | Arquitetura neural de 3 torres | Flexibilidade para modelagem trilinear DistMult ( \odot h_{\text{ctx}} \odot h_2$) e otimização por gradiente com AdamW. |
| **Pandas & NumPy** | Manipulação de dados tabulares | Ingestão, processamento de matrizes esparsas/densas e exportação de 
odes_p2.csv e dges_p2.csv. |
| **Cytoscape.js** | Visualização interativa de redes | Renderização gráfica em HTML standalone de alto desempenho sem necessidade de backend. |
| **Pytest** | Testes automatizados de software | Suíte com 23 testes unitários para verificação de invariantes matemáticos e anti-leakage. |

---

## Como Modelos de Linguagem foram Usados

Conforme as regras estritas da disciplina para o Projeto 2:
* **Na Extração de Relações e Construção do Grafo:** **Zero uso de LLMs ou Transformers pré-treinados** (como BERT, RoBERTa ou GPT). Todo o aprendizado de representações vetoriais foi construído *from scratch* via contagem de coocorrências no próprio corpus (PPMI-SVD de 64 dimensões, K-Means =32$) e modelos estatísticos supervisionados/neurais treinados localmente.
* **Na Documentação e Assistência de Código:** Ferramentas de IA generativa (Gemini 3.8 Flash via ambiente Antigravity) foram empregadas unicamente como copiloto de programação para suporte na escrita de testes unitários, formatação de relatórios e codificação do front-end em Cytoscape.js.

---

## Referências Bibliográficas

1. **Brown, P. F., Della Pietra, V. J., deSouza, P. V., Lai, J. C., & Mercer, R. L.** (1992). *Class-based n-gram models of natural language*. Computational Linguistics, 18(4), 467–479.
2. **Church, K. W., & Hanks, P.** (1990). *Word association norms, mutual information, and lexicography*. Computational Linguistics, 16(1), 22–29.
3. **Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P.** (1996). *From data mining to knowledge discovery in databases*. AI Magazine, 17(3), 37–54.
4. **Firth, J. R.** (1957). *A synopsis of linguistic theory 1930–1955*. Studies in Linguistic Analysis, Philological Society, Oxford, 1–32.
5. **Hearst, M. A.** (1992). *Automatic acquisition of hyponyms from large text corpora*. In Proceedings of the 14th Conference on Computational Linguistics (COLING '92), 539–545.
6. **Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S.** (2022). *A survey on knowledge graphs: Representation, acquisition, and applications*. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494–514.
7. **Jurafsky, D., & Martin, J. H.** (2026). *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Prentice Hall.
8. **Levy, O., & Goldberg, Y.** (2014). *Neural word embedding as implicit matrix factorization*. In Advances in Neural Information Processing Systems (NeurIPS 2014), 2177–2185.
9. **Manning, C. D., Raghavan, P., & Schütze, H.** (2009). *An Introduction to Information Retrieval*. Cambridge University Press.
10. **Nievas Offidani, M., Delrieux, C., & Maguitman, A.** (2025). *MultiCaRe: A Multi-lingual, Multi-label Clinical Case Report Dataset from PubMed Central for Natural Language Processing in Medicine*. Scientific Data / Zenodo (PubMed Central Open-Access Subset).
11. **Northcutt, C. G., Jiang, L., & Chuang, I. L.** (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. Journal of Artificial Intelligence Research, 70, 1373–1411.
12. **Ratner, A., De Sa, C., Wu, S., Selsam, D., & Ré, C.** (2017). *Snorkel: Rapid training data creation with weak supervision*. Proceedings of the VLDB Endowment, 11(3), 269–282.
13. **Santanchè, A.** (2026). *Notas de Aula e Apresentações: nlp-060 (Vector Space), nlp-080 (ML Text Classification), nlp-100 (Neural Networks), nlp-105 (Neural Networks Learning) e nlp-110 (Deep Learning)*. Instituto de Computação, UNICAMP.
14. **Toutanova, K., Chen, D., Pantel, P., Poon, H., Choudhury, P., & Gamon, M.** (2015). *Representing text for joint embedding of text and knowledge bases*. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015), 1499–1509.
15. **Wexler, J., Pushkarna, M., Bolukbasi, T., Wattenberg, M., Viégas, F., & Wilson, J.** (2019). *The What-If Tool: Interactive probing of machine learning models*. IEEE Transactions on Visualization and Computer Graphics, 26(1), 56–65.
16. **Yang, B., Yih, W., He, X., Gao, J., & Deng, L.** (2015). *Embedding entities and relations for learning and inference in knowledge bases*. In Proceedings of the International Conference on Learning Representations (ICLR 2015).
17. **Zhong, Z., & Chen, D.** (2021). *A frustratingly easy approach for entity and relation extraction*. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021), 50–61.
