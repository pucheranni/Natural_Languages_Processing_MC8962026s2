# Contexto Teórico, Legado do Projeto 1 e Brainstorm de Inovação para o Projeto 2
> **Disciplina:** MC896 — Processamento de Línguas Naturais (UNICAMP / 2026.2)  
> **Professor:** André Santanchè  
> **Equipe:** Caio (248243), Fábio Luiz (247079), Gustavo Sousa (247149), T.H. de Camargo J. (206235), Wallysson (200061)

---

## 1. Sumário Executivo e Metas do Projeto 2

### 1.1. Contexto e Transição P1 → P2
No **Projeto 1**, a equipe desenvolveu o **Extrator-clínico-inator**, um pipeline determinístico baseado em regras simbólicas, autômato Aho-Corasick com gazetteers MeSH, expressões regulares de dosagens e padrões sintáticos de Hearst. O resultado foi um grafo clínico consistente em duas camadas (episódica e ontológica) com 1.607 nós e 3.757 arestas em 56 casos clínicos do corpus MultiCaRe.

No **Projeto 2**, o desafio é **substituir ou aprimorar o determinismo estrito por métodos estatísticos e de aprendizado de máquina**. A proposta central é investigar até que ponto técnicas de aprendizado a partir dos próprios textos conseguem:
1. Superar o baixo recall das regras manuais de Hearst;
2. Generalizar para padrões de redação médica não previstos;
3. Descobrir termos clínicos compostos e novas relações implícitas;
4. Reduzir o ruído de falsos positivos da coocorrência ingênua.

### 1.2. Restrições Rígidas da Disciplina
* ❌ **Proibição Absoluta:** LLMs (GPT, Gemini, LLaMA, Claude) e modelos baseados em Transformers pré-treinados (BERT, BioBERT, ClinicalBERT, RoBERTa, etc.).
* ❌ **Sem Embeddings Pré-treinados Externos:** Não usar embeddings estáticos prontos de terceiros; qualquer representação densa/neural deve ser treinada do zero no corpus ou sobre os n-grams/vetores do projeto.
* ✅ **Obrigatório:** Presença de **N-grams** em pelo menos uma parte substantiva do pipeline (na identificação de entidades compostas, extração de relações ou representação de contexto).
* ✅ **Métodos Permitidos:** N-grams, TF-IDF, PMI/PPMI, Naive Bayes, Regressão Logística, SVM, Random Forest, CRF, Redes Neurais Feedforward (MLP), RNN, LSTM, BiLSTM e Supervisão Fraca (Weak Supervision).
* 🎯 **Critérios de Diferenciação Valorizados:** Criatividade, audácia científica (formular hipóteses desafiadoras e testá-las), comparação experimental rigorosa (ablation study) e recursos interativos de visualização do grafo.

---

## 2. Base de Conhecimento dos Slides de Aula (Prof. André Santanchè)

Esta seção mapeia os conceitos formais apresentados nas aulas e em seus slides de apoio, garantindo alinhamento conceitual e terminológico exato com a avaliação do professor.

```
                    MAPA CONCEITUAL DAS AULAS
┌─────────────────────────────────────────────────────────────────┐
│ Slide 060: Modelo de Espaço Vetorial, TF-IDF & Semântica        │
│   • TF, DF, IDF, TF-IDF Weighting                               │
│   • Distância Euclidiana vs. Similaridade Cosseno               │
│   • Matrizes Termo-Documento e Coocorrência Termo-Termo        │
│   • Vector Semantics: "You shall know a word by the company it  │
│     keeps" (Firth, 1957)                                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Slide 080: Aprendizado de Máquina & Classificação de Textos    │
│   • Knowledge Discovery in Databases (KDD - Fayyad et al. 1996) │
│   • Regressão Linear → Regressão Logística (Sigmoide, Log-Odds) │
│   • Classificador Naive Bayes (Laplace Smoothing, Priors)       │
│   • Engenharia de Atributos para Texto (Bag of Words, N-grams) │
│   • Avaliação: Precision, Recall, F1 (Macro/Micro), K-Fold CV   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Slides 100 & 105: Otimização & Redes Neurais                   │
│   • Da Regressão Logística ao Neurônio Artificial               │
│   • Cross-Entropy Loss / Negative Log-Likelihood                │
│   • Gradiente Descendente (Learning Rate η, Backpropagation)    │
│   • Mini-batch SGD e Superfícies de Perda                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Slide 110: Deep Learning & Redes Neurais Feedforward           │
│   • Multi-Layer Perceptron (MLP) de 2 Camadas (Jurafsky, 2026)  │
│   • Funções de Ativação: Sigmoide, Tanh, ReLU                   │
│   • Camada de Saída Softmax para Multiclasse                    │
│   • Regularização: Dropout, L2 Weight Decay                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1. Slide `nlp-060`: Espaço Vetorial e Recuperação Ranqueada
* **Term Frequency (TF):** A frequência bruta $tf_{t,d}$ superestima a importância de palavras repetidas. Adota-se comumente a escala logarítmica:
  $$w_{t,d} = 1 + \log_{10}(tf_{t,d}) \quad \text{se } tf_{t,d} > 0, \text{ senão } 0$$
* **Inverse Document Frequency (IDF):** Mede a especificidade de um termo no corpus de $N$ documentos:
  $$idf_t = \log_{10}\left(\frac{N}{df_t}\right)$$
* **Ponderação TF-IDF:** Combina relevância local no documento com discriminação global:
  $$\text{tf-idf}_{t,d} = tf_{t,d} \times idf_t$$
* **Espaço Vetorial e Similaridade Cosseno:** Documentos e queries tornam-se vetores em espaço $\mathbb{R}^{|V|}$. A proximidade semântica independe do comprimento do texto ao utilizar a normalização cosseno:
  $$\cos(\vec{x}, \vec{y}) = \frac{\vec{x} \cdot \vec{y}}{\|\vec{x}\| \|\vec{y}\|} = \frac{\sum_{i=1}^V x_i y_i}{\sqrt{\sum_{i=1}^V x_i^2} \sqrt{\sum_{i=1}^V y_i^2}}$$
* **Vector Semantics & Coocorrência:** Hipótese distributiva de Firth: palavras com distribuições estatísticas semelhantes compartilham significado semântico. Base para embeddings por contagem esparsa e Pointwise Mutual Information (PMI).

### 2.2. Slide `nlp-080`: Aprendizado de Máquina e Classificação de Texto
* **Processo KDD (Fayyad et al., 1996):**
  1. *Selection:* Seleção das instâncias de texto e pares de entidades.
  2. *Preprocessing:* Limpeza, tokenização, segmentação sentencial clínica.
  3. *Transformation:* Extração de N-grams, vetorização TF-IDF, balanceamento de classes.
  4. *Data Mining / Machine Learning:* Treinamento de classificadores (Naive Bayes, Regressão Logística, SVM, Random Forest).
  5. *Interpretation / Evaluation:* Análise de métricas, interpretabilidade de coeficientes e inspeção de arestas.
* **Regressão Logística:**
  - Função sigmoide:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \vec{w} \cdot \vec{x} + b$$
  - Probabilidade de classe de Bernoulli: $P(y=1 \mid \vec{x}) = \sigma(z)$ e $P(y=0 \mid \vec{x}) = 1 - \sigma(z)$.
  - Limiar de decisão operacional: $\tau = 0.5$ (ou ajustável para privilegiar Precision ou Recall).
  - Razão de chances (*Odds Ratio*) e Logit: $\ln\left(\frac{p}{1-p}\right) = \vec{w} \cdot \vec{x} + b$.
* **Naive Bayes Multinomial:**
  - Teorema de Bayes com suposição ingênua de independência condicional dos atributos dada a classe:
    $$\hat{c} = \arg\max_{c \in C} P(c) \prod_{i=1}^n P(w_i \mid c)$$
  - Laplace Smoothing (add-1) para evitar probabilidades nulas:
    $$P(w \mid c) = \frac{\text{count}(w, c) + 1}{\sum_{w' \in V} (\text{count}(w', c) + 1)}$$
* **Métricas Formais de Avaliação:**
  - Matriz de Confusão: Verdadeiros Positivos (TP), Falsos Positivos (FP), Falsos Negativos (FN), Verdadeiros Negativos (TN).
  - Precision: $\frac{TP}{TP + FP}$ (fração de relações extraídas que são clinicamente corretas).
  - Recall: $\frac{TP}{TP + FN}$ (fração de relações reais presentes no texto que o sistema capturou).
  - F1-Score: Média harmônica $2 \cdot \frac{P \cdot R}{P + R}$.
  - Macro-F1 (média simples do F1 de cada relação, tratando classes minoritárias com igual peso) vs. Micro-F1 (ponderada pelo volume total).
  - Validação Cruzada Estratificada ($K$-Fold Cross-Validation, $K=5$).

### 2.3. Slides `nlp-100` e `nlp-105`: Otimização e Treinamento Neural
* **Loss Function (Cross-Entropy Loss):**
  $$L_{CE}(\hat{y}, y) = - [y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$$
* **Gradiente Descendente e Atualização de Pesos:**
  $$w_j \leftarrow w_j - \eta \frac{\partial L}{\partial w_j}$$
  onde $\eta$ é a taxa de aprendizado (*learning rate*).
* **Retropropagação (Backpropagation):** Aplicação sistemática da regra da cadeia para calcular gradientes das camadas ocultas em relação à função de perda.

### 2.4. Slide `nlp-110`: Redes Neurais Feedforward (Deep Learning Clássico)
* **Arquitetura Multi-Layer Perceptron (MLP) de 2 Camadas:**
  - Entrada $\vec{x} \in \mathbb{R}^d$ (vetor TF-IDF de n-grams ou embeddings agregados).
  - Camada Oculta: $\vec{h} = g(W_1 \vec{x} + \vec{b}_1)$, com função de ativação não-linear $g(\cdot)$ como ReLU ($g(z) = \max(0, z)$) ou Tanh.
  - Camada de Saída Multiclasse: $\vec{z} = W_2 \vec{h} + \vec{b}_2$.
  - Softmax para distribuição de probabilidade sobre as $K$ relações possíveis:
    $$P(\text{rel} = k \mid \vec{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
* **Regularização:** Dropout (desativação estocástica de neurônios no treino para evitar overfitting) e decaimento de pesos L2 (*Weight Decay*).

---

## 3. Radiografia Completa do Projeto 1 (A Nossa Linha de Base)

Para planejar experimentos de confronto direto, é essencial documentar exatamente o que foi entregue no Projeto 1:

### 3.1. Dados de Entrada
* **Dataset MultiCaRe:** 56 relatos de casos clínicos (`cases.csv`) em inglês, contendo histórico de sintomas, exames laboratoriais, achados de imagem, procedimentos cirúrgicos e desfechos.
* **Metadados:** `metadata.csv` (contendo termos MeSH indexados manualmente por documentalistas para cada artigo) e `data_dictionary.csv`.

### 3.2. Módulos Implementados em `project1/src/`
1. `preprocessor.py`: Limpeza de texto, proteção de decimais (`850.5 U/L`, `0.8 cm`), unidades e abreviações médicas (`POD 7`, `Fig. 1`) para evitar quebras errôneas de sentença.
2. `entity_matcher.py`: Autômato Aho-Corasick para casamento em tempo $O(N)$ sobre vocabulários MeSH (Categorias C - Doenças, D - Fármacos/Químicos, E01 - Procedimentos Diagnósticos/Exames, A - Anatomia). Filtro determinístico de *Longest-Match* e limite estrito de palavras (`\b`).
3. `assertion.py`: Implementação do algoritmo NegEx com janelas direcionais e barreira de conjunções adversativas (`but`, `however`, `although`, `except`). Rótulos: `AFFIRMED`, `NEGATED`, `HISTORICAL`, `HYPOTHETICAL`.
4. `quant_extractor.py`: Gramática farmacológica tripartite com regex para capturar dosagens diretas, invertidas partitivas (*"50 mg of diltiazem intravenously"*) e formulações com via de administração e taxa contínua (`mL/h`).
5. `relation_builder.py`: Regras de associação:
   - Conexão do paciente com observações (`PRESENTS_WITH`, `DENIES_OR_ABSENT`, `UNDERWENT_TEST`, `DIAGNOSED_WITH`, `TREATED_WITH`).
   - Padrões léxico-sintáticos de Hearst entre exames e achados (`REVEALS`, `CONFIRMS`, `EXCLUDES`).
   - Coocorrência intra-sentença entre achados/exames e diagnóstico (`SUPPORTS`).
   - Coocorrência intra-sentença entre tratamento e diagnóstico (`TARGETS`).
   - Coocorrência anatômica (`LOCATED_IN`).
   - Arestas ontológicas MeSH (`INSTANCE_OF`).
6. `timeline_dag.py`: Extração de marcadores temporais relativos (*"3-day history"*, *"POD 7"*) e construção do DAG cronológico com arestas `PRECEDES`.
7. `graph_builder.py`: Orquestrador do grafo em duas tabelas relacionais (`nodes.csv` e `edges.csv`).
8. `app_visualizer.py`: Renderizador interativo HTML/CSS/JS standalone utilizando Cytoscape.js.
9. `evaluate_mesh.py`: Script de validação quantitativa contra os termos MeSH de `metadata.csv`.

### 3.3. Métricas e Resultados Alcançados no P1
* **Casos Processados:** 56 casos clínicos.
* **Nós Extraídos (`nodes.csv`):** 1.607 nós.
* **Arestas Geradas (`edges.csv`):** 3.757 arestas.
* **Integridade Referencial:** 100% (zero nós órfãos de arestas).
* **Benchmark de Entidades contra MeSH Ground-Truth (`evaluate_mesh.py`):**
  - *Baseline Ingênua (Substrings sem filtros):* Precision = 0.3193, Recall = 0.4212, F1 = 0.3348.
  - *Pipeline SOTA P1 (Gazetteers filtrados):* Precision = 0.2314, Recall = 0.3753, F1 = 0.2230.
  *(A precisão aparente menor ocorreu porque o gazetteer MeSH é estrito e os metadados humanos catalogam apenas termos de alto nível do artigo).*

### 3.4. Gargalos e Oportunidades Identificados no P1
| Ponto Fraco no P1 | Causa Técnica | Como o Projeto 2 pode resolver com ML / N-grams |
| :--- | :--- | :--- |
| **Baixo Recall em Relações Clínicas** | As regras de Hearst exigem verbos literais exatos (`revealed`, `confirmed`). Variações frasais são ignoradas. | Classificadores baseados em N-grams (TF-IDF de 1-3 grams) aprendem que padrões como *"noted on examination"* ou *"findings consistent with"* também expressam `REVEALS`/`CONFIRMS`. |
| **Falsos Positivos em Coocorrência Intra-Sentença** | Se um exame e um diagnóstico aparecem na mesma oração, o P1 gerava `SUPPORTS` cegamente. | Classificador de ML analisa o contexto intermediário e decide se há relação ou se deve classificar como `NONE`. |
| **Vocabulário Fechado de Entidades** | Termos compostos não indexados no MeSH são ignorados ou quebrados. | Análise estatística de collocations por PMI identifica bigramas e trigramas clinicamente coesos diretamente do corpus. |
| **Arestas Binárias sem Incerteza** | Todas as arestas têm o mesmo peso (1.0), sem distinção entre evidência explícita e implícita. | Modelos probabilísticos (Regressão Logística / MLP) atribuem um *score* de confiança contínuo $P \in [0, 1]$ a cada aresta. |

---

## 4. O Guia de Ideias do Professor (N-Grams & Saúde)

O professor forneceu diretrizes específicas de como os N-grams devem ser interpretados sob a ótica de conhecimento clínico:

### 4.1. Conhecimento Representado por N-Grams no Contexto Clínico
1. **Sintaxe / Gramática Clínica:**
   - $P(\text{with} \mid \text{treated}) > P(\text{for} \mid \text{treated})$ → Padrão sintático do verbo *treated with*.
   - $P(\text{by} \mid \text{caused}) > P(\text{with} \mid \text{caused})$ → Padrão causal.
   - $P(\text{to} \mid \text{secondary}) > P(\text{of} \mid \text{secondary})$ → Padrão etiológico *secondary to*.
2. **Semântica e Significado Clínico:**
   - $P(\text{infarction} \mid \text{myocardial}) \gg P(\text{table} \mid \text{myocardial})$ → Expressão com forte coerência léxico-semântica.
   - $P(\text{mellitus} \mid \text{diabetes}) \gg P(\text{fever} \mid \text{diabetes})$.
3. **Entidades Clínicas Compostas:**
   - $P(\text{infarction} \mid \text{myocardial}) \gg P(\text{infarction} \mid \text{acute})$ → Componente nuclear vs. modificador temporal.
4. **Relações Clínicas:**
   - Padrões como *X secondary to Y* indicam relação causal/etiológica `CAUSED_BY`.
   - Padrões como *X treated with Y* indicam `TREATED_WITH`.
   - Padrões como *X located in Y* indicam localização anatômica `LOCATED_IN`.
5. **Conhecimento sobre o Mundo Clínico:**
   - $P(\text{metastasis} \mid \text{cancer}) > P(\text{metastasis} \mid \text{unrelated})$.
   - $P(\text{chemotherapy} \mid \text{cancer}) > P(\text{chemotherapy} \mid \text{unrelated})$.

---

## 5. Brainstorm Detalhado de Ideias Inovadoras para o P2

Abaixo estão as 4 frentes de inovação propostas para atingir a pontuação máxima e os diferenciais de audácia.

```
                      ARQUITETURA DAS 4 FRENTES DE INOVAÇÃO
┌───────────────────────────────────────────────────────────────────────────────┐
│ FRENTE 1: Motor Estatístico de N-Grams e Collocations                         │
│ • Mineração de Bigramas e Trigramas via PMI (Pointwise Mutual Information)    │
│ • Detecção de Entidades Compostas Inéditas fora dos Dicionários MeSH          │
│ • Análise de N-grams de Caracteres para Morfologia Clínica (-itis, -ectomy)   │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FRENTE 2: Supervisão Fraca (Weak Supervision) e Silver Standard               │
│ • Extração de Pares Candidatos (Entidade_A, Entidade_B, Contexto)             │
│ • Uso das Regras do P1 como Labeling Functions (LFs)                          │
│ • Geração de Amostras Negativas (Classe NONE) via Distância Intra/Inter-frase │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FRENTE 3: Torneio de Representações e Modelos de Machine Learning             │
│ • Ablation Study: 1-gram vs. (1+2)-gram vs. (1+2+3)-gram; Count vs. TF-IDF    │
│ • Modelos Lineares: Naive Bayes Multinomial e Regressão Logística L2          │
│ • Modelos de Margem & Ensembles: Linear SVM e Random Forest                   │
│ • Deep Learning Clássico: MLP Neural 2-Layer (PyTorch/Sklearn, slides 110)    │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FRENTE 4: Descoberta de Conhecimento, Grafo Aprimorado & Visualização         │
│ • Predição de Novas Arestas Implícitas com Score de Probabilidade P ∈ [0, 1]  │
│ • Cytoscape.js Aprimorado com Slider Interativo de Confiança                  │
│ • Comparação Topológica Formal: P1 (Simbólico) vs. P2 (Estatístico/ML)        │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

### Frente 1: Motor Estatístico de N-Grams e Descoberta de Entidades (Collocations & PMI)

#### Conceito Matemático
A *Pointwise Mutual Information* (PMI) quantifica a discrepância entre a probabilidade conjunta de duas palavras e suas probabilidades individuais sob a hipótese de independência estatística:
$$\text{PMI}(w_1, w_2) = \log_2 \left(\frac{P(w_1, w_2)}{P(w_1)P(w_2)}\right) = \log_2 \left(\frac{C(w_1, w_2) \cdot N}{C(w_1)C(w_2)}\right)$$
Para evitar valores infinitos com frequências muito baixas, adota-se o corte por contagem mínima ($C(w_1, w_2) \ge 3$) e a versão positiva:
$$\text{PPMI}(w_1, w_2) = \max(0, \text{PMI}(w_1, w_2))$$

#### Aplicações Práticas no Projeto
1. **Identificação de Expressões Médicas Compostas:**
   - Varrer todo o corpus MultiCaRe e extrair os bi-gramas e tri-gramas com maior PPMI e *Student's t-test* (ex.: `pleural effusion`, `biliary tract`, `computed tomography`, `squamous cell carcinoma`).
   - Comparar os N-grams descobertos estatisticamente com o gazetteer do P1: demonstrar que o motor de N-grams é capaz de identificar expressões médicas especializadas **sem necessitar de ontologia prévia**.
2. **N-Grams de Caracteres para Morfologia Biomédica (Diferencial Audacioso):**
   - Treinar um classificador linear simples que recebe subpalavras de 3 a 5 caracteres (ex.: `*itis`, `*ectomy`, `*emia`, `*pathy`, `*scopy`, `*megaly`).
   - Avaliar a capacidade de prever o tipo semântico de uma entidade desconhecida (`Diagnosis` vs. `ProcedureInstance` vs. `Finding`).

---

### Frente 2: Supervisão Fraca (Weak Supervision) para Geração do Dataset de Relações

#### O Desafio da Anotação
No corpus MultiCaRe, temos o texto completo dos 56 casos, mas não possuímos anotação manual par-a-par de relações para treinar classificadores supervisionados.

#### A Solução Elegante: Paradigma Snorkel / Silver Standard
1. **Extração de Pares Candidatos:**
   - Em cada caso clínico, identificar todas as entidades reconhecidas e extrair todos os pares $(E_1, E_2)$ que ocorrem na mesma sentença ou em sentenças adjacentes.
2. **Rotulagem Fraca (*Silver Labels*) via Regras do P1:**
   - Se o par casar com um padrão de Hearst ou regra ontológica do P1, recebe a classe positiva correspondente (`REVEALS`, `CONFIRMS`, `EXCLUDES`, `SUPPORTS`, `TREATED_WITH`, `LOCATED_IN`, `TARGETS`).
3. **Geração Automática de Exemplos Negativos (`NONE`):**
   - Pares de entidades que aparecem no mesmo parágrafo mas separados por barreiras gramaticais fortes (ex.: pontuação de ponto final duplo, orações independentes sem nexo causal ou distâncias superiores a 20 tokens) recebem o rótulo `NONE`.
4. **Construção do Dataset Estruturado (`relation_dataset.csv`):**
   - `pair_id`: Identificador único.
   - `case_id`: ID do caso clínico.
   - `entity1_text`, `entity1_type`: Texto e tipo da primeira entidade.
   - `entity2_text`, `entity2_type`: Texto e tipo da segunda entidade.
   - `context_between`: Sequência de palavras estritamente entre $E_1$ e $E_2$.
   - `context_window`: Janela com 3 tokens antes de $E_1$ + contexto intermediário + 3 tokens após $E_2$.
   - `distance_tokens`: Distância em palavras entre as entidades.
   - `silver_label`: Relação atribuída pela regra ou `NONE`.

---

### Frente 3: O "Torneio de Modelos" e Estudo de Ablação (Ablation Study)

O coração experimental do Projeto 2 será um confronto sistemático entre diferentes representações e arquiteturas de ML.

#### Experimento 1: Comparação de Granularidade Textual (Ablation de N-Grams)
Fixando-se um classificador linear (Regressão Logística L2), avaliar o impacto da ordem dos N-grams na representação do contexto:
* **Configuração A (Unigrams):** Bag-of-Words tradicional ($1$-gram).
* **Configuração B (Unigrams + Bigrams):** $1$-gram + $2$-gram (captura locuções como *"showed no"*, *"treated with"*, *"secondary to"*).
* **Configuração C (Unigrams + Bigrams + Trigrams):** $1$-gram + $2$-gram + $3$-gram (captura padrões completos como *"was treated with"*, *"revealed no evidence"*).
* **Configuração D (TF-IDF vs. Contagem vs. PMI Weighting):** Investigar se a ponderação por relevância global (IDF) supera a frequência bruta na classificação de relações.

#### Experimento 2: Torneio de Algoritmos de Aprendizado de Máquina
Treinar e avaliar sob a mesma partição de dados (5-Fold Stratified Cross-Validation):
1. **Baseline Simbólica (Projeto 1):** Regras de regex de Hearst puras.
2. **Modelo Probabilístico:** Naive Bayes Multinomial com Laplace smoothing.
3. **Modelo Linear com Regularização:** Regressão Logística com penalidade L2 (Ridge) e L1 (Lasso para seleção de atributos esparsos).
4. **Modelo de Margem Máxima:** Linear Support Vector Machine (LinearSVC) com calibração Platt para geração de probabilidades.
5. **Modelo Não-Linear / Ensemble:** Random Forest Classifier (100 árvores de decisão).
6. **Rede Neural Clássica (Deep Learning sem Transformers - Slide 110):**
   - Multi-Layer Perceptron (MLP) com arquitetura: Entrada (vetor TF-IDF de dimensão $D$) → Camada Densa 1 (128 neurônios, ativação ReLU, Dropout 0.3) → Camada Densa 2 (64 neurônios, ReLU) → Camada de Saída (Softmax com $K$ classes).
   - Otimizador: Adam ou SGD com Momentum, taxa de aprendizado $\eta = 0.001$, perda Cross-Entropy.

#### Experimento 3: Interpretabilidade dos Coeficientes (Explicabilidade Clínica)
Um ponto muito elogiado pelo professor é a capacidade de explicar as decisões do modelo:
* Na Regressão Logística multiclasse, extrair os N-grams associados aos **maiores coeficientes positivos ($\beta$)** para cada classe de relação.
* **Exemplo de Verificação de Hipótese:**
  - Para a classe `CAUSED_BY`: os N-grams de maior peso devem ser *"secondary to"*, *"due to"*, *"caused by"*, *"result of"*.
  - Para a classe `TREATED_WITH`: *"prescribed"*, *"administered"*, *"started on"*, *"resection"*.
  - Isso demonstra que o modelo **redescobriu estatisticamente** os padrões que no P1 foram programados manualmente!

---

### Frente 4: Grafo Ponderado por Confiança e Visualização Diferenciada

#### Resolução de Incerteza e Descoberta de Novas Arestas
1. **Atribuição de Probabilidade às Arestas:**
   - Toda aresta gerada por ML recebe o atributo `confidence: float = P(relação | contexto)`.
   - As arestas originais do P1 recebem `confidence: 1.0` (origem determinística).
2. **Descoberta de Arestas Ocultas:**
   - Em pares de entidades onde o P1 não detectou relação (por ausência de verbo de Hearst exato), se o modelo de ML classificar uma relação com confiança $P \ge \tau$ (ex.: $\tau = 0.70$), essa aresta é inserida no grafo como **descoberta estatística**.
3. **Aprimoramento do Visualizador Cytoscape.js:**
   - Adicionar uma barra deslizante interativa (*slider*) de **Limiar de Confiança ($\tau \in [0.5, 1.0]$)** na interface HTML. O usuário pode filtrar o grafo em tempo real para visualizar desde uma rede densa e exploratória até uma rede ultraestrita de alta certeza.
   - **Estilização Visual por Fonte de Conhecimento:**
     - Arestas **Azuis Sólidas**: Extraídas por regras determinísticas do P1.
     - Arestas **Verdes Tracejadas**: Novas relações descobertas estatisticamente pelo modelo de ML do P2.
     - Espessura da aresta proporcional ao score de probabilidade do classificador.

---

## 6. Confronto Experimental Sistemático (P1 vs. P2)

Para documentar no `README.md` da entrega e nos slides da apresentação, propomos duas tabelas comparativas formais:

### 6.1. Comparação de Desempenho Preditivo de Relações
*(Resultados obtidos via 5-Fold Stratified Cross-Validation)*

| Abordagem / Modelo | Representação de Contexto | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Suporte de Generalização |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **P1: Regras Simbólicas** | Expressões Regulares de Hearst | *Alto nas regras* | *Baixo (rígido)* | *Moderado* | Nulo (falha em paráfrases) |
| **P2: Naive Bayes** | Unigrams (TF-IDF) | *A ser medido* | *A ser medido* | *A ser medido* | Baseline probabilística rápida |
| **P2: Regressão Logística** | Unigrams + Bigrams (TF-IDF) | *A ser medido* | *A ser medido* | *A ser medido* | Excelente interpretabilidade |
| **P2: Linear SVM** | Unigrams + Bigrams + Trigrams | *A ser medido* | *A ser medido* | *A ser medido* | Robusto em alta dimensionalidade |
| **P2: Random Forest** | N-grams + Features Estruturais | *A ser medido* | *A ser medido* | *A ser medido* | Não-linear com seleção de features |
| **P2: Neural MLP (2-Layer)** | N-grams TF-IDF Densa | *A ser medido* | *A ser medido* | *A ser medido* | Capacidade de abstração profunda |

### 6.2. Comparação Estrutural e Topológica dos Grafos Resultantes

| Métrica do Grafo de Conhecimento | Projeto 1 (Simbólico) | Projeto 2 (Híbrido Simbólico + ML) | Variação (%) | Interpretação Clínica |
| :--- | :---: | :---: | :---: | :--- |
| **Total de Nós** | 1.607 | 1.607 + (Entidades PMI) | $+ \Delta\%$ | Incorporação de novos conceitos compostos |
| **Total de Arestas** | 3.757 | 3.757 + (Novas Arestas ML) | $+ \Delta\%$ | Maior densidade relacional do histórico |
| **Densidade da Rede** | $\approx 0.0014$ | *A ser calculado* | $+ \Delta\%$ | Grafo mais coeso entre sintomas e tratamentos |
| **Nós Órfãos (Grau = 0)** | 0 | 0 | $0\%$ | Preservação da integridade referencial |
| **Componentes Conexos** | 56 clusters | 56 clusters | $0$ | Preservação da separação por paciente |
| **Taxa de Arestas com Incerteza** | 0% (todas binárias) | 100% calibradas ($P \in [0, 1]$) | N/A | Permite filtragem analítica parametrizável |

---

## 7. Estrutura de Pastas Padronizada (Cookiecutter Data Science)

Seguindo rigorosamente o template exigido pelo professor para a pasta `project2`:

```
project2/
├── README.md                          <- Relatório formal da Entrega 2 do projeto
├── brainstorm_contexto_plano.md       <- Este documento de planejamento e síntese
│
├── data/
│   ├── raw/                           <- Cópias ou links dos dados originais (cases.csv, metadata.csv)
│   ├── interim/                       <- relation_dataset.csv, collocations_pmi.csv, silver_labels.csv
│   └── processed/                     <- nodes_p2.csv, edges_p2.csv, benchmark_comparison.json
│
├── pipelines/
│   ├── notebooks/
│   │   ├── 01_ngram_collocations_pmi.ipynb     <- Descoberta estatística de N-grams e PMI
│   │   ├── 02_weak_supervision_dataset.ipynb   <- Geração de pares e rótulos fracos a partir do P1
│   │   ├── 03_model_tournament_ablation.ipynb  <- Torneio de ML (NB, LR, SVM, MLP) e curvas ROC/F1
│   │   └── 04_graph_enhancement_metrics.ipynb  <- Reconstrução do grafo P2 e métricas topológicas
│   └── workflows/
│
├── src/
│   ├── README.md                      <- Instruções de execução e dependências
│   ├── ngram_miner.py                 <- Classes para N-grams, PMI, PPMI e N-grams de caracteres
│   ├── weak_supervisor.py             <- Extrator de pares contextuais e gerador de silver standard
│   ├── relation_classifier.py         <- Pipeline Scikit-Learn e modelo PyTorch MLP 2-Layer
│   ├── graph_integrator.py            <- Fusor de arestas simbólicas e probabilísticas
│   └── visualizer_p2.py               <- Adaptador Cytoscape.js com slider de confiança
│
└── assets/
    ├── images/                        <- Gráficos de ablação, matrizes de confusão e capturas do grafo
    └── slides/                        <- Apresentação em PDF da Parte 2
```

---

## 8. Divisão Estratégica de Tarefas para a Equipe (5 Integrantes)

Para maximizar a produtividade e permitir trabalho paralelo sem gargalos de dependência:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EQUIPE E SUAS FRENTES                              │
├─────────────────────┬───────────────────────────────────────────────────────┤
│ Integrante          │ Responsabilidade Principal                            │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ 1. Gustavo Sousa    │ • Coordenação de integração e supervisão fraca        │
│                     │ • Construção do dataset de pares e silver labels      │
│                     │ • Orquestração do pipeline de dados (data/interim)    │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ 2. Fábio Luiz       │ • Motor de N-grams e Collocations estatísticas        │
│                     │ • Implementação das métricas de PMI, PPMI e t-test    │
│                     │ • Análise de expressões compostas e N-grams de chars  │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ 3. Caio             │ • Torneio de Modelos Clássicos de ML                  │
│                     │ • Implementação de Naive Bayes, Logistic Regression,  │
│                     │   Linear SVM e Random Forest                          │
│                     │ • Estudo de ablação (1-gram vs 2-gram vs 3-gram)      │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ 4. Wallysson        │ • Modelagem Neural Clássica (Deep Learning sem Transf)│
│                     │ • Implementação do MLP 2-Layer (PyTorch/Sklearn)      │
│                     │ • Curvas de aprendizado (Cross-Entropy), Dropout, L2  │
├─────────────────────┼───────────────────────────────────────────────────────┤
│ 5. T.H. de Camargo  │ • Integração do Grafo P2 e Métricas Topológicas       │
│                     │ • Atualização do visualizador Cytoscape com slider    │
│                     │ • Redação do README.md e confecção dos slides         │
└─────────────────────┴───────────────────────────────────────────────────────┘
```

---

## 9. Próximos Passos Imediatos Sugeridos

1. **Validação do Plano:** Revisar as 4 frentes e aprovar o escopo das entregas com o grupo.
2. **Setup da Estrutura:** Criar as pastas em `project2/` (`data/`, `pipelines/notebooks/`, `src/`, `assets/`).
3. **Execução do Bloco 1 (Dados & N-Grams):**
   - Rodar o script de collocations e PMI para listar os 50 termos compostos mais fortes do MultiCaRe.
   - Gerar o arquivo `data/interim/relation_dataset.csv` a partir das extrações do Projeto 1.
4. **Execução do Bloco 2 (Modelagem & Torneio):**
   - Treinar os primeiros modelos lineares de Regressão Logística e avaliar o ganho de Recall e F1.
