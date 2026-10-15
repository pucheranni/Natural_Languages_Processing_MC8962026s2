# Extrator Clínico Neuro-Simbólico (Projeto 2) - Código Fonte (`src/`)

Este diretório contém os módulos da arquitetura estatística e de aprendizado de máquina para extração de grafos de conhecimento clínico, sem o uso de LLMs ou modelos de linguagem baseados em Transformers.

## 📁 Estrutura dos Módulos

* **`ngram_miner.py`:**
  * `CollocationMiner`: Mineração de bigramas e trigramas com alta *Positive Pointwise Mutual Information* (PPMI) e significância estatística (*Student's t-test*).
  * `MorphologyOOVClassifier`: Classificador de morfologia de subpalavras (N-grams de 3 a 5 caracteres com `char_wb`) treinado sobre termos MeSH conhecidos para predizer a categoria ontológica de entidades *Out-of-Vocabulary*.
  * `DistributionalSVD`: Implementação do Teorema de Levy & Goldberg (2014) para gerar embeddings densos de 64 dimensões do zero a partir da fatoração por Truncated SVD da matriz de coocorrência PPMI.
* **`weak_supervisor.py`:**
  * `PositionalWeakSupervisor`: Extração de N-grams em zonas de contexto (`PRE_`, `BTW_`, `POST_`), direcionalidade (`DIR_L2R`, `DIR_R2L`) e mecanismo de *Trigger-Masking* (substituição de verbos literais de Hearst por `[TRIGGER_MASK]` com probabilidade $p=0.85$ durante o treino).
  * `WeakSupervisionDatasetGenerator`: Construção do dataset *Silver Standard* (`relation_dataset.csv`) com amostragem negativa controlada (`NONE`) e particionamento estrito por paciente (`StratifiedGroupKFold`, 5 splits).
* **`relation_classifier.py`:**
  * `ClinicalRelationClassifier`: Classificadores clássicos (Naive Bayes, Regressão Logística L2, Linear SVM, Random Forest) e estudo sistemático de ablação de N-grams.
  * `explain_prediction`: Explicabilidade Local (XAI Linear) que calcula o produto peso $\times$ TF-IDF ($\beta_i \cdot x_i$) para cada N-gram ativo.
* **`mlp_classifier.py`:**
  * `TwoLayerClinicalMLP`: Rede neural Feedforward de 2 camadas ($D \to 128 \to 64 \to K$) em PyTorch com ativações ReLU, Dropout estocástico e regularização L2 (Weight Decay), conforme Slide `nlp-110`.
* **`graph_integrator.py`:**
  * `NeuroSymbolicGraphIntegrator`: Calibração Bayesiana suave combinando o classificador textual $P_{\text{ML}}$ com o *Prior Ontológico MeSH* $P_{\text{onto}}$ via suavização de Laplace (Slide `nlp-080`).
  * `build_enhanced_graph`: Fusão de nós OOV promovidos, arestas determinísticas do P1 e novas arestas estatísticas ($P \ge \tau$).
* **`visualizer_p2.py`:**
  * `generate_cytoscape_p2_html`: Exportador da interface web interativa com Cytoscape.js, slider dinâmico de limiar de confiança $\tau \in [0.5, 1.0]$ e card lateral de Raio-X (XAI) ao clicar na aresta.
* **`run_pipeline_p2.py`:**
  * Script orquestrador mestre que executa todo o pipeline ponta-a-ponta e gera os resultados finais.

---

## 🚀 Como Executar

### 1. Pré-requisitos e Dependências
Certifique-se de ter Python 3.10+ com as seguintes bibliotecas:
```bash
pip install numpy scipy scikit-learn pandas torch pytest
```

### 2. Executar a Suíte Completa de Testes
```bash
python -m pytest project2/tests/ -v
```

### 3. Executar o Pipeline Ponta-a-Ponta
```bash
python project2/src/run_pipeline_p2.py
```

### 4. Abrir a Demonstração Interativa
Abra o arquivo HTML gerado no navegador:
```
project2/data/processed/graph_p2_visualization.html
```
ou execute o Jupyter Notebook:
```
project2/pipelines/notebooks/pipeline_p2_demo.ipynb
```
