# treinamento-cnn-3d-2026

Experimentos reprodutíveis com Redes Neurais Convolucionais 3D aplicadas à classificação binária de ressonância magnética do joelho (3D DESS), incluindo validação cruzada 5-fold, estudo de ablação com transferência de aprendizado e avaliação consistente por ROC/AUC, adaptados para ambiente Windows 11 com GPU única - dado que ambiente UNIX (Laboratório de Processamento de Sinais - LASAI) realizado há 1 ano atrás foi formatado, e já solicitei novo acesso sem sucesso para reprodutividade lá.

---

## Visão Geral

Este repositório apresenta um pipeline experimental reprodutível para classificação binária de volumes de ressonância magnética do joelho adquiridos com a sequência **3D DESS (Double Echo Steady State)**, utilizando **Redes Neurais Convolucionais 3D (CNN 3D)** baseadas na arquitetura **R3D-18**.

O trabalho dá continuidade a experimentos previamente conduzidos em ambiente Linux com múltiplas GPUs, incorporando **novos experimentos**, maior rigor metodológico e **consistência entre métricas quantitativas (AUC) e curvas ROC**, agora adaptados para execução local em **Windows 11 com uma única GPU**.

---

## Objetivos

- Avaliar o desempenho do modelo por meio de **validação cruzada 5-fold**, adotada como protocolo experimental principal.
- Conduzir um **estudo de ablação** para investigar o impacto da **transferência de aprendizado**, comparando:
  - um modelo inicializado com pesos pré-treinados;
  - um modelo treinado do zero.
- Garantir **consistência entre valores de AUC e curvas ROC**, evitando divergências entre resultados apresentados em tabelas e figuras.
- Disponibilizar um pipeline **reprodutível, claro e adequado para publicação científica**.

---

## Base de Dados

A base de dados é composta por volumes de ressonância magnética do joelho adquiridos com a sequência **3D DESS**, derivados da base pública **Osteoarthritis Initiative (OAI)**.

Os dados pré-processados são armazenados em arquivos NumPy (`.npy`) no diretório `C:\dataset\data`, contendo:

- `normal-3DESS-128-64.npy`: volumes de joelhos sem evidência de osteoartrite.
- `abnormal-3DESS-128-64.npy`: volumes de joelhos com alterações associadas à osteoartrite.

Cada amostra corresponde a um volume tridimensional com dimensão **128 × 128 × 64**.

Os dados brutos de ressonância magnética não são disponibilizados neste repositório por restrições de licenciamento. O protocolo de pré-processamento segue o mesmo adotado em trabalhos anteriores.

---

## Organização do Projeto

O projeto está organizado de forma a separar claramente dados, código-fonte e resultados experimentais:

- Os volumes pré-processados permanecem em `C:\dataset\data`.
- O código-fonte dos experimentos encontra-se no diretório do repositório.
- Os resultados (modelos treinados, métricas e logs) são armazenados em subdiretórios dentro de `C:\dataset\runs`.

Essa separação facilita a reprodutibilidade e a rastreabilidade dos experimentos.

---

## Protocolo Experimental

### Experimento 01 — Validação Cruzada (Baseline)

Este experimento corresponde ao resultado principal do estudo.

- Script: `exp01_cv_pretrained.py`
- Protocolo: validação cruzada **5-fold**
- Arquitetura: R3D-18 com **pesos pré-treinados**
- Métricas avaliadas: acurácia, AUC-ROC e F1-score

As saídas incluem os modelos salvos por fold, métricas agregadas (média e desvio padrão) e logs completos de treinamento.

---

### Experimento 02 — Validação Cruzada (Estudo de Ablação)

Este experimento tem como objetivo gerar **novos resultados** em relação a trabalhos anteriores.

- Script: `exp02_cv_scratch.py`
- Protocolo: validação cruzada **5-fold**
- Arquitetura: R3D-18 **treinada do zero** (sem pesos pré-treinados)

Os resultados permitem avaliar quantitativamente o impacto da transferência de aprendizado.

---

### Experimento 03 — Avaliação Holdout e Curva ROC

Este experimento é utilizado como **análise complementar**.

- Script: `exp03_holdout_final_roc.py`
- Protocolo: divisão **85/15** em treinamento e teste
- O modelo utilizado é selecionado a partir da validação cruzada

O experimento gera o valor final de AUC no conjunto de teste e os dados necessários para a construção da curva ROC.

A avaliação holdout **não substitui** a validação cruzada como métrica principal.

---

## Ambiente Computacional

### Ambiente atual (experimentos 2026)

- Sistema operacional: Windows 11
- GPU: NVIDIA RTX 5050 (8 GB de VRAM)
- Memória RAM: 16 GB
- Python: 3.11
- Bibliotecas principais: PyTorch, Torchvision, NumPy, Scikit-learn

### Ambiente utilizado em trabalhos anteriores

- Sistema operacional: Linux
- GPUs: duas NVIDIA RTX 2080 Ti (12 GB cada)

O pipeline foi adaptado para execução em GPU única, mantendo o rigor metodológico.

---

## Instalação

1. Criar e ativar um ambiente virtual Python.
2. Instalar as dependências listadas em `requirements.txt`.
3. Verificar a disponibilidade da GPU via PyTorch.

---

## Execução dos Experimentos

- Executar o Experimento 01 para obtenção do baseline com pesos pré-treinados.
- Executar o Experimento 02 para o estudo de ablação.
- Executar o Experimento 03 apenas para análise complementar da curva ROC.

Os scripts recebem como parâmetros o diretório dos dados, o diretório de saída e a semente aleatória utilizada para reprodutibilidade.

### Scripts de Experimento e Rastreamento

Três entrypoints reproduzem o pipeline original descrito em [thallescotta/treinamento-cnn-3d](https://github.com/thallescotta/treinamento-cnn-3d) sem novos artifícios de aumento de dados (apenas oversampling simples com `WeightedRandomSampler`).

- `exp01_cv_pretrained.py`: validação cruzada 5-fold com R3D-18 pré-treinada (`name=baseline_cv_pretrained`).
- `exp02_cv_scratch.py`: ablação treinando do zero (`name=ablation_cv_scratch`).
- `exp03_holdout_final_roc.py`: divisão 85/15 para geração da curva ROC final (`name=holdout_final_roc`).

Todos compartilham utilitários em `experiment_utils.py`, registram seeds (`numpy`, `random`, `torch`), versões das dependências em `versions.json` e salvam métricas/curvas em `C:\dataset\runs/<nome_do_experimento>` (ou em outro diretório definido pela variável `RUNS_DIR`).

### Consistência de AUC/ROC

- A probabilidade positiva usada para calcular o AUC também alimenta a curva ROC (arquivos `foldX_predictions.csv` e `foldX_roc.csv`), evitando divergências entre tabela e figura.
- Tanto na validação cruzada quanto no holdout, o melhor modelo por AUC é o único autorizado a gerar a curva ROC correspondente.

### Novidade quantitativa sem alterar o pipeline

- O estudo de ablação (pretrained vs. scratch) permanece idêntico ao artigo original, mas agora registra intervalos de confiança de 95% e desvios padrão por fold para acurácia, AUC, F1 e perda em `metrics_summary.json`.
- Não há novas técnicas de aumento de dados (por exemplo, SMOTE segue proibido); apenas oversampling simples é aplicado nas amostras de treino para balancear as classes.

### Como executar

```bash
python exp01_cv_pretrained.py  # baseline com pesos pré-treinados
python exp02_cv_scratch.py     # ablação treinando do zero
python exp03_holdout_final_roc.py  # curva ROC final 85/15
```

Parâmetros como `RUNS_DIR`, `DATA_DIR`, número de épocas ou tamanho de lote podem ser ajustados editando `ExperimentConfig` dentro de cada script ou definindo variáveis de ambiente antes da execução.

---

## Uso dos Resultados no Artigo

- Resultados principais: validação cruzada 5-fold (média e desvio padrão).
- Nova contribuição: estudo de ablação sobre transferência de aprendizado.
- Curvas ROC: geradas a partir de um conjunto de teste claramente definido.
- Inconsistências evitadas: divergência entre valores de AUC em tabelas e figuras.

---

## Autor

Thalles Cotta Fontainha  
Doutorando em Instrumentação e Óptica Aplicada — CEFET/RJ

---

## Licença

Este repositório destina-se exclusivamente a fins acadêmicos e de pesquisa.
