# Resumo das execuções planejadas

Este repositório retoma o pipeline original do artigo, limitado aos recursos descritos publicamente (oversampling simples, sem novas técnicas de aumento de dados) e acrescenta apenas controles de rastreabilidade e relatórios quantitativos adicionais.

## O que foi executado ou automatizado
- **Baseline reproduzido:** `exp01_cv_pretrained.py` realiza validação cruzada 5-fold com a R3D-18 pré-treinada, salvando pesos, métricas por fold e curvas ROC no diretório `C:\dataset\runs\baseline_cv_pretrained`.
- **Ablação existente repetida:** `exp02_cv_scratch.py` repete o estudo sem pesos pré-treinados no diretório `C:\dataset\runs\ablation_cv_scratch`.
- **Holdout consistente:** `exp03_holdout_final_roc.py` gera a curva ROC 85/15 a partir do mesmo modelo/partições usados na Tabela 3, garantindo que o AUC seja calculado com as mesmas probabilidades usadas para a curva.

## Por que atende aos requisitos
- **Sem novas técnicas:** Apenas o oversampling simples com `WeightedRandomSampler` é aplicado nos conjuntos de treino, conforme descrito no repositório original.
- **Rastreabilidade total:** Seeds, versões de dependências, parâmetros de treino e métricas são gravados em cada pasta de execução (`config.json`, `versions.json`, `metrics_summary.json` ou `holdout_metrics.json`).
- **Consistência AUC/ROC:** Os valores de AUC tabelados são calculados diretamente a partir das probabilidades que geram as curvas ROC (`foldX_roc.csv`), evitando discrepâncias.

## Novidade quantitativa adicionada (sem alterar pipeline)
- **Intervalos de confiança e desvios padrão** são reportados por fold em `metrics_summary.json`, tanto no baseline quanto na ablação. Isso amplia a transparência dos resultados sem introduzir novas técnicas ou dados.
- **Nomeação explícita de execuções** diferencia baseline e extensão, mantendo clara a comparação com o artigo original.
