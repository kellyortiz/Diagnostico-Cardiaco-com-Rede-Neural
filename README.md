# Análise de Fatores Cardíacos: Implementação de Rede Neural para Diagnóstico de Doenças Cardíacas

## Diagnóstico Cardíaco com Rede Neural

Este repositório contém o estudo de caso sobre o diagnóstico de doenças cardíacas usando uma rede neural artificial. O estudo foi baseado em dados reais de pacientes com histórico de problemas cardíacos fornecidos pela Cleveland Clinic Foundation. O objetivo foi identificar padrões e relações entre diferentes variáveis que podem indicar condições cardíacas.

## Descrição

Patrícia, uma estudante de engenharia biomédica, acessou um conjunto de dados de 303 pacientes, contendo 13 variáveis relacionadas à saúde cardíaca. O objetivo foi explorar esses dados para entender melhor as condições mais propensas ao desenvolvimento de doenças cardíacas, utilizando técnicas de aprendizado de máquina, especificamente redes neurais.

O estudo se concentra em como usar redes neurais artificiais para processar e analisar esses dados, superando limitações gráficas ao lidar com múltiplas variáveis.

## Estrutura dos Dados

A tabela de dados contém os seguintes parâmetros para cada paciente:

| Variável Solicitada | Nome no Dataset UCI | Descrição |
|---------------------|---------------------|-----------|
| Idade               | age                 | Idade do paciente. |
| Sexo                | sex                 | Sexo do paciente: 1 = homem, 0 = mulher. |
| Dor_Peito           | cp                  | Tipo de dor no peito: 1 = angina típica, 2 = angina atípica, 3 = dor não anginosa, 4 = assintomático. |
| PA_rep (Pressão arterial em repouso) | trestbps | Pressão arterial em repouso (em mm Hg). |
| Col_serico (Colesterol sérico) | chol | Colesterol sérico (em mg/dl). |
| ASJ (Açúcar no sangue em jejum) | fbs | Açúcar no sangue em jejum > 120mg/dl: 1 = verdade, 0 = falso. |
| ECG_rep (Eletrocardiograma em repouso) | restecg | Resultados do eletrocardiograma em repouso: 0 = normal, 1 = anormalidade da onda ST-T, 2 = hipertrofia ventricular esquerda. |
| MFC (Máxima frequência cardíaca alcançada) | thalach | Máxima frequência cardíaca alcançada (em bpm). |
| AINEX (Angina induzida por exercício) | exang | Angina induzida por exercício: 1 = sim, 0 = não. |
| DEPSTEX (Depressão do segmento ST induzida por exercício) | oldpeak | Depressão do segmento ST induzida por exercício (medido em mm). |
| INCLI (Inclinação do segmento ST) | slope | Inclinação do segmento ST no exercício: 1 = ascendente, 2 = plano, 3 = descendente. |
| CA (Número de vasos principais) | ca | Número de vasos principais (0 a 3) coloridos por fluoroscopia. |
| TAL (Talassemia) | thal | Tipo de talassemia: 3 = normal, 6 = defeito corrigido, 7 = defeito reversível. |
| NUM (Condição do paciente) | target | Diagnóstico de doença cardíaca: 0 = sadio, 1 = doente. |

**Link do Dataset**: [https://archive.ics.uci.edu/static/public/45/data.csv](https://archive.ics.uci.edu/static/public/45/data.csv)

## Pré-processamento de Dados

- **One-Hot Encoding**: As variáveis categóricas foram convertidas em variáveis binárias usando a técnica de one-hot encoding.
- **Divisão dos Dados**: Os dados foram divididos em 85% para treinamento e 15% para validação da rede neural.
- **Normalização**: Os dados foram normalizados para ter média zero e desvio padrão unitário.

## Implementação da Rede Neural

A rede neural foi criada e treinada utilizando a biblioteca Keras com TensorFlow. O objetivo foi classificar os pacientes como saudáveis (0) ou doentes (1) com base nas variáveis fornecidas.

## Arquivos

- **estudoDeCaso.py**  
Esse arquivo utiliza uma lista física com base nos prints do estudo de caso da faculdade. Nele, são realizados os primeiros passos de análise e exploração dos dados de maneira mais simples, com foco na visualização e análise de algumas variáveis.

- **analise_doencas_cardiacas_matriz_confusao.py**  
Esse arquivo é um estudo mais avançado que utiliza o conjunto de dados do UCI Machine Learning Repository para realizar uma análise mais profunda. Ele usa a técnica de redes neurais para classificar os pacientes e gerar uma matriz de confusão, permitindo a avaliação do desempenho do modelo.

## Google Colab

Para rodar o arquivo `analise_doencas_cardiacas_matriz_confusao.py` no Google Colab, é necessário instalar a biblioteca `ucimlrepo` antes de executar o código. Utilize o seguinte comando para instalar a biblioteca:

```python
!pip install ucimlrepo
