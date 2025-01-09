import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Dicionários
sexo_descricao = {1: "Homem", 0: "Mulher"}
dor_peito_descricao = {1: "angina típica", 2: "angina atípica", 3: "dor não anginosa", 4: "assintomático", 0: "valor desconhecido"}
asj_descricao = {1: "Verdadeiro", 0: "Falso"}
ECG_descricao = {1: "normal", 2: "anormalidade da onda ST-T", 3: "hipertrofia ventricular esquerda"}
AINEX_descricao = {1: "Sim", 0: "Não"}
INCLI_descricao = {1: "ascendente", 2: "plano", 3: "descida"}
TAL_descricao = {3: "normal", 6: "defeito corrigido", 7: "defeito reversível"}
NUM_descricao = {0: "sadio", 1: "doente"}

# Listas
idades = [63, 37, 41, 56, 57]
sexo = [1, 1, 0, 1, 0]
dor_peito = [1, 2, 1, 1, 0]
PA_rep = [145, 130, 130, 120, 120]
Col_serico = [233, 250, 204, 236, 354]
ASJ = [1, 0, 0, 0, 0]
ECG_rep = [1, 2, 3, 1, 2]
MFC = [150, 187, 172, 178, 163]
AINEX = [0, 0, 1, 0, 1]
DEPSTEX = [2.3, 3.5, 1.4, 0.8, 0.6]
INCLI = [1, 2, 3, 1, 2]
CA = [0, 0, 0, 0, 0]
TAL = [1, 3, 3, 3, 3]
NUM = [1, 1, 1, 1, 1]

# Função de transformação e validação
def transformar(lista, descricao, valores_validos=None):
    transformados = []
    for val in lista:
        if valores_validos is not None and val not in valores_validos:
            transformados.append("Valor inválido")
        else:
            transformados.append(descricao.get(val, "Desconhecido"))
    return transformados

# Validação e transformação das listas
sexo_transformado = transformar(sexo, sexo_descricao, valores_validos=[0, 1])
asj_transformado = transformar(ASJ, asj_descricao, valores_validos=[0, 1])
dor_peito_transformado = transformar(dor_peito, dor_peito_descricao, valores_validos=[0, 1, 2, 3, 4])
ECG_transformado = transformar(ECG_rep, ECG_descricao, valores_validos=[1, 2, 3])
AINEX_transformado = transformar(AINEX, AINEX_descricao, valores_validos=[0, 1])
INCLI_transformado = transformar(INCLI, INCLI_descricao, valores_validos=[1, 2, 3])
CA_validado = transformar(CA, TAL_descricao, valores_validos=[0, 1, 2, 3])
TAL_validado = transformar(TAL, TAL_descricao, valores_validos=[3, 6, 7])
NUM_transformado = transformar(NUM, NUM_descricao, valores_validos=[0, 1])

# Criando DataFrame
data = {
    "idade": idades,
    "sexo": sexo_transformado,
    "dor_peito": dor_peito_transformado,
    "PA_rep": PA_rep,
    "Col_serico": Col_serico,
    "ASJ": asj_transformado,
    "ECG_rep": ECG_transformado,
    "MFC": MFC,
    "AINEX": AINEX_transformado,
    "DEPSTEX": DEPSTEX,
    "INCLI": INCLI_transformado,
    "CA": CA_validado,
    "TAL": TAL_validado,
    "NUM": NUM_transformado
}

dataset = pd.DataFrame(data)

# Agrupando e garantindo todas as combinações possíveis
df_sex = dataset.groupby(['sexo', 'NUM']).size().reset_index(name='quantidade')
combinacoes = pd.MultiIndex.from_product([["Mulher", "Homem"], ["sadio", "doente"]], names=['sexo', 'NUM'])
df_sex = df_sex.set_index(['sexo', 'NUM']).reindex(combinacoes, fill_value=0).reset_index()

# Exibindo o DataFrame com as combinações
print(df_sex)

# Gráfico de pizza
if df_sex["quantidade"].sum() > 0:
    labels = df_sex.apply(lambda row: f"{row['sexo']}, {row['NUM']}", axis=1)
    plt.pie(df_sex["quantidade"], labels=labels, autopct='%1.1f%%', radius=1.5, textprops={"fontsize": 16})
    plt.show()
else:
    print("Não há dados suficientes para gerar o gráfico de pizza.")

# Modelagem de Machine Learning

# Preparando os dados para o modelo
X = np.array(list(zip(idades, sexo, PA_rep, Col_serico, MFC, DEPSTEX, INCLI, TAL)))
y = np.array(NUM)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Criando e treinando o modelo
modelo = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# Prevendo e avaliando o modelo
y_pred = modelo.predict(X_test)

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nMatriz de Confusão:")
print(cm)

# Gráfico da matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Sadio", "Doente"], yticklabels=["Sadio", "Doente"])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Sadio", "Doente"], labels=[0, 1], zero_division=0))

# Precisão
accuracy = np.mean(y_test == y_pred) * 100
print(f"\nPrecisão: {accuracy:.2f}%")
