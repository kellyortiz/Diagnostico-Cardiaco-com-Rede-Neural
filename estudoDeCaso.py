import pandas as pd
import matplotlib.pyplot as plt

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

# Função
def transformar(lista, descricao, valores_validos=None):
    transformados = []
    for val in lista:
        if valores_validos is not None and val not in valores_validos:
            transformados.append("Valor inválido")
        else:
            transformados.append(descricao.get(val, "Desconhecido"))
    return transformados

# Validação
sexo_transformado = transformar(sexo, sexo_descricao, valores_validos=[0, 1])
asj_transformado = transformar(ASJ, asj_descricao, valores_validos=[0, 1])
dor_peito_transformado = transformar(dor_peito, dor_peito_descricao, valores_validos=[0, 1, 2, 3, 4])
ECG_transformado = transformar(ECG_rep, ECG_descricao, valores_validos=[1, 2, 3])
AINEX_transformado = transformar(AINEX, AINEX_descricao, valores_validos=[0, 1])
INCLI_transformado = transformar(INCLI, INCLI_descricao, valores_validos=[1, 2, 3])
CA_validado = transformar(CA, TAL_descricao, valores_validos=[0, 1, 2, 3])
TAL_validado = transformar(TAL, TAL_descricao, valores_validos=[3, 6, 7])
NUM_transformado = transformar(NUM, NUM_descricao, valores_validos=[0, 1])

# DataFrame
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

# Criando o DataFrame com quantidade
df_sex = dataset.groupby(['sexo', 'NUM']).size().reset_index(name='quantidade')

# Garantindo todas as combinações possíveis
combinacoes = pd.MultiIndex.from_product([['Mulher', 'Homem'], ['sadio', 'doente']], names=['sexo', 'NUM'])
df_sex = df_sex.set_index(['sexo', 'NUM']).reindex(combinacoes, fill_value=0).reset_index()

print(df_sex)

# Gráfico de pizza
if df_sex["quantidade"].sum() > 0:
    labels = df_sex.apply(lambda row: f"{row['sexo']}, {row['NUM']}", axis=1)
    plt.pie(df_sex["quantidade"], labels=labels, autopct='%1.1f%%', radius=1.5, textprops={"fontsize": 16})
    plt.show()
else:
    print("Não há dados suficientes para gerar o gráfico de pizza.")
