import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Dataset
data_url = "https://archive.ics.uci.edu/static/public/45/data.csv"
df = pd.read_csv(data_url, na_values='NaN')

print("Primeiras 5 linhas do dataset:")
print(df.head())

categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
X = df.drop(columns=['num'])
Y = df['num']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ], remainder='passthrough'
)

# 85% treino, 15% teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42))
])

# Treinar o modelo
pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)

# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(Y_test, Y_pred))

# Precisão
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nPrecisão do modelo: {accuracy:.4f}")

# Matriz de Confusão
cm = confusion_matrix(Y_test, Y_pred)

# Plotando a Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Sadio", "Doente"], yticklabels=["Sadio", "Doente"])
plt.title("Matriz de Confusão")
plt.xlabel("Predição")
plt.ylabel("Realidade")
plt.show()

# Gráfico de Pizza
labels = ['Sadio', 'Doente']
sizes = [sum(Y_test == 0), sum(Y_test == 1)]
colors = ['#66ff66', '#ff6666']
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Distribuição das Classes no Conjunto de Teste")
plt.axis('equal')
plt.show()
