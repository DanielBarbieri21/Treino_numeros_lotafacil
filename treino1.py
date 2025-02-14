import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Carregando os dados a partir de um arquivo local 'treino.txt'
numeros_reais = []
with open('sequencias_formatadas.txt', 'r') as file:
    linhas = file.readlines()
    for linha in linhas:
        linha_formatada = linha.strip().replace('[', '').replace(']', '').replace(' ', '')
        if linha_formatada:  # Verifica se a linha não está vazia
            numeros_reais.append(list(map(int, linha_formatada.split(','))))

# Preprocessamento: criar uma matriz de frequência de 0 e 1 para os números de 1 a 25
X = np.zeros((len(numeros_reais), 25))
for i, sorteio in enumerate(numeros_reais):
    for numero in sorteio:
        X[i, numero-1] = 1

# Definindo os rótulos como os próprios números sorteados
y = X.copy()

# Divisão de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo com RandomForest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Função para gerar 10 variações de 15 números prováveis
def gerar_variacoes(modelo, n_variacoes=10, n_numeros=15):
    previsoes_variacoes = []
    for _ in range(n_variacoes):
        entrada_aleatoria = np.random.rand(1, 25) < X_train.mean(axis=0)
        previsao_proba = modelo.predict(entrada_aleatoria)[0]
        indices_numeros = np.argsort(previsao_proba)[-n_numeros:]
        numeros_ordenados = sorted(indices_numeros + 1)
        previsoes_variacoes.append(numeros_ordenados)
    return previsoes_variacoes

# Gerando as 10 variações
proximas_previsoes = gerar_variacoes(modelo)
print("10 variações dos próximos números possíveis:")
for i, variacao in enumerate(proximas_previsoes):
    print(f"Variação {i + 1}: {variacao}")

# Gráfico de Frequência dos Números de 01 a 25 com base no treinamento
frequencia_numeros = X_train.sum(axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(1, 26), frequencia_numeros, color='skyblue')
plt.title('Frequência dos Números Sorteados (Treinamento)')
plt.xlabel('Número')
plt.ylabel('Frequência')
plt.xticks(ticks=np.arange(1, 26, 1))
plt.show()

# Gráfico de Média com os números de 01 a 25
media_numeros = X_train.mean(axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(1, 26), media_numeros, color='lightgreen')
plt.title('Média dos Números Sorteados')
plt.xlabel('Número')
plt.ylabel('Média')
plt.xticks(ticks=np.arange(1, 26, 1))
plt.show()

# Gráfico de Moda com os números de 01 a 25
moda_numeros = pd.DataFrame(X_train).mode().iloc[0]
plt.figure(figsize=(10, 6))
plt.bar(range(1, 26), moda_numeros, color='lightcoral')
plt.title('Moda dos Números Sorteados')
plt.xlabel('Número')
plt.ylabel('Moda')
plt.xticks(ticks=np.arange(1, 26, 1))
plt.show()

# Análise de padrões de aparição
intervalo_aparicao = {}
for i in range(len(numeros_reais) - 1):
    for numero in numeros_reais[i]:
        if numero not in intervalo_aparicao:
            intervalo_aparicao[numero] = []
        intervalo_aparicao[numero].append(i)

conclusoes = []
for numero, aparicoes in intervalo_aparicao.items():
    diffs = np.diff(aparicoes)
    media_diffs = np.mean(diffs)
    conclusoes.append(f"O número {numero} tende a aparecer a cada {media_diffs:.1f} sorteios.")

print("Conclusões sobre os padrões:")
for conclusao in conclusoes:
    print(conclusao)
