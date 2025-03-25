import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Função para carregar os dados
def carregar_dados(arquivo):
    numeros_reais = []
    with open(arquivo, 'r') as file:
        linhas = file.readlines()
        for linha in linhas:
            linha_formatada = linha.strip().replace('[', '').replace(']', '').replace(' ', '')
            if linha_formatada:
                numeros_reais.append(list(map(int, linha_formatada.split(','))))
    return numeros_reais

# Função para criar features adicionais
def criar_features(numeros_reais):
    # Cada linha terá 25 features indicando se o número (de 1 a 25) foi sorteado
    X = np.zeros((len(numeros_reais), 25))
    features_extras = np.zeros((len(numeros_reais), 3))  # Pares, ímpares, soma
    for i, sorteio in enumerate(numeros_reais):
        for numero in sorteio:
            X[i, numero-1] = 1  # Ajusta para índice 0 a 24
        pares = sum(1 for n in sorteio if n % 2 == 0)
        impares = len(sorteio) - pares
        soma = sum(sorteio)
        features_extras[i] = [pares, impares, soma]
    # Retorna features combinadas e também a matriz original de 25 colunas
    return np.hstack((X, features_extras)), X

# Função para gerar variações com probabilidades
def gerar_variacoes_com_prob(modelo, X_train, n_variacoes=10, n_numeros=15):
    previsoes_variacoes = []
    for _ in range(n_variacoes):
        # Gera entrada aleatória para as 25 posições e adiciona as 3 features extras
        entrada_aleatoria = np.zeros((1, 28))  # 25 números + 3 features
        # A probabilidade de cada número ser 1 é baseada na média do treinamento
        entrada_aleatoria[:, :25] = np.random.rand(1, 25) < X_train[:, :25].mean(axis=0)
        
        # Calcula features extras (pares, ímpares, soma)
        numeros_selecionados = np.where(entrada_aleatoria[0, :25] == 1)[0] + 1  # Ajusta para números de 1 a 25
        pares = sum(1 for n in numeros_selecionados if n % 2 == 0)
        impares = len(numeros_selecionados) - pares
        soma = sum(numeros_selecionados)
        
        entrada_aleatoria[0, 25] = pares
        entrada_aleatoria[0, 26] = impares
        entrada_aleatoria[0, 27] = soma
        
        # Previsão para cada número (usando todas as 28 features)
        previsao_proba = modelo.predict_proba(entrada_aleatoria)
        # Para cada número, pega a probabilidade de classe 1
        probs = np.array([proba[0, 1] for proba in previsao_proba])
        # Seleciona os n_numeros com maior probabilidade (ordenados decrescentemente)
        indices_numeros = np.argsort(probs)[-n_numeros:][::-1]
        numeros_ordenados = sorted(indices_numeros + 1)  # Ajusta para numeração de 1 a 25
        probas = probs[indices_numeros]
        previsoes_variacoes.append((numeros_ordenados, probas))
    return previsoes_variacoes

# Carregamento e pré-processamento dos dados
numeros_reais = carregar_dados('sequencias_formatadas.txt')
X, _ = criar_features(numeros_reais)  # X possui 28 features (25 + 3)

# Para o modelo, vamos usar os 25 primeiros números como target multi-label (ex.: cada número: 0 ou 1)
y = X[:, :25]

# Divisão de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Otimização de hiperparâmetros com Grid Search
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
modelo = grid_search.best_estimator_

# Avaliação do modelo (usando média de acurácia para cada um dos 25 classificadores)
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

# Análise estatística com base nos 25 números (somente parte das features originais)
frequencia = X_train[:, :25].sum(axis=0)
media = X_train[:, :25].mean(axis=0)
mediana = np.median(X_train[:, :25], axis=0)
moda = pd.DataFrame(X_train[:, :25]).mode().iloc[0].values

# Geração de previsões
previsoes = gerar_variacoes_com_prob(modelo, X_train)

# Interface com Streamlit
st.title("Análise e Previsão de Números Sorteados")

# Exibir métricas do modelo
st.subheader("Métricas do Modelo")
st.write(f"Acurácia: {acuracia:.2f}")
st.write(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# Exibir estatísticas dos números utilizando gráficos customizados

# Média
st.subheader("Média dos Números (Treinamento)")
fig_media, ax_media = plt.subplots(figsize=(8, 4))
ax_media.bar(range(1, 26), media, color='coral')
ax_media.set_xlabel("Número")
ax_media.set_ylabel("Média")
ax_media.set_title("Média dos Números")
st.pyplot(fig_media)

# Mediana
st.subheader("Mediana dos Números (Treinamento)")
fig_mediana, ax_mediana = plt.subplots(figsize=(8, 4))
ax_mediana.bar(range(1, 26), mediana, color='mediumseagreen')
ax_mediana.set_xlabel("Número")
ax_mediana.set_ylabel("Mediana")
ax_mediana.set_title("Mediana dos Números")
st.pyplot(fig_mediana)

# Moda
st.subheader("Moda dos Números (Treinamento)")
fig_moda, ax_moda = plt.subplots(figsize=(8, 4))
ax_moda.bar(range(1, 26), moda, color='royalblue')
ax_moda.set_xlabel("Número")
ax_moda.set_ylabel("Moda")
ax_moda.set_title("Moda dos Números")
st.pyplot(fig_moda)

# Frequência
st.subheader("Frequência de Ocorrência (Treinamento)")
fig_freq, ax_freq = plt.subplots(figsize=(8, 4))
ax_freq.bar(range(1, 26), frequencia, color='purple')
ax_freq.set_xlabel("Número")
ax_freq.set_ylabel("Frequência")
ax_freq.set_title("Frequência de Ocorrência dos Números")
st.pyplot(fig_freq)

# Previsões com probabilidades
st.subheader("Previsões dos Próximos Números")
for i, (variacao, probas) in enumerate(previsoes):
    numeros_formatados = ', '.join(str(int(num)) for num in variacao)
    st.markdown(f"**Variação {i + 1}:** [{numeros_formatados}]")
    fig_prev, ax_prev = plt.subplots(figsize=(8, 4))
    ax_prev.bar(range(1, len(probas)+1), probas, color='teal')
    ax_prev.set_xlabel("Índice (Ordem Decrescente de Probabilidade)")
    ax_prev.set_ylabel("Probabilidade")
    ax_prev.set_title(f"Probabilidades - Variação {i + 1}")
    st.pyplot(fig_prev)

# Análise de padrões temporais
st.subheader("Padrões Temporais")
intervalo_aparicao = {}
for i, sorteio in enumerate(numeros_reais):
    for numero in sorteio:
        if numero not in intervalo_aparicao:
            intervalo_aparicao[numero] = []
        intervalo_aparicao[numero].append(i)
for numero, aparicoes in intervalo_aparicao.items():
    if len(aparicoes) > 1:
        diffs = np.diff(aparicoes)
        media_diffs = np.mean(diffs)
        st.write(f"Número {numero} aparece a cada {media_diffs:.1f} sorteios em média.")

# Heatmap de frequência customizado
st.subheader("Heatmap de Frequência Customizado")
heatmap_data = pd.DataFrame(X[:, :25], columns=range(1, 26))
fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
cax = ax_heat.matshow(heatmap_data.T, cmap='viridis')
fig_heat.colorbar(cax)
ax_heat.set_xticks(range(heatmap_data.shape[0]))
ax_heat.set_xticklabels(range(1, heatmap_data.shape[0] + 1))
ax_heat.set_yticks(range(25))
ax_heat.set_yticklabels(range(1, 26))
ax_heat.set_xlabel("Amostras")
ax_heat.set_ylabel("Números")
ax_heat.set_title("Heatmap de Frequência dos Números")
st.pyplot(fig_heat)