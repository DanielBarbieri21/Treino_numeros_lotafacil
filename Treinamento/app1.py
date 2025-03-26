import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import streamlit as st
import joblib
import os
from sklearn.dummy import DummyClassifier

# Função para carregar os dados
def carregar_dados(arquivo):
    """Carrega os sorteios de um arquivo de texto formatado."""
    numeros_reais = []
    with open(arquivo, 'r') as file:
        linhas = file.readlines()
        for linha in linhas:
            linha_formatada = linha.strip().replace('[', '').replace(']', '').replace(' ', '')
            if linha_formatada:
                numeros_reais.append(list(map(int, linha_formatada.split(','))))
    return numeros_reais

# Função para criar features adicionais
def criar_features(numeros_reais, n_ultimos=10):
    """Cria features para o modelo, incluindo frequência recente."""
    X = np.zeros((len(numeros_reais), 25))
    for i, sorteio in enumerate(numeros_reais):
        for numero in sorteio:
            X[i, numero-1] = 1

    features_extras = np.zeros((len(numeros_reais), 3))
    for i, sorteio in enumerate(numeros_reais):
        pares = sum(1 for n in sorteio if n % 2 == 0)
        impares = len(sorteio) - pares
        soma = sum(sorteio)
        features_extras[i] = [pares, impares, soma]

    freq_recente = np.zeros((len(numeros_reais), 25))
    for i in range(len(numeros_reais)):
        inicio = max(0, i - n_ultimos)
        for j in range(inicio, i):
            for num in numeros_reais[j]:
                freq_recente[i, num-1] += 1

    X_completo = np.hstack((X, features_extras, freq_recente))
    return X_completo, X

# Função para gerar variações com probabilidades
def gerar_variacoes_com_prob(modelo, X_train, n_variacoes=5, n_numeros=15, threshold=0.6):
    """Gera variações de previsões com base em probabilidades."""
    previsoes_variacoes = []
    for _ in range(n_variacoes):
        entrada_aleatoria = np.zeros((1, X_train.shape[1]))
        entrada_aleatoria[:, :25] = np.random.rand(1, 25) < X_train[:, :25].mean(axis=0)
        
        numeros_selecionados = np.where(entrada_aleatoria[0, :25] == 1)[0] + 1
        pares = sum(1 for n in numeros_selecionados if n % 2 == 0)
        impares = len(numeros_selecionados) - pares
        soma = sum(numeros_selecionados)
        entrada_aleatoria[0, 25] = pares
        entrada_aleatoria[0, 26] = impares
        entrada_aleatoria[0, 27] = soma
        entrada_aleatoria[0, 28:] = np.random.rand(1, 25) * 10
        
        # Previsão de probabilidades
        previsao_proba = modelo.predict_proba(entrada_aleatoria)
        # Para MultiOutputClassifier com RandomForest, previsao_proba é uma lista de arrays 2D (probabilidade das classes 0 e 1)
        probs = np.array([proba[0, 1] for proba in previsao_proba])  # Probabilidade da classe 1 para cada número
        
        # Selecionar os números com maior probabilidade
        indices_numeros = np.argsort(probs)[-n_numeros:][::-1]
        indices_numeros_list = indices_numeros.tolist()  # Convert to list explicitly
        numeros_ordenados = sorted([x + 1 for x in indices_numeros_list])  # Ajustando índices
        probas = probs[indices_numeros]
        previsoes_variacoes.append((numeros_ordenados, probas))
    return previsoes_variacoes

# Carregamento e pré-processamento
numeros_reais = carregar_dados('sequencias_formatadas.txt')

# Verificar duplicatas nos dados
numeros_reais_set = set(tuple(sorteio) for sorteio in numeros_reais)
if len(numeros_reais_set) != len(numeros_reais):
    print(f"Atenção: Foram encontradas {len(numeros_reais) - len(numeros_reais_set)} duplicatas nos dados!")
else:
    print("Nenhuma duplicata encontrada nos dados.")

# Criar features e target (prever o próximo sorteio)
X, X_base = criar_features(numeros_reais, n_ultimos=10)
# O target y será os números do próximo sorteio (t+1)
y = np.zeros((len(numeros_reais)-1, 25))
for i in range(len(numeros_reais)-1):
    for numero in numeros_reais[i+1]:
        y[i, numero-1] = 1
# X será os sorteios até t (excluímos o último, pois não temos y para ele)
X = X[:-1]

# Verificar e tratar NaN/infinite values
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Atenção: Valores NaN ou infinitos encontrados em X. Substituindo por 0...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    print("Atenção: Valores NaN ou infinitos encontrados em y. Substituindo por 0...")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# Divisão de treino e teste (usando divisão temporal para evitar vazamento de dados)
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Forçar retrreinamento deletando o modelo salvo
if os.path.exists('modelo_lotofacil.pkl'):
    os.remove('modelo_lotofacil.pkl')
    print("Modelo anterior deletado. Retrreinando...")

# Verificar se o modelo já foi treinado e salvo
try:
    modelo = joblib.load('modelo_lotofacil.pkl')
    print("Modelo carregado do arquivo.")
    f1 = f1_score(y_test, modelo.predict(X_test), average='micro')
except FileNotFoundError:
    # Baseline: Dummy Classifier (random predictions)
    dummy = MultiOutputClassifier(DummyClassifier(strategy='stratified', random_state=42), n_jobs=-1)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    f1_dummy = f1_score(y_test, y_pred_dummy, average='micro')
    print(f"F1-score Dummy Classifier (baseline): {f1_dummy:.2f}")

    # Otimização de hiperparâmetros para RandomForest
    param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [5, 10],
        'estimator__min_samples_split': [5, 10],
        'estimator__min_samples_leaf': [2, 4]
    }
    rf_base = RandomForestClassifier(random_state=42, class_weight={0: 1.0, 1: 0.8})
    modelo_rf = MultiOutputClassifier(rf_base, n_jobs=-1)
    grid_search = GridSearchCV(modelo_rf, param_grid, cv=3, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    modelo_rf = grid_search.best_estimator_

    # Avaliação do RandomForest
    y_pred_rf = modelo_rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average='micro')
    print(f"F1-score RandomForest: {f1_rf:.2f}")

    # Treinar e avaliar XGBoost com regularização ajustada
    xgb_base = XGBClassifier(
        random_state=42,
        objective='binary:logistic',
        n_jobs=-1,
        max_depth=3,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.5,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=0.67
    )
    modelo_xgb = MultiOutputClassifier(xgb_base, n_jobs=-1)
    modelo_xgb.fit(X_train, y_train)
    y_pred_xgb = modelo_xgb.predict(X_test)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='micro')
    print(f"F1-score XGBoost: {f1_xgb:.2f}")

    # Validação cruzada para avaliar o desempenho real
    scores = cross_val_score(modelo_xgb, X, y, cv=5, scoring='f1_macro')
    print(f"F1-score médio (validação cruzada, macro) XGBoost: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Seleção do melhor modelo
    modelo = modelo_xgb if f1_xgb > f1_rf else modelo_rf
    f1 = f1_xgb if f1_xgb > f1_rf else f1_rf
    print(f"Modelo selecionado: {'XGBoost' if modelo == modelo_xgb else 'RandomForest'}")

    # Verificar se o modelo está prevendo sempre a mesma classe
    y_pred = modelo.predict(X_test)
    unique_preds = np.unique(y_pred, axis=0)
    if len(unique_preds) == 1:
        print("Atenção: O modelo está prevendo sempre a mesma classe!")

    # Salvar o modelo treinado
    joblib.dump(modelo, 'modelo_lotofacil.pkl')
    print("Modelo salvo em 'modelo_lotofacil.pkl'.")

# Avaliação detalhada
print(classification_report(y_test, modelo.predict(X_test)))

# Estatísticas dos números
frequencia = X_train[:, :25].sum(axis=0)
media = X_train[:, :25].mean(axis=0)
mediana = np.median(X_train[:, :25], axis=0)
moda = pd.DataFrame(X_train[:, :25]).mode().iloc[0].values

# Geração de previsões
previsoes = gerar_variacoes_com_prob(modelo, X_train)

# Interface com Streamlit
st.title("Previsão de Números da Lotofácil")

# Métricas do modelo
st.subheader("Métricas do Modelo")
st.write(f"F1-score: {f1:.2f}")

# Gráficos de estatísticas
st.subheader("Média dos Números")
fig_media, ax_media = plt.subplots(figsize=(8, 4))
ax_media.bar(range(1, 26), media, color='coral')
ax_media.set_xlabel("Número")
ax_media.set_ylabel("Média")
st.pyplot(fig_media)

st.subheader("Frequência dos Números")
fig_freq, ax_freq = plt.subplots(figsize=(8, 4))
ax_freq.bar(range(1, 26), frequencia, color='purple')
ax_freq.set_xlabel("Número")
ax_freq.set_ylabel("Frequência")
st.pyplot(fig_freq)

# Previsões
st.subheader("Previsões")
for i, (variacao, probas) in enumerate(previsoes):
    numeros_formatados = ', '.join(str(int(num)) for num in variacao)
    st.markdown(f"**Variação {i + 1}:** [{numeros_formatados}]")
    fig_prev, ax_prev = plt.subplots(figsize=(8, 4))
    ax_prev.bar(range(1, len(probas)+1), probas, color='teal')
    ax_prev.set_xlabel("Índice")
    ax_prev.set_ylabel("Probabilidade")
    st.pyplot(fig_prev)

# Padrões temporais
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
        st.write(f"Número {numero}: aparece a cada {media_diffs:.1f} sorteios em média.")

# Heatmap de frequência
st.subheader("Heatmap de Frequência")
heatmap_data = pd.DataFrame(X_base[:, :25], columns=range(1, 26))
fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
cax = ax_heat.matshow(heatmap_data.T, cmap='viridis')
fig_heat.colorbar(cax)
ax_heat.set_xticks(range(heatmap_data.shape[0]))
ax_heat.set_xticklabels(range(1, heatmap_data.shape[0] + 1))
ax_heat.set_yticks(range(25))
ax_heat.set_yticklabels(range(1, 26))
ax_heat.set_xlabel("Sorteios")
ax_heat.set_ylabel("Números")
st.pyplot(fig_heat)