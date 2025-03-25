# Treino_numeros_lotafacil
Este treino envolve desde a capitação , limpeza e transformação dos Dados, para dados avaliaveis 
Link para rodar no Colab, por conta de melhor uso e processamento
# https://colab.research.google.com/drive/13tbUHCLSZ1dwqgPgAKOZ2g0vdtubR1Kl?usp=sharing


# Análise e Previsão de Números Sorteados da Lotofácil
# 📋 Visão Geral
Este projeto utiliza técnicas de aprendizado de máquina para analisar e prever números sorteados na Lotofácil, uma das loterias mais populares do Brasil. A Lotofácil consiste em sortear 15 números de um total de 25 possíveis (de 1 a 25), e o objetivo deste projeto é identificar padrões nos sorteios históricos e gerar previsões de combinações futuras com base em probabilidades.

# O código foi desenvolvido em Python e utiliza um modelo de Random Forest para prever números, combinado com uma interface interativa construída com Streamlit. Além disso, o projeto inclui análises estatísticas (média, mediana, moda, frequência) e visualizações gráficas para ajudar a entender os padrões dos sorteios.

# 🎯 Objetivo
O objetivo principal é:

# Analisar os números sorteados em concursos passados da Lotofácil.
Treinar um modelo de aprendizado de máquina para prever combinações de números com maior probabilidade de serem sorteadas.
Fornecer uma interface amigável para visualizar as previsões, estatísticas e padrões temporais dos números.
Embora as loterias sejam jogos de azar e os sorteios sejam aleatórios, este projeto explora padrões estatísticos e probabilísticos para fins educacionais e de curiosidade.

# 📊 Dados Utilizados
Os dados utilizados para treinar o modelo são números sorteados em concursos da Lotofácil, armazenados no arquivo sequencias_formatadas.txt. O formato do arquivo é o seguinte:

# Cada linha representa um sorteio.
Os números de cada sorteio estão no formato [n1, n2, ..., n15], onde n1 a n15 são os 15 números sorteados (valores entre 1 e 25).
Exemplo de linha: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 23, 24, 25].
Nota: O arquivo sequencias_formatadas.txt não está incluído neste repositório. Você deve fornecê-lo com os dados históricos da Lotofácil para que o projeto funcione. Esses dados podem ser obtidos em sites oficiais de loterias ou outras fontes confiáveis.

# 🛠️ Estrutura do Projeto
Arquivos
app.py: Código principal do projeto, contendo toda a lógica de carregamento de dados, treinamento do modelo, geração de previsões e interface com Streamlit.
sequencias_formatadas.txt: Arquivo de entrada com os dados históricos dos sorteios da Lotofácil (não incluído no repositório; deve ser fornecido pelo usuário).
README.md: Este arquivo, contendo a documentação do projeto.
Funcionalidades
Carregamento e Pré-processamento de Dados:
Os sorteios são lidos do arquivo sequencias_formatadas.txt.
Cada sorteio é transformado em um vetor de features, incluindo:
Uma matriz binária indicando quais números (de 1 a 25) foram sorteados.
Features adicionais: quantidade de números pares, ímpares e a soma dos números sorteados.
Treinamento do Modelo:
Um modelo RandomForestClassifier é treinado para prever a probabilidade de cada número (de 1 a 25) ser sorteado.
A otimização de hiperparâmetros é feita com GridSearchCV, testando diferentes valores para n_estimators, max_depth e min_samples_split.
Análise Estatística:
Cálculo de estatísticas como média, mediana, moda e frequência de ocorrência de cada número nos sorteios de treinamento.
Visualizações gráficas (histogramas) para cada métrica.
Geração de Previsões:
O modelo gera 10 variações de combinações de 15 números, ordenadas por probabilidade decrescente.
As probabilidades de cada número nas combinações são exibidas em gráficos de barras.
Padrões Temporais:
Análise da frequência com que cada número aparece ao longo dos sorteios, calculando o intervalo médio entre aparições.
Visualizações:
Gráficos de média, mediana, moda e frequência dos números.
Gráficos de probabilidade para cada variação prevista.
Heatmap mostrando a frequência de cada número ao longo dos sorteios.
# 📈 Exemplo de Saída
# A interface do Streamlit exibe as seguintes seções:

# Métricas do Modelo:
Acurácia do modelo no conjunto de teste.
Melhores hiperparâmetros encontrados pelo GridSearchCV.
Estatísticas dos Números:
Gráficos de média, mediana, moda e frequência de ocorrência dos números (de 1 a 25) no conjunto de treinamento.
Previsões dos Próximos Números:
10 variações de combinações de 15 números, exibidas como:
Variação 1: [2, 4, 5, 6, 8, 10, 11, 12, 13, 16, 18, 19, 20, 23, 24]
Gráficos de barras mostrando as probabilidades associadas a cada número nas variações.
Padrões Temporais:
Informações como: "Número 5 aparece a cada 3.2 sorteios em média."
Heatmap de Frequência:
Um heatmap visualizando a frequência de cada número ao longo dos sorteios.
