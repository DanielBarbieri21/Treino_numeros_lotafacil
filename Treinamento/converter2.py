# Abrir o arquivo com as 3206 linhas
with open('sequencias_formatadas.txt', 'r') as file:
    # Ler todas as linhas do arquivo original
    linhas = file.readlines()

# Inicializar uma lista para armazenar os arrays
arrays = []

# Processar cada linha, convertendo-a diretamente em uma lista de inteiros
for linha in linhas:
    # Remover espaços em branco ou quebras de linha
    linha = linha.strip()
    
    # Dividir a linha pelos separadores de vírgula e ignorar elementos vazios
    array = [int(numero) for numero in linha.split(',') if numero.strip()]
    
    # Adicionar o array à lista principal
    arrays.append(array)

# Exportar os arrays para um novo arquivo de texto
with open('sequencias_formatadas.txt', 'w') as output_file:
    for array in arrays:
        # Escrever cada array em uma nova linha no arquivo de saída
        output_file.write(f"{array}\n")

print("As sequências foram exportadas para 'sequencias_formatadas.txt'.")
