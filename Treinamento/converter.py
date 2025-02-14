# Abra o arquivo para leitura
with open('treino.txt', 'r') as file:
    # Crie uma lista para armazenar as sequências
    sequencias = []
    
    # Leia cada linha do arquivo
    for linha in file:
        # Remova espaços em branco e quebre a linha em números
        numeros = linha.strip().split()
        
        # Junte os números com vírgulas e adicione à lista
        sequencias.append(','.join(numeros))

# Salvar as sequências em um novo arquivo
with open('sequencias_formatadas.txt', 'w') as output_file:
    for seq in sequencias:
        output_file.write(seq + '\n')