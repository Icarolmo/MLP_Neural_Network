from mlp.network import Network
import numpy as np

dados_treinamento = np.load("X.npy")
dados_rotulos = np.load("Y_classe.npy")

matriz_dados = []

for dados in dados_treinamento:
    letra = dados.flatten().reshape(120, 1)
    letra[letra == -1] = 0 
    matriz_dados.append(letra)

network = Network(120, 60, 26)

print(network.feedForward(matriz_dados[1325]))

network.treino(
    numero_de_epocas=1000, 
    vetor_dados_entrada = matriz_dados[:858],
    vetor_dados_saida = dados_rotulos[:858],
    vetor_dados_entrada_validacao = matriz_dados[858:1196],
    vetor_dados_saida_validacao = dados_rotulos[858:1196]
    )

print(network.feedForward(matriz_dados[1325]))