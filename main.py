from rede_neural.mlp import RedeMLP
import numpy as np
import random
from datetime import datetime


# Carrega os dados e respectivos rotulos dos arquivos disponibilizados
dados_treinamento_e_teste = np.load("X.npy")
rotulos_dos_dados = np.load("Y_classe.npy")

# Ajusta vetor de dados para treinamento e teste para vetores unidimensionais de 120 posições
vetor_dos_dados = RedeMLP.converterVetorMultidimensional(dados_treinamento_e_teste)

# Criar um array de índices para Holdout
indices = list(range(len(vetor_dos_dados)))

# Embaralhar os índices
random.shuffle(indices)

# Reorganizar os arrays de dados e rótulos com base nos índices embaralhados randomicamente
vetor_embaralhados = [vetor_dos_dados[i] for i in indices]
rotulos_embaralhados = [rotulos_dos_dados[i] for i in indices]

# Calcular o tamanho dos arrays para partição dos dados de teste e treinamento
# neste caso realiza a partição 2/3 do conjunto para treinamento e 1/3 para teste
repartição = len(vetor_dos_dados) * 2 // 3

# Dividir os dados e rótulos
vetor_dos_dados_treinamento = vetor_embaralhados[:repartição]
rotulos_dos_dados_treinamento = rotulos_embaralhados[:repartição]
vetor_dos_dados_validacao = vetor_embaralhados[repartição:]
rotulos_dos_dados_validacao = rotulos_embaralhados[repartição:]

# Defini número de épocas, numéro de neurônios na camada escondida e a taxa de aprendizagem
numero_de_epocas = 200
tamanho_camada_escondida = 60
taxa_de_aprendizagem = 0.5

# Inicializa rede
rede = RedeMLP(120, tamanho_camada_escondida, 26, taxa_de_aprendizagem)

# Salva os pesos e bias inicias das camadas
rede.salvarPesos(diretorio="iniciais")

cross_validation = False

if cross_validation:
    # Inicializa treinamento da rede com cross-validation
    rede.iniciarTreinamentoComCrossValidation(
        vetor_dados_entrada = vetor_dos_dados_treinamento,
        vetor_dados_saida = rotulos_dos_dados_treinamento,
        numero_de_folds = 30
    )
else: 
    # Inicializa treinamento da rede apenas com holdout sem cross-validation
    rede.iniciarTreinamento(
        numero_de_epocas = numero_de_epocas, 
        vetor_dados_entrada = vetor_dos_dados_treinamento,
        vetor_dados_saida = rotulos_dos_dados_treinamento,
        vetor_dados_validacao = vetor_dos_dados_validacao,
        vetor_rotulos_validacao = rotulos_dos_dados_validacao
        )

# Salva os pesos e bias finais das camadas
rede.salvarPesos(diretorio="finais")

# Salva as configurações da rede na rodada atual
np.save(f'log/configuracao/configuracao_da_rede-{datetime.now().strftime("%H-%M-%S_%d-%m-%Y")}.npy', [numero_de_epocas, tamanho_camada_escondida, taxa_de_aprendizagem])

# Plota o gráfico com o erro quadratico médio dos dados de treinamento e teste durante o treinamento conforme as épocas
rede.plotarErroQuadraticoMedio(numero_de_epocas = numero_de_epocas)

