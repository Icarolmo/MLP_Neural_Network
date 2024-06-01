from rede_neural.mlp import RedeMLP
import numpy as np
import random
from datetime import datetime

# Carrega os dados e respectivos rotulos dos arquivos disponibilizados
dados_treinamento_e_teste = np.load("dados/X.npy")
rotulos_dos_dados = np.load("dados/Y_classe.npy")

# Ajusta vetor de dados para treinamento e teste para vetores unidimensionais de 120 posições
vetor_dos_dados = RedeMLP.converterVetorMultidimensional(dados_treinamento_e_teste)

# Criar um array de índices para Holdout
indices = list(range(len(vetor_dos_dados)))

# Embaralhar os índices
random.shuffle(indices)

# Reorganizar os arrays de dados e rótulos com base nos índices embaralhados randomicamente
vetor_dos_dados_embaralhados = [vetor_dos_dados[i] for i in indices]
rotulos_dos_dados_embaralhados = [rotulos_dos_dados[i] for i in indices]

# Calcular o tamanho do conjunto de dados para partição dos dados de teste e treinamento
# neste caso pega 80% do conjunto de dados para treinamento e armazera 20% para teste
numero_de_dados =  (len(vetor_dos_dados_embaralhados) // 10) * 8

# Dividir os dados e rótulos
vetor_dos_dados_treinamento = vetor_dos_dados_embaralhados[:numero_de_dados]
rotulos_dos_dados_treinamento = rotulos_dos_dados_embaralhados[:numero_de_dados]
vetor_dos_dados_teste = vetor_dos_dados_embaralhados[numero_de_dados:]
rotulos_dos_dados_teste = rotulos_dos_dados_embaralhados[numero_de_dados:]

# Define número de épocas, numéro de neurônios na camada escondida e a taxa de aprendizagem
numero_de_epocas = 50
tamanho_camada_escondida = 35
taxa_de_aprendizagem = 0.5

# Inicializa treinamento sem cross validation
redeSemValicaoCruzada = RedeMLP(120, tamanho_camada_escondida, 26, taxa_de_aprendizagem)

print("\n------------------------ INICIANDO TREINAMENTO SEM VALIDAÇÃO CRUZADA ------------------------\n")

# Salva pesos iniciais da rede sem validacao cruzada
redeSemValicaoCruzada.salvarPesos(
    diretorio="treinamento_sem_validacao_cruzada",
    tipo_pesos="iniciais",
    rodada="",
    acuracia=""
)

# Inicia treinamento da rede
redeSemValicaoCruzada.iniciarTreinamento(
    numero_de_epocas = numero_de_epocas,
    vetor_dados_entrada = vetor_dos_dados_treinamento[:(len(vetor_dos_dados_treinamento) // 10) * 8],
    vetor_dados_saida = rotulos_dos_dados_treinamento[:(len(rotulos_dos_dados_treinamento) // 10) * 8],
    vetor_dados_validacao = vetor_dos_dados_treinamento[(len(vetor_dos_dados_treinamento) // 10) * 8:],
    vetor_rotulos_validacao = rotulos_dos_dados_treinamento[(len(rotulos_dos_dados_treinamento) // 10) * 8:]
    )

# Estima a acuracia da rede após treinada
acuracia = redeSemValicaoCruzada.estimarAcuracia(vetor_dos_dados_teste, rotulos_dos_dados_teste)

# Calcula e plota o gráfico com erro quadratico medio da rede com dados de treinamento e validacao
redeSemValicaoCruzada.plotarErroQuadraticoMedio(numero_de_epocas=numero_de_epocas)

# Plota a matriz de confusão para os dados de teste
redeSemValicaoCruzada.plotarMatrizDeConfusao(
    vetor_dos_dados_teste, 
    rotulos_dos_dados_teste,
    "treinamento_sem_validacao_cruzada",
    "",
    f"{acuracia:.2f}"
    )

# Salva os pesos finais da rede
redeSemValicaoCruzada.salvarPesos(
    diretorio="treinamento_sem_validacao_cruzada",
    tipo_pesos="finais",
    rodada="",
    acuracia=f"{acuracia:.2f}"
)

print("\n------------------------ INICIANDO TREINAMENTO COM VALIDAÇÃO CRUZADA ------------------------\n")

# Define número de folds para treinamento com validacao cruzada
numero_de_folds = 10

# Particiona os vetores conforme o numero de folds definido
vetores_treinamento_particionados = np.array_split(vetor_dos_dados_treinamento, numero_de_folds)
rotulos_treinamento_particionados = np.array_split(rotulos_dos_dados_treinamento, numero_de_folds)

# Inicializa treinamento com validação cruzada iterando sobre o numero de folds
for i in range(numero_de_folds):
    
    # Inicializa rede para configuração de folds atual
    rede = RedeMLP(120, tamanho_camada_escondida, 26, taxa_de_aprendizagem)
    
    # Salva os pesos iniciais da configuracao atual
    rede.salvarPesos(
        diretorio="treinamento_com_validacao_cruzada",
        tipo_pesos="iniciais",
        rodada=i+1,
        acuracia="",
        )
    
    print(f"\nInicio do treinamento com validacao cruzada: fold de validação nº{i+1} de {numero_de_folds} folds\n")
    
    # Divide os folds de treinamento e o de validacao em uma proporcao de 9 folds para treinamento e 1 para validacao
    fold_de_validacao = vetores_treinamento_particionados[i]
    # Faz a união dos folds de treinamento retirando do conjunto deles o folds para validacao
    folds_de_treinamento = RedeMLP.unirFolds(
        RedeMLP.retirarVetorDaLista(vetores_treinamento_particionados, i)
        )
    rotulos_fold_de_validacao = rotulos_treinamento_particionados[i]
    rotulos_folds_de_treinamento = RedeMLP.unirFolds(
        RedeMLP.retirarVetorDaLista(rotulos_treinamento_particionados, i)
        )
    
    # Para cada configuração de folds itera sobre o numero de epocas e realiza o treinamento
    for epoca in range(numero_de_epocas):
        # Para cada epoca itera sobre os folds de treinamento realizando o treinamento (feed foward e backpropagation)
        for k in range(len(folds_de_treinamento)):
            # realiza o feed forward e backpropagation 
            vetor_avanco_calculado = rede.feedForward(folds_de_treinamento[k])
            rede.backpropagation(folds_de_treinamento[k], vetor_avanco_calculado, rotulos_folds_de_treinamento[k])

        # Calcula erro quadratico medio a cada epoca para conjuto de treinamento e de validacao
        rede.erro_quadratico_medio_treinamento.append(rede.estimarErroQuadraticoMedio(folds_de_treinamento, rotulos_folds_de_treinamento))
        rede.erro_quadratico_medio_validacao.append(rede.estimarErroQuadraticoMedio(fold_de_validacao, rotulos_fold_de_validacao))    
    
    # Calcula a acuraria para esta configuracao da rede treinada com o conjunto de dados de testes (20% do conjunto total de dados) separado no inicio do código 
    acuracia = rede.estimarAcuracia(vetor_dos_dados_teste, rotulos_dos_dados_teste)
    
    print(f"fim do treinamento: folds de validação nº{i+1}. acurácia da rede:{acuracia}\n")
    
    # Plota o gráfico com o erro quadratico médio dos dados de treinamento e teste durante o treinamento conforme as épocas
    rede.plotarErroQuadraticoMedio(numero_de_epocas = numero_de_epocas)
    
    # Plota a matriz de confusão da rede
    rede.plotarMatrizDeConfusao(
        vetor_dos_dados_teste,
        rotulos_dos_dados_teste,
        "treinamento_com_validacao_cruzada",
        f"{i+1}",
        f"{acuracia:.2f}"
    )
    
    # Salva os pesos da configuracao atual
    rede.salvarPesos(
        diretorio="treinamento_com_validacao_cruzada",
        tipo_pesos="finais",
        rodada=i+1,
        acuracia=f"{acuracia:.2f}",
        )