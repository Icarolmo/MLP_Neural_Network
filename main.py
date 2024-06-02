from itertools import product
import numpy as np
from rede_neural.mlp import RedeMLP

# Define número de épocas, numéro de neurônios na camada escondida, numero de folds, taxa de aprendizagem e acuracia mínima para parada antecipada
configuracao = {
    'camadas': [120, 50, 26],
    'taxa_de_aprendizagem': 0.5,
    'numero_de_epocas': 50,
    'numero_de_folds': 5,
    'acuracia_min_para_parada_antecipada': 86
}

# limpa ou cria as pastas para armazenamento dos pesos e matrizes de confusão
RedeMLP.criarLimparPastasConfiguracao(configuracao['numero_de_folds'])

# Carrega os dados e respectivos rotulos dos arquivos disponibilizados
dados_carregados = np.load("dados/X.npy")
rotulos = np.load("dados/Y_classe.npy")

# Ajusta vetor de dados para treinamento e teste para vetores unidimensionais de 120 posições
dados = RedeMLP.converterVetorMultidimensional(dados_carregados)

# Realiza holdout dos dados embaralhandos aleatoriamente e dividindo 80% para treinamento e 20% para teste
dados_treinamento, rotulos_treinamento, dados_teste, rotulos_teste = RedeMLP.holdout(dados, rotulos)


print("\n------------------------ Treinamento 1: sem validação cruzada ou parada antecipada ------------------------\n")

print("\n----------Configurações paras as redes-------------\n")
for chave, valor in configuracao.items():
    print(f"{chave} : {valor}")


# Inicializa treinamento sem validacao cruzada e parada antecipada
redeSemValicaoCruzada = RedeMLP(120, configuracao["camadas"][1], 26, configuracao["taxa_de_aprendizagem"])

# Salva pesos iniciais da rede sem validacao cruzada
redeSemValicaoCruzada.salvarPesos(
    diretorio="treinamento_sem_validacao_cruzada/iniciais",
)

# Inicia treinamento da rede e divide os dados de treinamento e validação em uma proporção de 80% para treinamento e 20% validacao
redeSemValicaoCruzada.iniciarTreinamento(
    numero_de_epocas = configuracao["numero_de_epocas"],
    dados_treinamento = dados_treinamento[:(len(dados_treinamento) // 10) * 8],
    rotulos_treinamento = rotulos_treinamento[:(len(rotulos_treinamento) // 10) * 8],
    dados_validacao = dados_treinamento[(len(dados_treinamento) // 10) * 8:],
    rotulos_validacao = rotulos_treinamento[(len(rotulos_treinamento) // 10) * 8:]
    )

# Estima a acuracia da rede após treinada
acuracia = redeSemValicaoCruzada.estimarAcuracia(dados_teste, rotulos_teste)

numero_de_epocas = configuracao["numero_de_epocas"]

print(f"\nfim do treinamento: número de epocas percorridas: {numero_de_epocas}, acurácia da rede: {acuracia} \n")

# Plota a matriz de confusão para os dados de teste e a salva no formato excel
redeSemValicaoCruzada.plotarMatrizDeConfusao(
    dados_teste, 
    rotulos_teste,
    "treinamento_sem_validacao_cruzada/finais",
    )

# Calcula e plota o gráfico com erro quadratico medio da rede com dados de treinamento e validacao
redeSemValicaoCruzada.plotarErroQuadraticoMedio(numero_de_epocas=configuracao["numero_de_epocas"])

# Salva os pesos finais da rede
redeSemValicaoCruzada.salvarPesos(
    diretorio="treinamento_sem_validacao_cruzada/finais",
)

print("\n------------------------ Treinamento 2: com validação cruzada e parada antecipada ------------------------\n")

print("\n----------Configurações paras as redes-------------\n")
for chave, valor in configuracao.items():
    print(f"{chave} : {valor}")

numero_de_folds = configuracao["numero_de_folds"]

# Particiona os vetores conforme o numero de folds definido
dados_treinamento_particionados = np.array_split(dados_treinamento, numero_de_folds)
rotulos_treinamento_particionados = np.array_split(rotulos_treinamento, numero_de_folds)

# Inicializa treinamento com validação cruzada iterando sobre o numero de folds
for i in range(numero_de_folds):
    
    # Inicializa rede para configuração de folds atual
    rede = RedeMLP(configuracao["camadas"][0], configuracao["camadas"][1], configuracao["camadas"][2], configuracao["taxa_de_aprendizagem"])
    
    # Salva os pesos iniciais da configuracao atual
    rede.salvarPesos(
        diretorio=f"treinamento_com_validacao_cruzada/config_folds_{i+1}/iniciais",
        )
    
    print(f"\nInicio do treinamento com validacao cruzada: fold de validação nº{i+1} de {numero_de_folds} folds\n")
    
    # Divide os folds de treinamento e o de validacao em uma proporcao de n-1 folds para treinamento e 1 para validacao
    fold_de_validacao = dados_treinamento_particionados[i]
    # Faz a união dos folds de treinamento retirando do conjunto deles o folds para validacao
    folds_de_treinamento = RedeMLP.unirFolds(
        RedeMLP.retirarVetorDaLista(dados_treinamento_particionados, i)
        )
    rotulos_fold_de_validacao = rotulos_treinamento_particionados[i]
    rotulos_folds_de_treinamento = RedeMLP.unirFolds(
        RedeMLP.retirarVetorDaLista(rotulos_treinamento_particionados, i)
        )
    
    numero_de_epocas = configuracao["numero_de_epocas"]
    
    # Para cada configuração de folds itera sobre o numero de epocas e realiza o treinamento
    for epoca in range(numero_de_epocas):
        # Para cada epoca itera sobre os folds de treinamento realizando o treinamento (feed forward e backpropagation)
        for k in range(len(folds_de_treinamento)):
            # realiza o feed forward e backpropagation 
            vetor_avanco_calculado = rede.feedForward(folds_de_treinamento[k])
            rede.backpropagation(folds_de_treinamento[k], vetor_avanco_calculado, rotulos_folds_de_treinamento[k])

        # Calcula erro quadratico medio a cada epoca para conjuto de treinamento e de validacao
        rede.erro_quadratico_medio_treinamento.append(
            rede.estimarErroQuadraticoMedio(folds_de_treinamento, rotulos_folds_de_treinamento)
            )
        rede.erro_quadratico_medio_validacao.append(
            rede.estimarErroQuadraticoMedio(fold_de_validacao, rotulos_fold_de_validacao)
            )  
        
        # Calcula a acuracia encima do conjunto de validacao para verificar parada antecipada
        acuracia_para_antecipada = rede.estimarAcuracia(fold_de_validacao, rotulos_fold_de_validacao)
        
        if acuracia_para_antecipada > configuracao['acuracia_min_para_parada_antecipada']:
            numero_de_epocas = epoca + 1
            break
    
    # Calcula a acuraria para esta configuracao da rede treinada com o conjunto de dados de testes (20% do conjunto total de dados) separado no inicio do código 
    acuracia = rede.estimarAcuracia(dados_teste, rotulos_teste)
    
    print(f"fim do treinamento: fold de validação nº{i+1}, número de epocas percorridas: {numero_de_epocas}, acurácia da rede:{acuracia}\n")
    
    # Plota o gráfico com o erro quadratico médio dos dados de treinamento e teste durante o treinamento conforme as épocas
    rede.plotarErroQuadraticoMedio(numero_de_epocas = numero_de_epocas)
    
    # Plota a matriz de confusão da rede
    rede.plotarMatrizDeConfusao(
        dados_teste,
        rotulos_teste,
        f"treinamento_com_validacao_cruzada/config_folds_{i+1}/finais",
    )
    
    # Salva os pesos da configuracao atual
    rede.salvarPesos(
        diretorio=f"treinamento_com_validacao_cruzada/config_folds_{i+1}/finais",
        )


# Realiza o Grid Search
melhor_acuracia = 0
melhores_parametros = None

# Define o range dos parametros a ser procurado a combinacao de melhor acuracia
neuronios_camada_escondida_range = range(35, 40, 1)
taxa_aprendizado_range = np.arange(0.3, 0.5, 0.1)
numero_de_epocas_range = range(5, 15, 1)
numero_folds_range = range(5, 8, 1)

print("\n---------------------------------- Iniciando Grid Search ----------------------------------\n")

# Printa o número de combinações a serem verificadas
numero_de_combinacoes =  len(neuronios_camada_escondida_range) *  len(taxa_aprendizado_range) * len(numero_de_epocas_range) * len(numero_folds_range)
print(f"Numero de combinações a serem verificadas: {numero_de_combinacoes}")

contador = 0

# Itera sobre todas as combinações pegando cada uma das possiveis
for tam_camada_escondida, taxa_aprendizado, epocas, numero_de_folds in product(
    neuronios_camada_escondida_range, 
    taxa_aprendizado_range, 
    numero_de_epocas_range, 
    numero_folds_range
    ):
    
    # printa combinação atual a ser calculada
    contador += 1
    print(f"Verificando combinação de número: {contador}...")
    
    # Particiona os vetores conforme o numero de folds definido
    dados_treinamento_particionados = np.array_split(dados_treinamento, numero_de_folds)
    rotulos_treinamento_particionados = np.array_split(rotulos_treinamento, numero_de_folds)

    # Inicializa treinamento com validação cruzada iterando sobre o numero de folds
    for i in range(numero_de_folds):
        
        # Inicializa rede para configuração de folds atual
        rede = RedeMLP(configuracao["camadas"][0], tam_camada_escondida, configuracao["camadas"][2], taxa_aprendizado)
        
        # Divide os folds de treinamento e o de validacao em uma proporcao de 9 folds para treinamento e 1 para validacao
        fold_de_validacao = dados_treinamento_particionados[i]
        # Faz a união dos folds de treinamento retirando do conjunto deles o folds para validacao
        folds_de_treinamento = RedeMLP.unirFolds(
            RedeMLP.retirarVetorDaLista(dados_treinamento_particionados, i)
            )
        rotulos_fold_de_validacao = rotulos_treinamento_particionados[i]
        rotulos_folds_de_treinamento = RedeMLP.unirFolds(
            RedeMLP.retirarVetorDaLista(rotulos_treinamento_particionados, i)
            )
        
        numero_de_epocas = epocas
        
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
            
            # Calcula a acuracia encima do conjunto de validacao para verificar parada antecipada
            acuracia_para_antecipada = rede.estimarAcuracia(fold_de_validacao, rotulos_fold_de_validacao)
            
            if acuracia_para_antecipada > configuracao['acuracia_min_para_parada_antecipada']:
                numero_de_epocas = epoca + 1
                break
        
        # Calcula a acuraria para esta configuracao da rede treinada com o conjunto de dados de testes (20% do conjunto total de dados) separado no inicio do código 
        acuracia = rede.estimarAcuracia(dados_teste, rotulos_teste)
        
        # Verifica se a acuracia da configuracao da rede atual é melhor que a atual melhor acuracia, se for salva os parametros e a acuracia desta configuracao
        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhores_parametros = (tam_camada_escondida, taxa_aprendizado, epocas, numero_de_folds)

# Printa esta configuracao e a acuracia da rede
print(f'Melhor Acurácia: {melhor_acuracia}')
print(f'Parametros da melhor acurácia: Camada escondida={melhores_parametros[0]},Taxa de aprendizado={melhores_parametros[1]}, Epocas={melhores_parametros[2]}, Numero de folds={melhores_parametros[3]}')
    

