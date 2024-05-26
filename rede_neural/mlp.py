import numpy as np
import matplotlib.pyplot as plt

# RedeMLP classe que representa uma rede MLP com as estruturas e métodos necessários para funcionamento
class RedeMLP:

    # __init__ método construtor da classe RedeMLP que recebe o número de neurônios de cada camada e a 
    # taxa de aprendizagem e inicializa os pesos aleatoriamente. Por default, caso não seja passado nenhum 
    # valor especifico, temos 120 neurônios na camada de entrada, 30 na camada escondida e 26 na camada 
    # de saida e 0.5 sendo a taxa de aprendizagem
    def __init__(self, tam_camada_entrada=120, tam_camada_escondida=30, tam_camada_saida=26, taxa_de_aprendizagem=0.5):
        self.tam_camada_entrada = tam_camada_entrada
        self.tam_camada_escondida = tam_camada_escondida
        self.tam_camada_saida = tam_camada_saida
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        
        self.pesos_entrada_escondida = np.random.uniform(-1, 1, [self.tam_camada_entrada, self.tam_camada_escondida])
        self.pesos_escondida_saida = np.random.uniform(-1, 1, [self.tam_camada_escondida, self.tam_camada_saida])
        self.bias_camada_entrada_escondida = np.random.uniform(-1, 1)
        self.bias_camada_escondida_saida = np.random.uniform(-1, 1) 
        
        self.erro_quadratico_medio_treinamento = []
        self.erro_quadratico_medio_validacao = []
        
        
    # salvarPesos utilizado para salvar no diretorio "log" os pesos iniciais e finais das camadas
    def salvarPesos(self, diretorio):
        np.save(f'log/{diretorio}/pesos-camada_entrada_para_escondida.npy', self.pesos_entrada_escondida)
        np.save(f'log/{diretorio}/pesos-camada_escondida_para_saida.npy', self.pesos_escondida_saida)
        np.save(f'log/{diretorio}/bias-camada_entrada_para_escondida.npy', self.bias_camada_entrada_escondida)
        np.save(f'log/{diretorio}/bias-camada_escondida_para_saida.npy', self.bias_camada_escondida_saida)
    
    # iniciarTreinamento recebe o número de epocas e os vetores de treinamento e testes junto a seus respectivos
    # rótulos e realiza o treinamento da rede utilizando para passagem das camadas feedFoward com a função de 
    # ativação Sigmoid, calcula o erro e ajusta com backpropagation.
    def iniciarTreinamentoComCrossValidation(self,
                           vetor_dados_entrada = [], 
                           vetor_dados_saida = [],
                           numero_de_folds = 10
                           ):
        
        # particiona os vetores conforme o numero de folds definido
        vetores_particionados = np.array_split(vetor_dados_entrada, numero_de_folds)
        rotulos_particionados = np.array_split(vetor_dados_saida, numero_de_folds)
        
        # itera sobre os vetores particionados
        for i in range(len(vetores_particionados)): 
            # para cada particao itera sobre o conjunto de vetores com dados daquela partição
            for j in range(len(vetores_particionados[i])): 
                if i == j: continue # verifica se a particao atual é a partição de teste, se for pula para próxima
                vetor = vetores_particionados[i][j]
                rotulo = rotulos_particionados[i][j]
                
                # realiza o feed forward e backpropagation para a particao atual
                vetor_avanco_calculado = self.feedForward(vetor)
                self.backpropagation(vetor, vetor_avanco_calculado, rotulo)

            self.erro_quadratico_medio_validacao.append(self.estimarErroQuadraticoMedio(vetores_particionados[i], rotulos_particionados[i]))
        
        self.plotarErroQuadraticoMedioValidacao(numero_de_epocas=numero_de_folds)
                
    # iniciarTreinamento recebe o número de epocas e os vetores de treinamento e testes junto a seus respectivos
    # rótulos e realiza o treinamento da rede utilizando para passagem das camadas feedFoward com a função de 
    # ativação Sigmoid, calcula o erro e ajusta com backpropagation.    
    def iniciarTreinamento(self, 
                           numero_de_epocas = 100, 
                           vetor_dados_entrada = [], 
                           vetor_dados_saida = [],
                           vetor_dados_validacao = [],
                           vetor_rotulos_validacao = []
                           ):
        
        # iterar o número de epocas
        for epoca in range(numero_de_epocas):
            
            # para cada epoca itera sobre o vetor de dados e realiza o treinamento com feed foward e backpropagation
            for index in range(len(vetor_dados_entrada)):
                vetor_avanco_calculado = self.feedForward(vetor_dados_entrada[index])
                self.backpropagation( vetor_dados_entrada[index], vetor_avanco_calculado, vetor_dados_saida[index])
            
            # calcula o erro quadratico medio para o conjunto de treinamento e teste para a epoca atual
            self.erro_quadratico_medio_treinamento.append(self.estimarErroQuadraticoMedio(vetor_dados_entrada, vetor_dados_saida))
            self.erro_quadratico_medio_validacao.append(self.estimarErroQuadraticoMedio(vetor_dados_validacao, vetor_rotulos_validacao))
            
            if epoca % 99 == 0:
                print("Epoca: ", epoca + 1)
                
    
    # plotarErroQuadraticoMedio plota um gráfico com erro quadratico médio dos dados de treinamento e validacao conforme o treinamento durante as épocas
    def plotarErroQuadraticoMedio(self, numero_de_epocas):
        t = np.linspace(0, numero_de_epocas, numero_de_epocas)
        plt.plot(t, self.erro_quadratico_medio_treinamento, 'r', label='Treinamento')
        plt.plot(t, self.erro_quadratico_medio_validacao, 'y', label='Validação')
        plt.legend()
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio')
        plt.show()
        
    # plotarErroQuadraticoMedioValidacao plota um gráfico com erro quadratico médio conforme o treinamento durante as épocas
    def plotarErroQuadraticoMedioValidacao(self, numero_de_epocas):
        t = np.linspace(0, numero_de_epocas, numero_de_epocas)
        plt.plot(t, self.erro_quadratico_medio_validacao, 'y', label='Validação')
        plt.legend()
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio')
        plt.show()
        
        
    # estimarErroQuadraticoMedio estima o erro quadrático médio entre a saída prevista e a esperada
    def estimarErroQuadraticoMedio(self, matriz_entrada, matriz_saida_esperada):
        total_erro = 0

        # Loop sobre cada amostra do conjunto de dados
        for vetor_entrada, vetor_saida_esperado in zip(matriz_entrada, matriz_saida_esperada):
            saida_rede = np.array(self.feedForward(vetor_entrada))
            erro_saida = np.array(vetor_saida_esperado) - saida_rede
            total_erro += np.sum(np.power(erro_saida, 2))

        return total_erro / len(matriz_entrada) 
    
    # estimarAcuracia estima a acurácia do modelo comparando as saídas previstas com as esperadas          
    def estimarAcuracia(self, matriz_entrada, matriz_saida_esperada):
        total_acertos = 0
        
        # Loop sobre cada amostra do conjunto de dados
        for vetor_entrada, vetor_saida_esperado in zip(matriz_entrada, matriz_saida_esperada):
            saida_rede = np.array(self.feedForward(vetor_entrada))
            if np.argmax(saida_rede) == np.argmax(vetor_saida_esperado):
                total_acertos += 1
        
        return (total_acertos / len(matriz_entrada)) * 100  
    
    # backpropagation realiza o ajuste dos pesos com base no erro calculado entre a saída prevista e a esperada
    def backpropagation(self, vetor_saida, vetor_saida_calculado, vetor_saida_esperada):
        # Erro na camada de saída
        vetor_erro = np.array(vetor_saida_esperada) - np.array(vetor_saida_calculado)
        
        # Calculo do delta da camada de saída
        camada_saida_delta = vetor_erro * self.funcaoDerivadaSigmoid(self.calculo_avanco_escondida_saida)
        self.ajuste_pesos_escondida_saida = (
            self.calculo_saida_escondida.T * np.array(camada_saida_delta * self.taxa_de_aprendizagem).reshape(-1, 1)
        )
        
        # Calculo do delta da camada escondida
        camada_escondida_delta = (self.pesos_escondida_saida.dot(camada_saida_delta).T * self.funcaoDerivadaSigmoid(self.calculo_avanco_entrada_escondida))
        self.ajuste_pesos_entrada_escondida = (
            np.array(vetor_saida).reshape(-1, 1) * np.array(camada_escondida_delta * self.taxa_de_aprendizagem).reshape(-1, 1).T
        )
        
        # Atualização dos pesos
        self.pesos_escondida_saida = (
            self.pesos_escondida_saida + self.ajuste_pesos_escondida_saida.T
        )
        self.pesos_entrada_escondida = (
            self.pesos_entrada_escondida + self.ajuste_pesos_entrada_escondida
        )
    
    # funcaoSigmoid calcula a ativação Sigmoid
    def funcaoSigmoid(self, valor):
        return 1 / (1 + np.exp(-valor))

    # funcaoDerivadaSigmoid calcula a derivada da função de ativação Sigmoid
    def funcaoDerivadaSigmoid(self, valor):
        return np.exp(-valor) / np.power(1 + np.exp(-valor), 2)               
    
    # feedForward realiza a passagem dos dados de entrada pela rede até a saída
    def feedForward(self, vetor):
        # Passagem da entrada para a camada escondida
        self.calculo_avanco_entrada_escondida = np.dot(self.pesos_entrada_escondida.T, vetor) + self.bias_camada_entrada_escondida
        self.calculo_saida_escondida = self.funcaoSigmoid(self.calculo_avanco_entrada_escondida)
        
        # Passagem da camada escondida para a camada de saída
        self.calculo_avanco_escondida_saida = np.dot(self.pesos_escondida_saida.T, self.calculo_saida_escondida) + self.bias_camada_escondida_saida
        self.vetor_saida = self.funcaoSigmoid(self.calculo_avanco_escondida_saida)
        
        return np.around(self.vetor_saida, 3)
            
            
    
    # converterVetorMultidimensional recebe vetor de dados de treinamento e transforma em vetor bidimensional
    @staticmethod
    def converterVetorMultidimensional(vetor_multidimensional):
        matriz_dados = []
        
        for dados in vetor_multidimensional:
            # Transforma-se o array complexo (cada elemento do arquivo_x - uma letra) em um array unidimensional, por meio da função flatten
            letra = dados.flatten()
            letra[letra == -1] = 0 # Substitui-se os valores -1 por 0
            matriz_dados.append(letra)

        return matriz_dados