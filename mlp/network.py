import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, tam_camada_entrada, tam_camada_escondida, tam_camada_saida):
        self.tam_camada_entrada = tam_camada_entrada
        self.tam_camada_escondida = tam_camada_escondida
        self.tam_camada_saida = tam_camada_saida
        
        self.pesos_entrada_escondida = np.random.uniform(-1, 1, [self.tam_camada_entrada, self.tam_camada_escondida])
        np.save('pesos/pesos_camada_entrada_para_escondida.npy', self.pesos_entrada_escondida)
        
        self.pesos_escondida_saida = np.random.uniform(-1, 1, [self.tam_camada_escondida, self.tam_camada_saida])
        np.save('pesos/pesos_camada_escondida_para_saida.npy', self.pesos_escondida_saida)
        
        self.bias_camada_entrada_escondida = np.random.uniform(-1, 1)
        self.bias_camada_escondida_saida = np.random.uniform(-1, 1)
        
        np.save('pesos/pesos_dos_bias_entrada_escondida.npy', self.bias_camada_entrada_escondida)
        np.save('pesos/pesos_dos_bias_escondida_saida.npy', self.bias_camada_escondida_saida)
        
        self.taxa_de_aprendizagem = 0.5
    
    def treino(self, numero_de_epocas = 1000, vetor_dados_entrada = [], vetor_dados_saida = [], vetor_dados_entrada_validacao = [], vetor_dados_saida_validacao = []):
        erro_quadratico_medio_treinamento = []
        erro_quadratico_medico_validacao = []
        
        for epoca in range(numero_de_epocas):
            
            for index in range(len(vetor_dados_entrada)):
                vetor_avanco_calculado = self.feedForward(vetor_dados_entrada[index])
                self.backpropagation( vetor_dados_entrada[index], vetor_avanco_calculado, vetor_dados_saida[index])
                
            erro_quadratico_medio_treinamento.append(self.calculo_erro_quadratico_medio(vetor_dados_entrada, vetor_dados_saida)) 
            erro_quadratico_medico_validacao.append(
                self.calculo_erro_quadratico_medio(vetor_dados_entrada_validacao, vetor_dados_saida_validacao)
            )
        
        t = np.linspace(0, numero_de_epocas, numero_de_epocas)
        plt.plot(t, erro_quadratico_medio_treinamento, 'r')
        plt.plot(t, erro_quadratico_medico_validacao, 'b')
        plt.show()
    
    def calculo_erro_quadratico_medio(self, matriz_entrada, matriz_saida_esperada):
        total_erro = 0

        for vetor_entrada, vetor_saida_esperado in zip(matriz_entrada, matriz_saida_esperada):
            saida_rede = np.array(self.feedForward(vetor_entrada))
            erro_saida = np.array(vetor_saida_esperado) - saida_rede
            total_erro += np.sum(np.power(erro_saida, 2))

        return total_erro / len(matriz_entrada) 
                 
    def backpropagation(self, vetor_saida, vetor_saida_calculado, vetor_saida_esperada):
        # Erro na camada de saída
        vetor_erro = np.array(vetor_saida_esperada).reshape(26, 1) - np.array(vetor_saida_calculado)
        
        camada_saida_delta = vetor_erro * self.funcaoDerivadaSigmoid(self.calculo_avanco_escondida_saida)
        self.ajuste_pesos_escondida_saida = (
            self.calculo_saida_escondida.T * np.array(camada_saida_delta * self.taxa_de_aprendizagem).reshape(-1, 1)
        )
        
        camada_escondida_delta = (self.pesos_escondida_saida.dot(camada_saida_delta) * self.funcaoDerivadaSigmoid(self.calculo_avanco_entrada_escondida))
        self.ajuste_pesos_entrada_escondida = (
            vetor_saida * np.array(camada_escondida_delta * self.taxa_de_aprendizagem).reshape(-1, 1).T
        )
        
        
        self.pesos_escondida_saida = (
            self.pesos_escondida_saida + self.ajuste_pesos_escondida_saida.T
        )
        
        self.pesos_entrada_escondida = (
            self.pesos_entrada_escondida + self.ajuste_pesos_entrada_escondida
        )
    
    def funcaoSigmoid(self, valor):
        return 1 / (1 + np.exp(-valor))

    def funcaoDerivadaSigmoid(self, valor):
        return np.exp(-valor) / np.power(1 + np.exp(-valor), 2)               
    
    def feedForward(self, vetor):
        # Passagem da entrada para a camada escondida
        self.calculo_avanco_entrada_escondida = np.dot(self.pesos_entrada_escondida.T, vetor) + self.bias_camada_entrada_escondida
        self.calculo_saida_escondida = self.funcaoSigmoid(self.calculo_avanco_entrada_escondida)
        
        # Passagem da camada escondida para a camada de saída
        self.calculo_avanco_escondida_saida = np.dot(self.pesos_escondida_saida.T, self.calculo_saida_escondida) + self.bias_camada_escondida_saida
        self.vetor_saida = self.funcaoSigmoid(self.calculo_avanco_escondida_saida)
        
        return np.around(self.vetor_saida, 3)