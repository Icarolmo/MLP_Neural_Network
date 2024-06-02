import os
import random
import shutil
import string
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# RedeMLP classe que representa uma rede MLP com as estruturas e métodos necessários para funcionamento
class RedeMLP:

    # __init__ método construtor da classe RedeMLP que recebe o número de neurônios de cada camada e a 
    # taxa de aprendizagem e inicializa os pesos aleatoriamente.
    def __init__(self, tam_camada_entrada=120, tam_camada_escondida=55, tam_camada_saida=26, taxa_de_aprendizagem=0.5):
        self.tam_camada_entrada = tam_camada_entrada
        self.tam_camada_escondida = tam_camada_escondida
        self.tam_camada_saida = tam_camada_saida
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        
        # Inicializando os pesos aleatoriamente com valores no intervalo de -1 a 1
        self.pesos_entrada_escondida = np.random.uniform(-1, 1, [self.tam_camada_entrada, self.tam_camada_escondida])
        self.pesos_escondida_saida = np.random.uniform(-1, 1, [self.tam_camada_escondida, self.tam_camada_saida])
        self.bias_camada_entrada_escondida = np.random.uniform(-1, 1)
        self.bias_camada_escondida_saida = np.random.uniform(-1, 1) 
        
        self.erro_quadratico_medio_treinamento = []
        self.erro_quadratico_medio_validacao = []
    
    
    # salvarPesos salva os pesos dos neurônios e bias no respectivo diretório passado por parâmetro    
    def salvarPesos(self, diretorio):
        np.save(f'log/configuracao/{diretorio}/pesos_camada_entrada_para_escondida.npy', self.pesos_entrada_escondida)
        np.save(f'log/configuracao/{diretorio}/pesos_camada_escondida_para_saida.npy', self.pesos_escondida_saida)
        np.save(f'log/configuracao/{diretorio}/pesos_bias_camada_entrada_para_escondida.npy', self.bias_camada_entrada_escondida)
        np.save(f'log/configuracao/{diretorio}/pesos_bias_camada_escondida_para_saida.npy', self.bias_camada_escondida_saida)
    
    # iniciarTreinamento recebe o número de epocas e os vetores de treinamento e testes junto a seus respectivos
    # rótulos e realiza o treinamento da rede sem validação cruzada e parada antecipada utilizando como função de
    # ativação a Sigmoid, feed foward para passagem dos valores e backpropagation para ajuste dos pesos. 
    def iniciarTreinamento(self, 
                           numero_de_epocas = 50, 
                           dados_treinamento = [], 
                           rotulos_treinamento = [],
                           dados_validacao = [],
                           rotulos_validacao = []
                           ):
        
        # iterar o número de epocas
        for epoca in range(numero_de_epocas):
            
            # para cada epoca itera sobre o vetor de dados e realiza o treinamento com feed foward e backpropagation
            for index in range(len(dados_treinamento)):
                rotulo_previsto = self.feedForward(dados_treinamento[index])
                self.backpropagation(dados_treinamento[index], rotulo_previsto, rotulos_treinamento[index])
            
            # calcula o erro quadratico medio para o conjunto de treinamento e teste para a epoca atual
            self.erro_quadratico_medio_treinamento.append(
                self.estimarErroQuadraticoMedio(dados_treinamento, rotulos_treinamento)
                )
            self.erro_quadratico_medio_validacao.append(
                self.estimarErroQuadraticoMedio(dados_validacao, rotulos_validacao)
                )
            
    
    # backpropagation realiza o ajuste dos pesos com base no erro calculado entre a saída prevista e a esperada
    def backpropagation(self, vetor_saida, rotulo_previsto, rotulo_esperado):
        # Erro do rotulo previsto na camada de saida
        erro = np.array(rotulo_esperado) - np.array(rotulo_previsto)
        
        # Calculo do delta da camada de saída e calculo do ajuste dos pesos da escondida para saida
        delta_da_camada_saida = erro * self.funcaoDerivadaSigmoid(self.calculo_avanco_escondida_saida)
        self.ajuste_pesos_escondida_saida = (self.calculo_saida_escondida.T * np.array(delta_da_camada_saida * self.taxa_de_aprendizagem).reshape(-1, 1))
        
        # Calculo do delta da camada escondida e calculo do ajuste dos pesos da entrada para escondida
        delta_da_camada_escondida = (self.pesos_escondida_saida.dot(delta_da_camada_saida).T * self.funcaoDerivadaSigmoid(self.calculo_avanco_entrada_escondida))
        self.ajuste_pesos_entrada_escondida = (np.array(vetor_saida).reshape(-1, 1) * np.array(delta_da_camada_escondida * self.taxa_de_aprendizagem).reshape(-1, 1).T)
        
        # Atualização dos pesos
        self.pesos_escondida_saida = (self.pesos_escondida_saida + self.ajuste_pesos_escondida_saida.T) 
        self.pesos_entrada_escondida = (self.pesos_entrada_escondida + self.ajuste_pesos_entrada_escondida)
        
        # Atualização dos bias
        self.bias_camada_escondida_saida = self.bias_camada_escondida_saida + self.taxa_de_aprendizagem * delta_da_camada_saida
        self.bias_camada_entrada_escondida = self.bias_camada_entrada_escondida + self.taxa_de_aprendizagem * delta_da_camada_escondida
    
    # funcaoSigmoid utilizada como função de ativação dos neurônios
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
        
        return np.around(self.vetor_saida, 5)
    
    # plotarErroQuadraticoMedio plota um gráfico com erro quadratico médio dos dados de treinamento e validacao conforme o treinamento durante as épocas
    def plotarErroQuadraticoMedio(self, numero_de_epocas):
        t = np.linspace(0, numero_de_epocas, numero_de_epocas)
        plt.plot(t, self.erro_quadratico_medio_treinamento, 'r', label='Treinamento')
        plt.plot(t, self.erro_quadratico_medio_validacao, 'y', label='Validação')
        plt.legend()
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio')
        plt.show()
        
    # estimarErroQuadraticoMedio estima o erro quadrático médio entre a saída prevista e a esperada
    def estimarErroQuadraticoMedio(self, dados, rotulos):
        erro_total = 0

        # Loop sobre cada amostra do conjunto de dados
        for dado, rotulo_esperado in zip(dados, rotulos):
            rotulo_previsto = np.array(self.feedForward(dado))
            erro = np.array(rotulo_esperado) - rotulo_previsto
            erro_total += np.sum(np.power(erro, 2))

        return erro_total / len(dados) 
    
    # estimarAcuracia estima a acurácia do modelo comparando as saídas previstas com as esperadas          
    def estimarAcuracia(self, dados, rotulos):
        total_acertos = 0
        
        # Loop sobre cada amostra do conjunto de dados
        for dado, rotulo_esperado in zip(dados, rotulos):
            rotulo_previsto = np.array(self.feedForward(dado))
            if np.argmax(rotulo_previsto) == np.argmax(rotulo_esperado):
                total_acertos += 1
        
        return round((total_acertos / len(dados)) * 100, 2)  
    
    # plotarMatrizDeConfusao plota uma matriz de confusão dado os dados de entrada e seus respectivos rotulos
    def plotarMatrizDeConfusao(self, dados, rotulos, diretorio):
        # Criar uma lista com todas as letras do alfabeto
        letras_alfabeto = list(string.ascii_uppercase)
        # Criar um DataFrame com índices e colunas como as letras do alfabeto, inicializando com zeros
        df = pd.DataFrame(0, index=letras_alfabeto, columns=letras_alfabeto)
        
        # Função para obter a letra dado uma posição de maior valor de um array de 26 posições
        def pegarLetraCorrespondente(array):
            # Obter o índice do maior valor no array
            maior_indice = np.argmax(array)
            # Converter o índice na letra correspondente
            letra = string.ascii_uppercase[maior_indice]
            
            return letra

        # Função para incrementar uma célula na tabela
        def incrementarCelula(df, linha, coluna):
            df.at[linha, coluna] += 1

        # Iterando sobre os dados 
        for dado, rotulo_esperado in zip(dados, rotulos):
            # Calcula com o feed forward a letra prevista pela rede
            rotulo_previsto = np.array(self.feedForward(dado))
            # Dado o array de saida pega a letra correspondente para o rotulo previsto pela rede e o rotulo esperado
            letra_prevista = pegarLetraCorrespondente(rotulo_previsto)
            letra_esperada = pegarLetraCorrespondente(rotulo_esperado)
            # Incrementa na matriz de confusão a posição na letra prevista e a letra esperada
            incrementarCelula(df, letra_esperada, letra_prevista)
            
        # Ajuste para incluir as letras na primeira linha e primeira coluna
        df.index.name = ''
        df.columns.name = ''
        
        # Salva a matriz em um arquivo excel no diretório passado por parâmetro
        df.to_excel(f"log/configuracao/{diretorio}/matriz_de_confusao.xlsx")
    
    # converterVetorMultidimensional recebe vetor de dados de treinamento e transforma em vetor bidimensional para melhor manipulação
    @staticmethod
    def converterVetorMultidimensional(vetor_multidimensional):
        matriz_dados = []
        # Itera sobre o vetor de maior dimensões
        for dados in vetor_multidimensional:
            # utiliza da função flatten para transformar o vetor "dados" de mais dimensões em um vetor unidimensional
            dado = dados.flatten()
            # Substitui os valores -1 por 0 para melhor manipulação
            dado[dado == -1] = 0 
            # Adiciona o vetor agora unidimensional a matriz de dados
            matriz_dados.append(dado)

        return matriz_dados
    
    # retiraVetorDaLista recebe uma lista e retira dela o vetor na posição dada pelo indice passado por parâmetro
    @staticmethod
    def retirarVetorDaLista(vetor, indice):
        return [item for i, item in enumerate(vetor) if i != indice]    
    
    # unirFolds recebe uma lista que contém os folds de treinamento e realiza a união deles em um vetor bidimensional
    @staticmethod    
    def unirFolds(lista):
        return [lista[x][y] for x in range(len(lista)) for y in range(len(lista[x]))]
    
    # holdout realiza o holdoult dos dados embaralhando-os aleatoriamente e retorando os dados de treinamento e 
    # teste em uma proporção de 80% para treinamento e 20% para teste
    @staticmethod  
    def holdout(dados, rotulos):
        # Criar um array de índices para Holdout
        indices = list(range(len(dados)))

        # Embaralhar os índices
        random.shuffle(indices)

        # Reorganizar os arrays de dados e rótulos com base nos índices embaralhados randomicamente
        dados_embaralhados = [dados[i] for i in indices]
        rotulos_embaralhados = [rotulos[i] for i in indices]

        # Calcular o tamanho do conjunto de dados para repartição dos dados de teste e treinamento
        # na proporção já citada
        reparticao =  (len(dados_embaralhados) // 10) * 8

        # Dividi os dados de treinamento e teste de acordo com a repaticao
        dados_treinamento = dados_embaralhados[:reparticao]
        rotulos_treinamento = rotulos_embaralhados[:reparticao]
        dados_teste = dados_embaralhados[reparticao:]
        rotulos_teste = rotulos_embaralhados[reparticao:]
        
        # retorna os dados de treinamento e teste com seus respectivos rotulos
        return dados_treinamento, rotulos_treinamento, dados_teste, rotulos_teste
    
    # criarLimparPastaConfiguracao realiza a limpeza (exclusão de arquivos) na pasta de log/configuracao
    # e cria subspastas para armazenamento de pesos e matrizes de confusão 
    @staticmethod
    def criarLimparPastasConfiguracao(numero_de_folds):
        # Função para excluir arquivos já existentes na pasta ou criar pasta caso não exista
        def LimpaCriaPasta(path):
            if os.path.exists(path):
            # Apaga todo o conteúdo do diretório
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # Remove o arquivo ou link simbólico
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Remove o diretório e todo seu conteúdo
                    except Exception as e:
                        print(f'Falha ao apagar {file_path}. Razão: {e}')
            else:
                # Cria o diretório se não existir
                os.makedirs(path)
                print(f'Diretório {path} criado.')
        
        iniciais = "iniciais"
        finais = "finais"
        treinamento_sem_validacao_cruzada = "log/configuracao/treinamento_sem_validacao_cruzada/"
        treinamento_com_validacao_cruzada = "log/configuracao/treinamento_com_validacao_cruzada/"
        
        # Limpa ou cria as pastas para o treinamento sem validação cruzada
        LimpaCriaPasta(treinamento_sem_validacao_cruzada+iniciais)
        LimpaCriaPasta(treinamento_sem_validacao_cruzada+finais)
        # Limpa ou cria as pastas para os folds para o treinamento com validação cruzada
        for fold in range(numero_de_folds):
            LimpaCriaPasta(treinamento_com_validacao_cruzada + f"config_folds_{fold+1}/" + iniciais)
            LimpaCriaPasta(treinamento_com_validacao_cruzada + f"config_folds_{fold+1}/" + finais)