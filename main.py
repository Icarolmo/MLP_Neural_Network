from readfile import readfile
from mlp.network import Network

x_train = readfile.getXtrainData("X.txt")
y_train = readfile.getYTrainData("Y_letra.txt")

mlp = Network(input_size=120, hidden_size=60, output_size=26)

mlp.train(x_train, y_train, epochs=1000, learning_rate=0.1)

train_accuracy = mlp.calculate_accuracy(x_train, y_train)

print(f'Acurácia no conjunto de treinamento: {train_accuracy * 100:.2f}%')

mlp.plot_loss()