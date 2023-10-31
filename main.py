
import numpy as np

def sigmoid(x):
    return 1/(1 + np.e**-x)

class Layer:
    #le pasamos el numero de conexiones de la capa y el numero de neuronas de esa capa
    def __init__(self, con, neuron):
        #vamos a hacer un bias para cada neurona
        #self.b = np.random.rand(1, neuron) * 2 - 1
        self.b = np.ones((1, neuron))
        #ahora hacemos la matriz de pesos
        #self.w = np.random.rand(con, neuron) * 2 - 1
        self.w = np.ones((con, neuron))
class NeuralNetwork:
    #le pasamos la topologia que va a ser una lista de enteros
    #la funcion de activacion sera el sigmoide
    def __init__(self, top = [], act_f = sigmoid):
        self.top = top
        self.act_f = act_f
        self.model = self.define_model() #define_model genera la red neuronal y almacenamos el atributo
    def define_model(self):
        NeuralNetwork = [] #vamos a agregar los layers a cada espacio de la lista
        for i in range(len(self.top[:-1])):
            NeuralNetwork.append(Layer(self.top[i], self.top[i + 1])) #vamos a agregar un objeto tipo layer
            #el numero de neuronas de la capa actual tiene que ser igual al numero de conexiones de la capa anterior.
        return NeuralNetwork
    #el metodo predict le pasamos los datos/inputs de los que queremos hacer predicciones
    #que es nuestro array de inputs
    def predict(self, X = []):
        out = X
        #vamos a recorrer cada layer de nuestro modelo
        for i in range(len(self.model)):
            Z = self.act_f(out @ self.model[i].w + self.model[i].b) #Va a ser la salida de cada layer y igualamos a la interaccion pasada
            #primero multiplicamos los datos, despu{es los multiplicamos por los pesos de la red neuronal y les sumamos los bias de la primera capa
            out = Z # ahora la salida va a ser igual a la del primer layer, mas bien la entrada del primer layer va a ser igual a la del layer anterior
        return out

def main():
    brain = NeuralNetwork(top=[2, 3, 2], act_f=sigmoid)  # generamos una red neuronal
    print(brain.predict(X=[0, 0]))
    print(brain.predict(X=[0, 1]))
    print(brain.predict(X=[1, 0]))
    print(brain.predict(X=[1, 1]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
