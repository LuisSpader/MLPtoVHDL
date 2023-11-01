import tensorflow as tf
import math
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from skimage.transform import resize
from keras.utils import to_categorical
import multiprocessing
from multiprocessing import Value

np.set_printoptions(suppress=True)

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
path = os.path.dirname(script_path)
print("Script path:", path)

###     CONFIGURATION    ####
# Passe as configurações do modelo que estão salvas na pasta /models.

# O layers está ligado com o número de layers, é a forma em que temos salvado
# os modelos,
layers = "4_4_4_4_4_4_4_4_4"  
n_layers = "10"
neurons = "4"   # Número de neurônios do modelo treinado


# Mude os diretórios de acordo com o seu modelo, MNIST/4X4/LAYERS...
# significa que o modelo é o MNIST, as imagens são 4X4 e o que os modelos
# variam um do outro é a quantidade de camadas.
model = tf.keras.models.load_model(f'{path}/models/MNIST/4X4/LAYERS/{neurons}_NEURONS/NN_{n_layers}Layers_16_{layers}_10.h5')
print("\n\n\n\n")
B = 12 # Number of bits
write_path = f'{path}/model_results/ACCURACY/4X4/LAYERS/{neurons}_NEURONS/2_bitsmul/NN_{n_layers}Layers_{B}bits_16_{layers}_10.txt'


# Define a shared manager to create shared data structures
manager = multiprocessing.Manager()

# Create a shared counter for correct answers
correct_answer_counter = Value('i', 0)


class NumeroTruncado:
    def __init__(self, valor, B):
        self.B = B
        self.int_bits = B // 2
        self.frac_bits = B - self.int_bits

        self.valor = valor
        self.truncar()

    def truncar(self):
        # shifting the decimal point to the right by self.frac_bits places.
        temp = int(self.valor * (1 << self.frac_bits))  # Fixed point multiplication
        if temp & (1 << (self.B - 1)):  # If the number is negative in two's complement
            temp |= ~((1 << self.B) - 1)  # Extend the sign bit
        self.valor = temp / float(1 << self.frac_bits)

    def __add__(self, other):
        resultado = self.valor + other.valor
        res_truncado = NumeroTruncado(resultado, self.B)
        return res_truncado

    def __mul__(self, other):
        resultado = self.valor * other.valor
        res_truncado = NumeroTruncado(resultado, self.B * 2)  # Increase number of bits
        return res_truncado

    def __sub__(self, other):
        resultado = self.valor - other.valor
        res_truncado = NumeroTruncado(resultado, self.B)
        return res_truncado

    def __ge__(self, other):
        return self.valor >= other.valor

    def exp(self):
        resultado = math.exp(self.valor)
        res_truncado = NumeroTruncado(resultado, self.B)
        return res_truncado

    def __truediv__(self, other):
        resultado = self.valor / other.valor
        res_truncado = NumeroTruncado(resultado, self.B)
        return res_truncado

    def relu(self):
        resultado = max(0, self.valor)
        res_truncado = NumeroTruncado(resultado, self.B // 2)
        self.valor = res_truncado.valor
        self.B = res_truncado.B


    def sigmoid(self):
        resultado = 1 / (1 + math.exp(-self.valor))
        res_truncado = NumeroTruncado(resultado, self.B)
        self.valor = res_truncado.valor
        self.B = res_truncado.B
        

###   PREPROCESSING OF THE MODEL   ###

def weight_parser(model):
    values = model.get_weights()
    weights_list = []
    biases_list = []
    for i in range(len(values)):
        if i % 2 == 0:
            weights_list.append(values[i])
        else:
            biases_list.append(values[i])

    return weights_list, biases_list

###   TEMPORARY FUNCTIONS  ###

def relu(x):
    return max(0, x)
    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(z):
    assert len(z.shape) == 2

    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

from keras.datasets import mnist


###   SHOW THE 28 x 28 IMAGE  ###
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()
first_image = np.array(x_test[image_index], dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

'''

###   PREPROCESSING OF THE INPUT   ###
'''
def preprocess_mnist():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Resize the images to 8x8
    x_train_resized = np.zeros((x_train.shape[0], 4, 4))
    for i in range(x_train.shape[0]):
        x_train_resized[i] = resize(x_train[i], (4, 4))

    x_test_resized = np.zeros((x_test.shape[0], 4, 4))
    for i in range(x_test.shape[0]):
        x_test_resized[i] = resize(x_test[i], (4, 4))

    x_train_flat = x_train_resized / 255
    x_test_flat = x_test_resized / 255

    # Flatten the images
    x_train_flat = x_train_resized.reshape(x_train_resized.shape[0], -1)
    x_test_flat = x_test_resized.reshape(x_test_resized.shape[0], -1)

    # Normalize the pixel values
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)

    model.evaluate(x_test_flat, y_test_onehot)
    
    return x_train_flat, x_test_flat, y_train_onehot, y_test_onehot


###   PERSISTING THE PREPROCESSED INPUT   ###

x_train_flat, x_test_flat, y_train_onehot, y_test_onehot = preprocess_mnist()
with open(f'{path}/models/x_4x4.pkl', 'wb') as f:
    pickle.dump((x_train_flat, x_test_flat, y_train_onehot, y_test_onehot), f)
'''

def neural_network(input_matrix, B):
    # Parsing the weights and biases from the model
    weights, biases = weight_parser(model)

    # Convert weights to NumeroTruncado objects
    weights_trunc = []
    for weight_matrix in weights:
        weight_matrix_trunc = np.zeros(weight_matrix.shape, dtype=object)
        for i in range(weight_matrix.shape[0]):
            for j in range(weight_matrix.shape[1]):
                weight_matrix_trunc[i, j] = NumeroTruncado(weight_matrix[i, j], B)
        weights_trunc.append(weight_matrix_trunc)

    # Convert biases to NumeroTruncado objects
    biases_trunc = []
    for bias_vector in biases:
        bias_vector_trunc = np.zeros(bias_vector.shape, dtype=object)
        for i in range(bias_vector.shape[0]):
            bias_vector_trunc[i] = NumeroTruncado(bias_vector[i], B)
        biases_trunc.append(bias_vector_trunc)

    # Convert input_matrix to NumeroTruncado objects
    input_matrix_trunc = np.zeros(input_matrix.shape, dtype=object)
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            input_matrix_trunc[i, j] = NumeroTruncado(input_matrix[i, j], B)
            
    counter = 0  # For n_MAC and n

    # Iterating through the layers except the last one because of the activation function
    for i in range(len(weights_trunc) - 1):
        # Calculating the output matrix by matrix vector multiplication and adding the bias
        output_matrix = np.dot(input_matrix_trunc, weights_trunc[i]) + biases_trunc[i]

        # Applying the relu function to each element of the matrix by using map and lambda
        np.array(list(map(lambda x: list(map(lambda y: y.relu(), x)), output_matrix)))
        
        # Setting the input_matrix to the next layer of activations
        input_matrix_trunc = output_matrix

    output_matrix = input_matrix_trunc  # In case there is only one layer
    # Calculating the last dot product to apply the softmax function
    output_matrix = np.dot(output_matrix, weights_trunc[-1]) + biases_trunc[-1]
    # Apply softmax function to each row of the output matrix
    output_matrix = softmax(output_matrix)
    
    # To use sigmoid instead of softmax, uncomment the next line and comment the previous one
    #np.array(list(map(lambda x: list(map(lambda y: y.sigmoid(), x)), output_matrix)))
    return output_matrix

def write_output(accuracy):
    with open(write_path, 'a') as f:
        f.write(f"\n\n")
        f.write(f'----------------------------    Results   ----------------------------\n\n')
        f.write(f"\n Accuracy of the truncated model: {accuracy}\n\n")
        f.write(f"Accuracy of the tensorflow model: {model.evaluate(x_test_flat, y_test_onehot)[1]}\n\n")
    
###   LOADING THE PREPROCESSED INPUT   ###
# Load variables from file
with open(f'{path}/models/x_4x4.pkl', 'rb') as f:
    x_train_flat, x_test_flat, y_train_onehot, y_test_onehot = pickle.load(f)
    

def process_image(image_index, x_test_flat, B):
    # with open(write_path, 'a') as f:
    #     f.write(f'\nImage {image_index}\n\n')
    ###   NEURAL NETWORK INPUT   ###
    
    image = x_test_flat[image_index]  # Choose index arbitrarily from the image set
    ###   SHOW THE 8 x 8 IMAGE   ###
    '''

    first_image = np.array(image, dtype='float')
    pixels = first_image.reshape((8, 8))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    '''
    image = np.array(image)
    image = np.reshape(image, (1, -1))
        
    X = image

    ###   NEURAL NETWORK OUTPUT   ###

    result = neural_network(X, B)
    result_values = result[0]
    max_guess = max(result_values, key=lambda x: x.valor)  # Find the maximum 'valor' attribute
    truncated_answer_index = np.argmax([x.valor for x in result_values])  # Find the index with maximum 'valor'

    if y_test_onehot[image_index][truncated_answer_index] == 1:
        with correct_answer_counter.get_lock():
            correct_answer_counter.value += 1

if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    print("ok")

    for i in range(len(x_test_flat)):
        pool.apply_async(process_image, (i, x_test_flat, B))

    pool.close()
    pool.join()
    print(correct_answer_counter.value)

    accuracy = correct_answer_counter.value / (len(x_test_flat))

    write_output(accuracy)
