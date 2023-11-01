import tensorflow as tf
import math
import os
import numpy as np
from matplotlib import pyplot as plt

from keras.utils import to_categorical

np.set_printoptions(suppress=True)

# OBS: Este arquivo deve ser utilizado para debug
# 

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
path = os.path.dirname(script_path)
print("Script path:", path)
layers = "8_8_8_8_8_8_8_8_8"
n_layers = "10"
neurons = "8"

model = tf.keras.models.load_model(f'{path}/models/MNIST/4X4/LAYERS/{neurons}_NEURONS/NN_{n_layers}Layers_16_{layers}_10.h5')
print("\n\n\n\n")

B = 12 # Number of bits

write_path = f'{path}/model_results/MACs/4X4/LAYERS/{neurons}_NEURONS/2_bitsmul/NN_{n_layers}Layers_{B}bits_16_{layers}_10.txt'

class NumeroTruncado:
    def __init__(self, valor, B):
        self.B = B
        self.int_bits = B // 2
        self.frac_bits = B - self.int_bits

        self.valor = valor
        self.truncar()

    def truncar(self):
        temp = int(self.valor * (1 << self.frac_bits))
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


import pickle

###   PREPROCESSING OF THE INPUT   ###
'''
def preprocess_mnist():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Resize the images to 8x8
    x_train_resized = np.zeros((x_train.shape[0], 8, 8))
    for i in range(x_train.shape[0]):
        x_train_resized[i] = resize(x_train[i], (8, 8))

    x_test_resized = np.zeros((x_test.shape[0], 8, 8))
    for i in range(x_test.shape[0]):
        x_test_resized[i] = resize(x_test[i], (8, 8))

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
with open(f'{path}/models/x_test_flat.pkl', 'wb') as f:
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
    with open(write_path, 'a') as f:
        for i in range(len(weights_trunc) - 1):
            f.write(f'Layer {i}\n\n')
            # Calculating the output matrix by matrix vector multiplication and adding the bias
            output_matrix = np.dot(input_matrix_trunc, weights_trunc[i]) + biases_trunc[i]
            # Print all the values of output_matrix
            counter_aux = counter
            for j in range(output_matrix.shape[0]):
                for k in range(output_matrix.shape[1]):
                    f.write(f'n{counter}_MAC: {output_matrix[j, k].valor}\n')
                    counter += 1
            f.write('\n')
                
            # Applying the relu function to each element of the matrix by using map and lambda
            np.array(list(map(lambda x: list(map(lambda y: y.relu(), x)), output_matrix)))
            for j in range(output_matrix.shape[0]):
                for k in range(output_matrix.shape[1]):
                    f.write(f'n{counter_aux}: {output_matrix[j, k].valor}\n')
                    counter_aux += 1
            
            f.write('\n')
            
            # Setting the input_matrix to the next layer of activations
            input_matrix_trunc = output_matrix

        output_matrix = input_matrix_trunc  # In case there is only one layer
        # Calculating the last dot product to apply the softmax function
        f.write(f'Layer {len(weights_trunc) - 1}\n\n')
        output_matrix = np.dot(output_matrix, weights_trunc[-1]) + biases_trunc[-1]
        for j in range(output_matrix.shape[0]):
            for k in range(output_matrix.shape[1]):
                f.write(f'n{counter}_MAC: {output_matrix[j, k].valor}\n')
                counter += 1
        # Apply softmax function to each row of the output matrix
        output_matrix = softmax(output_matrix)
        
        f.write('\n')
        
        for j in range(output_matrix.shape[0]):
            for k in range(output_matrix.shape[1]):
                f.write(f'n{counter_aux}: {output_matrix[j, k].valor}\n')
                counter_aux += 1
        
        # To use sigmoid instead of softmax, uncomment the next line and comment the previous one
        #np.array(list(map(lambda x: list(map(lambda y: y.sigmoid(), x)), output_matrix)))

        return output_matrix
    
###   LOADING THE PREPROCESSED INPUT   ###
# Load variables from file
with open(f'{path}/models/x_4x4.pkl', 'rb') as f:
    x_train_flat, x_test_flat, y_train_onehot, y_test_onehot = pickle.load(f)
    

# for comparison with the model
correct_answer_counter = 0


for i in range(len(x_test_flat) - 9990):
    with open(write_path, 'a') as f:
        f.write(f'\nImage {i}\n\n')
    
    max_guess = 0
    
    ###   SHOW THE 28 x 28 IMAGE  ###
    

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    first_image = np.array(x_test[i], dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    ###   NEURAL NETWORK INPUT   ###

    image = x_test_flat[i]  # Choose index arbitrarily from the image set
    
    ###   SHOW THE 8 x 8 IMAGE   ###

    first_image = np.array(image, dtype='float')
    pixels = first_image.reshape((4, 4))
    plt.imshow(pixels, cmap='gray')
    plt.show()

        
    image = np.array(image)
    image = np.reshape(image, (1, -1))
        
    X = image

    ###   NEURAL NETWORK OUTPUT   ###

    result = neural_network(X, B)
    result_values = result[0]
    max_guess = max(result_values, key=lambda x: x.valor)  # Find the maximum 'valor' attribute
    truncated_answer_index = np.argmax([x.valor for x in result_values])  # Find the index with maximum 'valor'

    if y_test_onehot[i][truncated_answer_index] == 1:
        correct_answer_counter += 1
        
print(correct_answer_counter)

def write_output():
    with open(write_path, 'a') as f:
        f.write(f"\n\n")
        f.write(f'----------------------------    Results   ----------------------------\n\n')
        f.write(f"\n Accuracy of the model for the 10 images: {correct_answer_counter / (len(x_test_flat) - 9990)}\n\n")
        # write the tensorflow model accuracy
        f.write(f"Accuracy of the tensorflow model: {model.evaluate(x_test_flat, y_test_onehot)[1]}\n\n")

write_output()