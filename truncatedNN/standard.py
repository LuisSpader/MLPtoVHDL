import tensorflow as tf
import math
import os
import numpy as np
import pickle

from skimage.transform import resize

np.set_printoptions(suppress=True)

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
path = os.path.dirname(script_path)
print("Script path:", path)

layers = "4_4_4_4_4_4_4"
n_layers = "8"
neurons = "4"

model = tf.keras.models.load_model(f'{path}/models/MNIST/4X4/LAYERS/{neurons}_NEURONS/NN_{n_layers}Layers_16_{layers}_10.h5')
print("\n\n\n\n")

write_path = f'{path}/model_results/MACs/4X4/LAYERS/{neurons}_NEURONS/2_bitsmul/NN_{n_layers}Layers_Float_16_{layers}_10.txt'


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


# Example usage:

with open(f'{path}/models/x_4x4.pkl', 'rb') as f:
    x_train_flat, x_test_flat, y_train_onehot, y_test_onehot = pickle.load(f)

index = 0
image = x_test_flat[index]

# num_trunc_matrix = np.vectorize(NumeroTruncado, excluded=['B'])(x_test, B=N)

num_trunc_list = []
# print(image)
for i in range(len(image)):
    num_trunc_list.append(image[i])


X = [num_trunc_list]
#print(X)

#a = bias_converter(X)
#a = np.array(a)
def neural_network(input_matrix):

    weights, biases = weight_parser(model)
    input_matrix = np.array(input_matrix)
    counter = 0  # For n_MAC and n

    with open(write_path, 'a') as f:
        for i in range(len(weights) - 1):
            f.write(f'Layer {i}\n\n')
            output_matrix = np.dot(input_matrix, weights[i]) + biases[i]
            counter_aux = counter
            
            for j in range(output_matrix.shape[0]):
                for k in range(output_matrix.shape[1]):
                    f.write(f'n{counter}_MAC: {output_matrix[j, k]}\n')
                    counter += 1
            f.write('\n')
            
            
            output_matrix = np.array(list(map(lambda x: list(map(lambda y: relu(y), x)), output_matrix)))
            for j in range(output_matrix.shape[0]):
                for k in range(output_matrix.shape[1]):
                    f.write(f'n{counter_aux}: {output_matrix[j, k]}\n')
                    counter_aux += 1
            
            f.write('\n')
            
            input_matrix = output_matrix

        #print(output_matrix)

        output_matrix = np.dot(output_matrix, weights[-1]) + biases[-1]
        for j in range(output_matrix.shape[0]):
            for k in range(output_matrix.shape[1]):
                f.write(f'n{counter}_MAC: {output_matrix[j, k]}\n')
                counter += 1
                
        f.write('\n')
        output_matrix = softmax(output_matrix)
        
        for j in range(output_matrix.shape[0]):
            for k in range(output_matrix.shape[1]):
                f.write(f'n{counter_aux}: {output_matrix[j, k]}\n')
                counter_aux += 1

    # Para usar o softmax:
    #output_matrix = np.array(list(map(lambda x: list(map(lambda y: softmax(y), x)), output_matrix)))

    return output_matrix

for i in range(len(x_test_flat) - 9990):
    with open(write_path, 'a') as f:
        f.write(f'\nImage {i}\n\n')
    
    max_guess = 0

    ###   NEURAL NETWORK INPUT   ###

    image = x_test_flat[i]  # Choose index arbitrarily from the image set
    


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

    result = neural_network(X)
    result_values = result[0]


