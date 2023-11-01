import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from skimage.transform import resize
import numpy as np
import os
import pickle

script_path = os.path.abspath(__file__)
path = os.path.dirname(script_path)
print("Script path:", path)



# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


###   PREPROCESSING   ###
'''
# Resize the images to 4x4
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


# Save the preprocessed input
with open(f'{path}/x_4x4.pkl', 'wb') as f:
    pickle.dump((x_train_flat, x_test_flat), f)
'''


# Load variables from file
with open(f'{path}/x_4x4.pkl', 'rb') as f:
    x_train_flat, x_test_flat = pickle.load(f)


# One-hot encode the labels
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

###   PARAMETERS   ###
input_shape = (16,)  # Fixed
middle_layers = 16  # Changeable
last_layer = 10  # Fixed


model = Sequential()
model.add(Dense(middle_layers, activation='relu', input_shape=input_shape))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(middle_layers, activation='relu'))
model.add(Dense(last_layer, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_flat, y_train_onehot, epochs=30, batch_size=32)

# Evaluate the model
model.evaluate(x_test_flat, y_test_onehot)
model.save(f'{path}/MNIST/4X4/LAYERS/16_NEURONS/NN_10Layers_{input_shape[0]}_{middle_layers}_{middle_layers}_{middle_layers}_{middle_layers}_{middle_layers}_{middle_layers}_{middle_layers}_{last_layer}.h5')