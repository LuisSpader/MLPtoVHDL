import tensorflow as tf
import os.path
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=2))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(16))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))

sgd = tf.keras.optimizers.SGD(lr = 0.1)

model.compile(
    optimizer=sgd,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, batch_size=1, epochs=1000)
print(model.predict(X))

model.save('models/xor_v3.h5')
    
#xor = tf.keras.models.load_model('models/xor.h5')
#print(xor.get_weights())

#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=False)