from keras.datasets import cifar10
import keras

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
print(X_train.min(),X_train.max(),y_train.min(),y_train.max())

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

X_train = X_train/ 255.0 - 0.5
X_test = X_test/ 255.0 - 0.5

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(6,(3,3),
                              activation='relu',
                              input_shape=(32,32,3),
                              kernel_initializer=keras.initializers.truncated_normal(0,0.2)))
model.add(keras.layers.MaxPool2D())

model.add(keras.layers.Conv2D(16,(3,3),
                              activation='relu',
                              input_shape=(32,32,3),
                              kernel_initializer=keras.initializers.truncated_normal(0,0.2)))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Dropout(0.9))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)
# 40000/40000 [==============================] - 34s - loss: 1.6589 - acc: 0.3847 - val_loss: 1.4902 - val_acc: 0.4756

