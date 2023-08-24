import pandas as pd
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras

# Number of classes in the target variable
NB_CLASSES = 3

VERBOSE = 1

BATCH_SIZE = 16

EPOCHS = 10

VALIDATION_SPLIT = 0.2
# Extracting an preparing data
iris_data = pd.read_csv("iris.csv")

label_encoder = preprocessing.LabelEncoder()
iris_data["Species"] = label_encoder.fit_transform(
    iris_data["Species"]
)

np_iris = iris_data.to_numpy()

X_data = np_iris[:, 0:4]
Y_data = np_iris[:, 4]

scaler = StandardScaler().fit(X_data)

Y_data = tf.keras.utils.to_categorical(Y_data, 3)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10)

# Creating model with Keras

model = tf.keras.models.Sequential()

model.add(keras.layers.Dense(128,
                             input_shape=(4,),
                             name='Hidden-Layer-1',
                             activation='relu'))

model.add(keras.layers.Dense(128,
                             name='Hidden-Layer-2',
                             activation='relu'))

model.add(keras.layers.Dense(NB_CLASSES,
                             name='Output-Layer',
                             activation='softmax'))

# Compile the model with loss & metrics
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training and evaluating model

history = model.fit(X_train,
                    Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy improvements with Epoch")
plt.show()

model.evaluate(X_test,Y_test)

# Saving and loading models

model.save("iris_save")

loaded_model = keras.models.load_model("iris_save")
loaded_model.summary()

# Prediction with model

prediction_input = [[6.6, 3., 4.4, 1.4]]
scaled_input = scaler.transform(prediction_input)

raw_prediction = model.predict(scaled_input)
print("Raw Prediction Output (Probabilities) :" , raw_prediction)

prediction = np.argmax(raw_prediction)
print("Prediction is ", label_encoder.inverse_transform([prediction]))
