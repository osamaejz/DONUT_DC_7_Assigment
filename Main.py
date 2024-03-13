from MI_Process import MI_Preprocessing, EOG_Correction
import numpy as np

Path = "C:\\Users\\Dell\\Desktop\\Dataset\\"

LH_X_Train, RH_X_Train, LH_X_Test, RH_X_Test = MI_Preprocessing(path = Path, Model = 4)

# Performing EOG correction
R2_L_X_Train, LH_X_Train = EOG_Correction(LH_X_Train)
R2_H_X_Train, RH_X_Train = EOG_Correction(RH_X_Train)
R2_L_X_Test, LH_X_Test = EOG_Correction(LH_X_Test)
R2_H_X_Test, RH_X_Test = EOG_Correction(RH_X_Test)

#Creating binary labels for classification
LH_y_Train = np.zeros((LH_X_Train.shape[0], 1))# LH_class = 0
RH_y_Train = np.ones((RH_X_Train.shape[0], 1))# RH_class = 1
LH_y_Test = np.zeros((LH_X_Test.shape[0], 1))
RH_y_Test = np.ones((RH_X_Test.shape[0], 1))

#Combining left and right hand imagination training and testing features and labels
X_train = np.concatenate((LH_X_Train, RH_X_Train), axis=0)
y_train = np.concatenate((LH_y_Train, RH_y_Train), axis=0)

X_test = np.concatenate((LH_X_Test, RH_X_Test), axis=0)
y_test = np.concatenate((LH_y_Test, RH_y_Test), axis=0)

# Implementing CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.utils import shuffle

# Reshape the input data to be compatible with Conv2D layers
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

# Data Shuffling to overcome biasness
X_train_reshaped, y_train = shuffle(X_train_reshaped, y_train, random_state=42)

# Create the CNN model for 3 Channels EEG data of 9 subjects

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, 9, 1)))
model.add(MaxPooling2D((1, 2)))  # Adjusted MaxPooling2D layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((1, 2)))  # Adjusted MaxPooling2D layer
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Model Evaluation using Test set
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Saving Trained Model
model.save('Models/EEG only/Model_4_without_EOG_Correction.h5')




















