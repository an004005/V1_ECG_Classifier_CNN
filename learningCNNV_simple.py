from keras import layers, models
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.metrics import f1_score, accuracy_score

Test = io.loadmat('./testImg_half_shape_V1.mat')
X, y = Test['data'], Test['label']

X = X/255.0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

X_train = X_train.reshape(X_train.shape[0], 96, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 96, 48,1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()

# model
def get_model():
    input_tensor = Input(shape=(96, 48, 1), dtype='float32', name='input')
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.4)(x)
    output_tensor = layers.Dense(3, activation='softmax')(x)
    
    myvgg = Model(input_tensor, output_tensor)
    myvgg.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    myvgg.summary()
    return myvgg

 
model = get_model()
file_path="my_ecg.h5"
checkpoint = ModelCheckpoint(file_path, 
            monitor='loss', 
            mode='min', 
            save_best_only=True)
early = EarlyStopping(monitor="val_acc", mode="min", patience=7, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="min", patience=3, verbose=2)
callbacks_list=[checkpoint, redonplat]

model.fit(X_train, y_train, epochs=25, callbacks=callbacks_list, verbose=2, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)

f1 = f1_score(y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(y_test, pred_test)

print("Test accuracy score : %s "% acc)