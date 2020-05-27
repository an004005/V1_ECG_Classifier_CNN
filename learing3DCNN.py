from keras import layers, models
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.metrics import f1_score, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Test = io.loadmat('./testImg_3D_half.mat')
X, y = Test['data'], Test['label']

types = {'sinus': 0, 'sinuslbbb': 1, 'sinusrbbb': 2, 'pvclbbb': 3, 'pvcrbbb': 4, 'vtlbbb': 5, 'vtrbbb': 6}

tmp = []

for i in y:
    tmp.append(types[i.replace(' ', '')])
y = np.array(tmp)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
X_train = X
y_train = y
X_test=X
y_test = y
print(X.shape)

# model
def get_model():
    input_tensor = Input(shape=(15, 128, 32, 1), dtype='float32', name='input')
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling3D((1,1,2))(x)
    
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling3D((1,1,2))(x)
    
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, kernel_initializer='he_normal')(x)
    x = layers.Dense(4096, kernel_initializer='he_normal')(x)
    output_tensor = layers.Dense(7, activation='softmax')(x)
    
    myvgg = Model(input_tensor, output_tensor)
    myvgg.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    myvgg.summary()
    return myvgg

 
model = get_model()
file_path="my_ecg_3D.h5"
checkpoint = ModelCheckpoint(file_path, 
            monitor='loss', 
            mode='min', 
            save_best_only=True)
early = EarlyStopping(monitor="val_acc", mode="min", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="min", patience=3, verbose=2)
callbacks_list=[checkpoint, early, redonplat]

model.fit(X_train, y_train, epochs=15, callbacks=callbacks_list, verbose=2, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(y_test, pred_test)

print("Test accuracy score : %s "% acc)