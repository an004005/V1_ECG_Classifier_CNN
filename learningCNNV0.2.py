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
from sklearn.preprocessing import MultiLabelBinarizer

# Test = io.loadmat('./testImg_half_II.mat')
# Test = io.loadmat('./testImg_half_V5.mat')
Test = io.loadmat('./testImg_half_V1.mat')
X, y = Test['data'], Test['label']
print(y.shape)

X = X/255.0


print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))
classes = len(mlb.classes_)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

X_train = X_train.reshape(X_train.shape[0], 96, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 96, 48,1)


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
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(4096, kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.3)(x)
    output_tensor = layers.Dense(classes, activation='sigmoid')(x)
    
    myvgg = Model(input_tensor, output_tensor)
    myvgg.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    myvgg.summary()
    return myvgg

 
model = get_model()
# file_path="my_ecg_multilable_II.h5"
# file_path="my_ecg_multilable_V5.h5"
file_path="my_ecg_multilable_V1.h5"
checkpoint = ModelCheckpoint(file_path, 
            monitor='loss', 
            mode='min', 
            save_best_only=True)
early = EarlyStopping(monitor="val_acc", mode="min", patience=7, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="min", patience=5, verbose=2)
callbacks_list=[checkpoint, redonplat]

model.fit(X_train, y_train, epochs=25, callbacks=callbacks_list, verbose=2, validation_split=0.1)
# model.load_weights(file_path)

# pred_test = model.predict(X_test)
# pred_test = np.argmax(pred_test, axis=-1)

# f1 = f1_score(y_test, pred_test, average="macro")

# print("Test f1 score : %s "% f1)

# acc = accuracy_score(y_test, pred_test)

# print("Test accuracy score : %s "% acc)