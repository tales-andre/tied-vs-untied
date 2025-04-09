import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# Configurações
dataset_path = "dataset"
IMG_SIZE = (128, 128)

# Função de carregamento (mantida igual)
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['amarrado', 'desamarrado']:
        path = os.path.join(folder, label)
        class_num = 0 if label == 'amarrado' else 1
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(class_num)
    return np.array(images), np.array(labels)

# Carregar e pré-processar os dados
X, y = load_images_from_folder(dataset_path)
X = preprocess_input(X.astype('float32'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation ajustado
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
datagen.fit(X_train)

# Construir o modelo
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
]

# Treinar o modelo
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history_fine = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Avaliar e salvar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")
model.save("modelo_tenis_transfer_learning.h5")

# --------------------------
# 6. Relatório de Classificação e Matriz de Confusão
# --------------------------
# Gerar predições para o conjunto de teste (limite de 0.5 para classificação binária)
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Imprimir o Relatório de Classificação
print("\nRelatório de Classificação:\n")
report = classification_report(y_test, y_pred, target_names=["amarrado", "desamarrado"])
print(report)

# Calcular a Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)

# Plot da Matriz de Confusão
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
classes = ["amarrado", "desamarrado"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Predito')
plt.tight_layout()
plt.show()

# --------------------------
# 7. Visualização do Treinamento para Análise de Overfitting
# --------------------------
plt.figure(figsize=(12, 5))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o', label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], marker='o', label='Acurácia de Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

# Gráfico de Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o', label='Loss de Treino')
plt.plot(history.history['val_loss'], marker='o', label='Loss de Validação')
plt.title('Loss por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()