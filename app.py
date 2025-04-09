import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import os
import time

# Cria as pastas para salvar os feedbacks, se ainda não existirem
feedback_dir = "dataset"
amarrado_dir = os.path.join(feedback_dir, "amarrado")
desamarrado_dir = os.path.join(feedback_dir, "desamarrado")
vazio_dir = os.path.join(feedback_dir, "vazio")
os.makedirs(amarrado_dir, exist_ok=True)
os.makedirs(desamarrado_dir, exist_ok=True)
os.makedirs(vazio_dir, exist_ok=True)

# Carrega o modelo treinado previamente para predição
model = load_model("modelo_tenis_transfer_learning.h5")

# Define o tamanho de entrada (128x128)
IMG_SIZE = (128, 128)

# Inicia a captura da webcam (índice 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a webcam")
    exit()

print("Pressione 'q' para sair da aplicação.")
print("Pressione 'f' para fornecer feedback (e salvar a imagem na pasta correspondente).")

label_map = {0: "amarrado", 1: "desamarrado", 2: "vazio"}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    # --- Pré-processamento para predição ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
    frame_preprocessed = preprocess_input(frame_resized.astype("float32"))
    frame_preprocessed_batch = np.expand_dims(frame_preprocessed, axis=0)

    # --- Previsão ---
    prediction = model.predict(frame_preprocessed_batch)
    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class]
    label_pred = label_map[pred_class]
    text = f"{label_pred}: {confidence:.2f}"

    # Exibe a predição no frame original
    frame_display = frame.copy()
    cv2.putText(frame_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_display, "Pressione 'f' para fornecer feedback", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Webcam - Feedback para Treinamento", frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        print("\nModo de feedback ativado:")
        print("Pressione '0' para 'amarrado', '1' para 'desamarrado' ou '2' para 'vazio'.")
        fb_key = cv2.waitKey(0) & 0xFF
        if fb_key == ord('0'):
            label_name = "amarrado"
            save_dir = amarrado_dir
        elif fb_key == ord('1'):
            label_name = "desamarrado"
            save_dir = desamarrado_dir
        elif fb_key == ord('2'):
            label_name = "vazio"
            save_dir = vazio_dir
        else:
            print("Entrada inválida, feedback ignorado.")
            continue

        frame_resized_bgr = cv2.resize(frame, IMG_SIZE)
        filename = os.path.join(save_dir, f"feedback_{label_name}_{int(time.time()*1000)}.jpg")
        cv2.imwrite(filename, frame_resized_bgr)
        print(f"Imagem salva em: {filename}")

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
