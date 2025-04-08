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
os.makedirs(amarrado_dir, exist_ok=True)
os.makedirs(desamarrado_dir, exist_ok=True)

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    # --- Pré-processamento para predição ---
    # Converte de BGR para RGB (pois o modelo foi treinado com imagens RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Redimensiona para 128x128 (como usado no treinamento)
    frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
    # Aplica o preprocess_input do MobileNetV2 e converte para float32
    frame_preprocessed = preprocess_input(frame_resized.astype("float32"))
    # Adiciona a dimensão de batch
    frame_preprocessed_batch = np.expand_dims(frame_preprocessed, axis=0)
    
    # --- Previsão ---
    prediction = model.predict(frame_preprocessed_batch)
    prob = prediction[0][0]
    # Se o valor for menor que 0.5, classifica como "amarrado"; caso contrário, "desamarrado"
    label_pred = "amarrado" if prob < 0.5 else "desamarrado"
    text = f"{label_pred}: {prob:.2f}"
    
    # Exibe a predição no frame original
    frame_display = frame.copy()
    cv2.putText(frame_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_display, "Press 'f' para feedback", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Webcam - Feedback para Treinamento", frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Modo de feedback: solicita ao usuário a classe correta via teclado
        print("\nModo de feedback ativado:")
        print("Pressione '0' para 'amarrado' ou '1' para 'desamarrado'.")
        fb_key = cv2.waitKey(0) & 0xFF  # Bloqueia a execução até que o usuário forneça input
        if fb_key == ord('0'):
            correct_label = 0
            save_dir = amarrado_dir
            label_name = "amarrado"
        elif fb_key == ord('1'):
            correct_label = 1
            save_dir = desamarrado_dir
            label_name = "desamarrado"
        else:
            print("Entrada inválida, feedback ignorado.")
            continue
        
        # Para salvar, redimensionamos a imagem original (em BGR) para 128x128
        frame_resized_bgr = cv2.resize(frame, IMG_SIZE)
        # Cria um nome de arquivo único usando timestamp
        filename = os.path.join(save_dir, f"feedback_{label_name}_{int(time.time()*1000)}.jpg")
        cv2.imwrite(filename, frame_resized_bgr)
        print(f"Imagem salva em: {filename}")

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
