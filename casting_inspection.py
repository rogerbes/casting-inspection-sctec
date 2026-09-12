"""
================================================================================
MINI-PROJETO AVALIATIVO: INSPEÇÃO DE QUALIDADE EM PEÇAS DE FUNDIÇÃO (CASTING)
Abordagem Híbrida: Visão Computacional Clássica (OpenCV) e Deep Learning (CNN / TensorFlow)
================================================================================
Estrutura das Sprints Industriais:
- Sprint 1: Configuração e Versionamento de Ambiente.
- Sprint 2: Análise Exploratória Clássica (Grayscale + Blur).
- Sprint 3: Destaque de Características (Threshold + Canny + Morfologia).
- Sprint 4: Ingestão de Dados em Lote e Data Augmentation Dinâmico.
- Sprint 5: Construção e Treinamento da Arquitetura CNN.
- Sprint 6: Auditoria Gráfica (Curvas de Loss/Acurácia) e Avaliação de Overfitting.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================================================================
# SPRINT 1: CONFIGURAÇÃO DO AMBIENTE E PARÂMETROS GERAIS
# ==============================================================================

IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset_casting"  # Diretório contendo as pastas 'ok' e 'def_front'

def run_opencv_exploratory_analysis(sample_image_path):

#============================================================================
# SPRINT 2 & SPRINT 3: PROCESSAMENTO CLÁSSICO COM OPENCV
#============================================================================

    print("[INFO] Executando Sprints 2 & 3: Análise Exploratória Clássica (OpenCV)...")

    # Leitura da imagem de amostragem
    img_bgr = cv2.imread(sample_image_path)
    if img_bgr is None:
        print(f"[ERRO] Imagem de amostragem não encontrada em: {sample_image_path}")
        return

    # SPRINT 2: Conversão de cor e Filtro de Suavização
    # Conversão de BGR (padrão OpenCV) para Grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Aplicação do Filtro Gaussiano (5x5) para atenuação de ruído metálico
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # SPRINT 3: Limiarização, Canny e Operações Morfológicas
    # Limiarização Binarizada Automatizada (Método de Otsu)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Algoritmo Canny para Extração de Bordas Destacadas
    img_canny = cv2.Canny(img_blur, threshold1=100, threshold2=200)

    # Definição do Elemento Estruturante (Kernel 3x3) para Morfologia
    kernel = np.ones((3, 3), np.uint8)

    # Dilatação Morfológica: Expande as bordas detectadas para unir fendas
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)

    # Erosão Morfológica: Suprime pequenos pontilhados de ruído residual
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

    # Geração do Painel Visual de Comparação (6 Etapas da Análise Clássica)
    titles = [
        '1. Imagem Original (RGB)', '2. Escala de Cinza', 
        '3. Suavização (Gaussian Blur)', '4. Limiarização (Otsu)', 
        '5. Bordas (Canny)', '6. Morfologia (Dilatação/Erosão)'
    ]
    images = [
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), img_gray, 
        img_blur, img_thresh, 
        img_canny, img_eroded
    ]

    plt.figure(figsize=(14, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=11, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("pipeline_opencv_exploratorio.png", dpi=300)
    plt.close()
    print("[SUCESSO] Painel da análise exploratória OpenCV salvo em 'pipeline_opencv_exploratorio.png'.")

def build_and_train_cnn(dataset_dir):

#============================================================================

    print("[INFO] Executando Sprint 4: Carregamento do Dataset e Data Augmentation...")

    # Sprint 4: Ingestão de Dados via image_dataset_from_directory (Split 80% Treino / 20% Validação)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Otimização de Pipeline de Memória (Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # SPRINT 4: Camada Sequencial de Data Augmentation Dinâmico
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="Data_Augmentation_Layer")

    # SPRINT 5: Construção da Arquitetura Convolucional Sequencial (CNN)
    print("[INFO] Executando Sprint 5: Construindo e Compilando a Rede Neural Convolucional...")
    model = models.Sequential([
        # Entradas e Normalização dos Pixels [0, 255] -> [0, 1]
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        layers.Rescaling(1./255),

        # Bloco Convolucional 1: Extração de Bordas e Texturas Primárias
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Bloco Convolucional 2: Extração de Formas e Padrões Complexos
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Bloco Convolucional 3: Extração de Traços Específicos de Defeito (Fendas/Ranhuras)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Transição Matricial (Flattening) e Classificação Densa
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevenção de Overfitting
        layers.Dense(1, activation='sigmoid') # Saída Binária: Probabilidade de Peça Defeituosa
    ], name="CNN_Casting_Quality_Inspector")

