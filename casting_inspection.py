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
# ROTEIRO DO VÍDEO (Sprint 1):
# "Aqui definimos a padronização das imagens e hiperparâmetros. Ajustamos a resolução
# para 300x300 pixels para preservar os detalhes visuais das ranhuras e trincas,
# definimos o batch size de 32 imagens e dividimos o pipeline em tarefas claras."

IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset_casting"  # Diretório contendo as pastas 'ok' e 'def_front'
