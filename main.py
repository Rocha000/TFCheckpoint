

import os
import pandas as pd
import cv2
from cv2 import imwrite
from funcoes import classify_to_data, open_img, ssim, calculate_entropy, jpeg, gaussian_noise, grayscale, resize, crop_as_zoom

# Diretórios principais de origem e saída
PATH_TO_DIR_VAL = "imagenette2/val/"
PATH_TO_DIR_TRAIN = "imagenette2/train/"
PATH_TO_SAVE_VAL = "imagensdistorcidas/val/"
PATH_TO_SAVE_TRAIN = "imagensdistorcidas/train/"

# Criar diretórios principais para salvar imagens distorcidas
os.makedirs(PATH_TO_SAVE_VAL, exist_ok=True)
os.makedirs(PATH_TO_SAVE_TRAIN, exist_ok=True)

# Definição das colunas do DataFrame
HEADERS = [
    "FILE_PATH", "ORIG_CLASSES", "ORIG_ENTROPY",
    "GAUSS_SSIM", "GAUSS_CLASSES", "GAUSS_ENTROPY",
    "JPEG20_SSIM", "JPEG20_CLASSES", "JPEG20_ENTROPY",
    "JPEG70_SSIM", "JPEG70_CLASSES", "JPEG70_ENTROPY",
    "GRAYSCALE_SSIM", "GRAYSCALE_CLASSES", "GRAYSCALE_ENTROPY",
    "ZOOM_SSIM", "ZOOM_CLASSES", "ZOOM_ENTROPY",
    "RESIZE_SSIM", "RESIZE_CLASSES", "RESIZE_ENTROPY",
]
df = pd.DataFrame(columns=HEADERS)

# Função para redimensionar a imagem para o tamanho original e 3 canais (se necessário)
def resize_to_original(img, original_shape):
    img = cv2.resize(img, (original_shape[1], original_shape[0]))
    if len(img.shape) == 2:  # Se for grayscale, converte para 3 canais
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# Função para processar imagens e calcular distorções e métricas
def process_files(input_dir, output_dir):
    global df
    for root, _, files in os.walk(input_dir):
        # Subdiretório correspondente para salvar as imagens distorcidas
        save_dir = os.path.join(output_dir, os.path.relpath(root, input_dir))
        os.makedirs(save_dir, exist_ok=True)

        for file in files:
            path = os.path.join(root, file)
            save_base = os.path.join(save_dir, file.removesuffix(".JPEG"))

            # Carregar imagem original
            orig_img = open_img(path=path)
            if orig_img is None:
                print(f"Erro ao carregar a imagem {file}")
                continue

            # Classe e entropia da imagem original
            classes_orig = classify_to_data(cv2.resize(orig_img, (224, 224)))
            entropy_orig = calculate_entropy(orig_img)
            row_data = [path, classes_orig, entropy_orig]

            # Aplicar distorções e calcular métricas
            # 1. Gaussian Noise
            after_gauss = gaussian_noise(orig_img, mean=10)
            after_gauss = resize_to_original(after_gauss, orig_img.shape)
            imwrite(f"{save_base}_dist_gaussian.JPEG", after_gauss)
            ssim_gauss = ssim(orig_img, after_gauss)
            classes_gauss = classify_to_data(cv2.resize(after_gauss, (224, 224)))
            entropy_gauss = calculate_entropy(after_gauss)
            row_data.extend([ssim_gauss, classes_gauss, entropy_gauss])

            # 2. JPEG com qualidade 20%
            after_jpeg20 = jpeg(orig_img, 20)
            after_jpeg20 = resize_to_original(after_jpeg20, orig_img.shape)
            imwrite(f"{save_base}_dist_jpeg20.JPEG", after_jpeg20)
            ssim_jpeg20 = ssim(orig_img, after_jpeg20)
            classes_jpeg20 = classify_to_data(cv2.resize(after_jpeg20, (224, 224)))
            entropy_jpeg20 = calculate_entropy(after_jpeg20)
            row_data.extend([ssim_jpeg20, classes_jpeg20, entropy_jpeg20])

            # 3. JPEG com qualidade 70%
            after_jpeg70 = jpeg(orig_img, 70)
            after_jpeg70 = resize_to_original(after_jpeg70, orig_img.shape)
            imwrite(f"{save_base}_dist_jpeg70.JPEG", after_jpeg70)
            ssim_jpeg70 = ssim(orig_img, after_jpeg70)
            classes_jpeg70 = classify_to_data(cv2.resize(after_jpeg70, (224, 224)))
            entropy_jpeg70 = calculate_entropy(after_jpeg70)
            row_data.extend([ssim_jpeg70, classes_jpeg70, entropy_jpeg70])

            # 4. Grayscale
            after_grayscale = grayscale(orig_img)
            after_grayscale = resize_to_original(after_grayscale, orig_img.shape)
            imwrite(f"{save_base}_dist_grayscale.JPEG", after_grayscale)
            ssim_grayscale = ssim(orig_img, after_grayscale)
            classes_grayscale = classify_to_data(cv2.resize(after_grayscale, (224, 224)))
            entropy_grayscale = calculate_entropy(after_grayscale)
            row_data.extend([ssim_grayscale, classes_grayscale, entropy_grayscale])

            # 5. Zoom (com corte central)
            after_zoom = crop_as_zoom(orig_img, zoom_factor=1.5)
            imwrite(f"{save_base}_dist_zoom.JPEG", after_zoom)
            ssim_zoom = ssim(orig_img, after_zoom)
            classes_zoom = classify_to_data(cv2.resize(after_zoom, (224, 224)))
            entropy_zoom = calculate_entropy(after_zoom)
            row_data.extend([ssim_zoom, classes_zoom, entropy_zoom])

            # 6. Resize
            after_resize = resize(orig_img, 64, 64)
            after_resize_back = resize_to_original(after_resize, orig_img.shape)
            imwrite(f"{save_base}_dist_resize.JPEG", after_resize_back)
            ssim_resize = ssim(orig_img, after_resize_back)
            classes_resize = classify_to_data(cv2.resize(after_resize_back, (224, 224)))
            entropy_resize = calculate_entropy(after_resize_back)
            row_data.extend([ssim_resize, classes_resize, entropy_resize])

            # Adicionar linha ao DataFrame
            df = pd.concat([df, pd.DataFrame([row_data], columns=HEADERS)], ignore_index=True)

# Processar todos os arquivos nas pastas de validação e treinamento
process_files(PATH_TO_DIR_VAL, PATH_TO_SAVE_VAL)
process_files(PATH_TO_DIR_TRAIN, PATH_TO_SAVE_TRAIN)

# Salvar o DataFrame consolidado em um único CSV
df.to_csv("imagensdistorcidas/imagensdistorcidas-all-data.csv", sep=";", index=False)


