from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from math import log10, sqrt

# Carregar modelo ResNet50 pré-treinado no ImageNet
MODEL = ResNet50(weights='imagenet')

# Função para carregar imagem
def open_img(path):
    return cv2.imread(path)

# Função para classificar imagem e retornar classe e confiança
def classify(img):
    try:
        x = cv2.resize(img, (224,224))
        x = x[:,:,::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds)[0]
        for c in classes:
            print(f"\t{c[1]} ({c[0]}): {c[2]*100:.2f}%")
        return classes[0][1], classes[0][2], preds  # Retorna classe, confiança e as probabilidades
    except Exception as e:
        print("Falha na classificação.")
        return None, None, None

# Função para calcular SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2) * 100

# Função para calcular PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # PSNR máximo se não houver diferença
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))

# Função para calcular Entropia
def calculate_entropy(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return shannon_entropy(gray_img)

# Função para calcular Diferença nas Probabilidades
def calculate_probability_difference(original_probs, distorted_probs):
    return np.sum(np.abs(original_probs - distorted_probs))

# Funções de distorções

# Compressão JPEG
def jpeg(img, quality):
    _, x = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(x, cv2.IMREAD_COLOR)

# Redimensionar imagem
def resize(img, w, h):
    orig_h, orig_w = img.shape[:2]
    x = cv2.resize(img, (w, h))
    return cv2.resize(x, (orig_w, orig_h))

# Detecção de bordas (Canny)
def canny(img):
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Canny(x, 100, 200)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

# Função para aplicar ruído gaussiano
def add_gaussian_noise(img, mean=0, var=0.1):
    row, col, ch = img.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col, ch))
    noisy_img = img + gaussian
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# Função para conversão de cores (RGB para escala de cinza)
def convert_to_grayscale(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

# Função para recortar imagem
def crop_image(img):
    cropped_img = img[50:200, 50:200]
    return cv2.resize(cropped_img, (224, 224))

# Função para exibir cada distorção separadamente com as métricas corretas
def display_individual_distortions(img, img_path):
    print("Imagem Original:")
    original_img = open_img(img_path)
    original_class, original_conf, original_probs = classify(original_img)

    distorcoes = {
        "Compressão JPEG 70%": jpeg(original_img, 70),
        "Redimensionamento para 64x64": resize(original_img, 64, 64),
        "Ruído Gaussiano": add_gaussian_noise(original_img),
        "Conversão para Escala de Cinza": convert_to_grayscale(original_img),
        "Detecção de Bordas (Canny)": canny(original_img),
        "Corte (Cropping)": crop_image(original_img)
    }

    # Definir as métricas a serem aplicadas para cada distorção
    metrics_for_each = {
        "Compressão JPEG 70%": ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],
        "Redimensionamento para 64x64": ['SSIM', 'PSNR', 'Diferença nas Probabilidades'],
        "Ruído Gaussiano": ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],
        "Conversão para Escala de Cinza": ['Entropia', 'Diferença nas Probabilidades'],
        "Detecção de Bordas (Canny)": ['Entropia', 'Diferença nas Probabilidades'],
        "Corte (Cropping)": ['SSIM', 'PSNR', 'Diferença nas Probabilidades']
    }

    for nome_distorcao, img_distorcida in distorcoes.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(img_distorcida)
        
        # Aplicar métricas conforme definido em 'metrics_for_each'
        ssim_score = calculate_ssim(original_img, img_distorcida) if 'SSIM' in metrics_for_each[nome_distorcao] else "N/A"
        psnr_value = calculate_psnr(original_img, img_distorcida) if 'PSNR' in metrics_for_each[nome_distorcao] else "N/A"
        entropy_value = calculate_entropy(img_distorcida) if 'Entropia' in metrics_for_each[nome_distorcao] else "N/A"
        class_name, confidence, distorted_probs = classify(img_distorcida)
        prob_diff = calculate_probability_difference(original_probs, distorted_probs) if distorted_probs is not None and 'Diferença nas Probabilidades' in metrics_for_each[nome_distorcao] else "N/A"

        # Exibir imagem distorcida com as métricas
        plt.title(f"{nome_distorcao}\nSSIM: {ssim_score:.2f}% | PSNR: {psnr_value:.2f}\n"
                  f"Classe: {class_name}, Confiança: {confidence:.2f}\n"
                  f"Entropia: {entropy_value:.2f} | Diferença nas Prob: {prob_diff:.4f}")
        plt.axis('off')
        plt.show()

# Caminho atualizado para a imagem original
img_path = 'imagenette2/train/n02102040/n02102040_7886.JPEG'

# Exibir as distorções e suas métricas
display_individual_distortions(open_img(img_path), img_path)
