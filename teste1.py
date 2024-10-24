import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from math import log10, sqrt

# Carregar a ResNet-50 pré-treinada no ImageNet
model = ResNet50(weights='imagenet')

# Função para carregar uma imagem, redimensionar e preparar para a rede
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB
    image = cv2.resize(image, (224, 224))  # Redimensionar para o tamanho esperado pela ResNet
    image_array = np.expand_dims(image, axis=0)  # Adicionar uma dimensão para lote
    image_array = preprocess_input(image_array)  # Pré-processar a imagem para a ResNet
    return image, image_array

# Função para classificar uma imagem e retornar a classe predita, confiança e probabilidades
def classify_image(image_array):
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    class_name = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]
    return class_name, confidence, predictions

# Funções de distorções

# Compressão JPEG
def compress_jpeg(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, 1)
    return compressed_image

# Redimensionar imagem
def resize_image(image, size=(64, 64)):
    resized_image = cv2.resize(image, size)
    return cv2.resize(resized_image, (224, 224))  # Redimensionar de volta para 224x224 para a ResNet

# Adicionar ruído gaussiano
def add_gaussian_noise(image, mean=0, var=0.1):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col, ch))  # Gerar o ruído gaussiano
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Limitar os valores de pixel entre 0 e 255
    return noisy_image

# Conversão de Cores para escala de cinza
def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

# Detecção de bordas (Canny)
def apply_canny_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Converter de volta para 3 canais
    return edges_colored

# Crop (corte) da imagem
def crop_image(image):
    cropped_image = image[50:200, 50:200]  # Cortar uma parte do meio da imagem
    return cv2.resize(cropped_image, (224, 224))  # Redimensionar para 224x224 para compatibilidade

# Funções de Métricas

# Calcular SSIM
def calculate_ssim(original, distorted):
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    distorted_gray = cv2.cvtColor(distorted, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(original_gray, distorted_gray, full=True)
    return score

# Calcular PSNR
def calculate_psnr(original, distorted):
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:  # Significa que não há diferença entre as imagens
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * log10(PIXEL_MAX / sqrt(mse))
    return psnr

# Calcular Entropia
def calculate_entropy(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return shannon_entropy(gray_image)

# Calcular Diferença nas Probabilidades
def calculate_probability_difference(original_probs, distorted_probs):
    diff = np.abs(original_probs - distorted_probs).sum()
    return diff

# Função para exibir todas as distorções em uma única janela
# Função para exibir cada distorção separadamente com melhor visualização do título
def display_individual_distortions(original_image, original_probs, distortions, distortion_names, metrics_for_each):
    # Exibir a imagem original com as métricas
    class_name, confidence, _ = classify_image(preprocessed_image)
    original_entropy = calculate_entropy(original_image)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.title(f"Original\nClasse: {class_name}, Confiança: {confidence:.2f}\nEntropia: {original_entropy:.2f}", fontsize=10)
    plt.axis('off')
    plt.show()

    # Iterar sobre as distorções e calcular as métricas relevantes
    for distorted_image, name, metrics in zip(distortions, distortion_names, metrics_for_each):
        preprocessed_distorted_image = preprocess_input(np.expand_dims(cv2.resize(distorted_image, (224, 224)), axis=0))
        class_name, confidence, distorted_probs = classify_image(preprocessed_distorted_image)

        # Calcular as métricas relevantes para a distorção
        metrics_results = []
        if 'SSIM' in metrics:
            ssim_score = calculate_ssim(original_image, distorted_image)
            metrics_results.append(f"SSIM: {ssim_score:.2f}")
        if 'PSNR' in metrics:
            psnr_value = calculate_psnr(original_image, distorted_image)
            metrics_results.append(f"PSNR: {psnr_value:.2f}")
        if 'Entropia' in metrics:
            entropy = calculate_entropy(distorted_image)
            metrics_results.append(f"Entropia: {entropy:.2f}")
        if 'Diferença nas Probabilidades' in metrics:
            prob_difference = calculate_probability_difference(original_probs, distorted_probs)
            metrics_results.append(f"Diferença nas prob: {prob_difference:.4f}")
        
        metrics_results.append(f"Classe: {class_name}, Confiança: {confidence:.2f}")

        # Exibir a imagem distorcida com as métricas relevantes
        plt.figure(figsize=(6, 6))
        plt.imshow(distorted_image)
        
        # Ajustar o título para quebrar em várias linhas e ajustar o tamanho da fonte
        title_text = f"{name}\n" + "\n".join(metrics_results)
        plt.title(title_text, fontsize=9, wrap=True)  # Fonte menor e quebra automática
        plt.axis('off')
        plt.show()

# Carregar uma imagem da pasta 'val' da Imagenette
image_path = 'imagenette2/train/n02102040/n02102040_7886.JPEG'
original_image, preprocessed_image = load_and_preprocess_image(image_path)

# Classificar a imagem original
class_name, confidence, original_probs = classify_image(preprocessed_image)

# Aplicar diferentes distorções
jpeg_compressed_image = compress_jpeg(original_image, 70)
resized_image = resize_image(original_image, size=(64, 64))
noisy_image = add_gaussian_noise(original_image)
grayscale_image = convert_to_grayscale(original_image)
canny_image = apply_canny_edge_detection(original_image)
cropped_image = crop_image(original_image)

# Armazenar as distorções em uma lista para exibição conjunta
distortions = [jpeg_compressed_image, resized_image, noisy_image, grayscale_image, canny_image, cropped_image]
distortion_names = [
    "Compressão JPEG (70%)", 
    "Redimensionamento para 64x64", 
    "Ruído Gaussiano", 
    "Conversão de Cores (Grayscale)", 
    "Detecção de Bordas (Canny)", 
    "Cropping (Corte)"
]

# Definir quais métricas aplicar para cada distorção
metrics_for_each = [
    ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],  # Compressão JPEG
    ['SSIM', 'PSNR', 'Diferença nas Probabilidades'],  # Redimensionamento
    ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],  # Ruído Gaussiano
    ['Entropia', 'Diferença nas Probabilidades'],  # Conversão de Cores
    ['Entropia', 'Diferença nas Probabilidades'],  # Canny
    ['SSIM', 'PSNR', 'Diferença nas Probabilidades']  # Cropping
]

# Exibir as distorções individualmente com as métricas
display_individual_distortions(original_image, original_probs, distortions, distortion_names, metrics_for_each)
