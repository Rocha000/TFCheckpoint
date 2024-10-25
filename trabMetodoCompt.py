import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Carregar o modelo ResNet50 pré-treinado no ImageNet
MODEL = ResNet50(weights='imagenet')

# Função para garantir que todas as imagens distorcidas sejam redimensionadas para 224x224
def resize_to_original(img, target_size=(224, 224)):
    return cv2.resize(img, target_size)

# Função para carregar imagem e converter de BGR para RGB
def open_img(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter de BGR para RGB

# Função para calcular SSIM com redimensionamento
def calculate_ssim(img1, img2):
    img1_resized = resize_to_original(img1)
    img2_resized = resize_to_original(img2)
    return ssim(img1_resized, img2_resized, channel_axis=2) * 100

# Função para calcular PSNR
def calculate_psnr(img1, img2):
    img1_resized = resize_to_original(img1)
    img2_resized = resize_to_original(img2)
    mse = np.mean((img1_resized - img2_resized) ** 2)
    if mse == 0:
        return 100  # PSNR máximo se não houver diferença
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Função para calcular entropia
def calculate_entropy(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    hist_prob = hist / hist.sum()
    return -np.sum(hist_prob * np.log2(hist_prob + 1e-7))

# Função para calcular a diferença nas probabilidades de classificação
def calculate_probability_difference(original_probs, distorted_probs):
    return np.sum(np.abs(original_probs - distorted_probs))

# Função para compressão JPEG e conversão para RGB
def jpeg(img, quality):
    _, x = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    jpeg_image = cv2.imdecode(x, cv2.IMREAD_COLOR)
    return cv2.cvtColor(jpeg_image, cv2.COLOR_BGR2RGB)  # Converter para RGB após compressão

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

# Função para detecção de bordas (Canny)
def canny(img):
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Canny(x, 100, 200)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

# Função para recortar imagem
def crop_image(img):
    cropped_img = img[50:200, 50:200]
    return resize_to_original(cropped_img)

# Função para classificar imagem e retornar classe e confiança
def classify(img):
    try:
        x = cv2.resize(img, (224, 224))
        x = x[:, :, ::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds, top=5)[0]
        for c in classes:
            print(f"\t{c[1]} ({c[0]}): {c[2]*100:.2f}%")
        return classes[0][1], classes[0][2], preds  # Retorna classe, confiança e probabilidades
    except Exception as e:
        print(f"Erro na classificação: {e}")
        return None, None, None

# Exibir distorções e calcular métricas
def display_individual_distortions(img, img_path):
    print("Imagem Original:")
    original_img = open_img(img_path)
    original_class, original_conf, original_probs = classify(original_img)

    # Distorções aplicadas e garantido que as imagens estejam em RGB
    distorcoes = {
        "Compressão JPEG 70%": jpeg(original_img, 70),
        "Redimensionamento para 64x64": resize_to_original(original_img, (64, 64)),
        "Ruído Gaussiano": add_gaussian_noise(original_img),
        "Conversão para Escala de Cinza": convert_to_grayscale(original_img),
        "Detecção de Bordas (Canny)": canny(original_img),
        "Corte (Cropping)": crop_image(original_img)
    }

    metrics_for_each = {
        "Compressão JPEG 70%": ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],
        "Redimensionamento para 64x64": ['SSIM', 'PSNR', 'Diferença nas Probabilidades'],
        "Ruído Gaussiano": ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],
        "Conversão para Escala de Cinza": ['Entropia', 'Diferença nas Probabilidades'],
        "Detecção de Bordas (Canny)": ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades'],
        "Corte (Cropping)": ['SSIM', 'PSNR', 'Entropia', 'Diferença nas Probabilidades']
    }

    for nome_distorcao, img_distorcida in distorcoes.items():
        img_distorcida_resized = resize_to_original(img_distorcida)  # Garantir tamanho 224x224
        plt.figure(figsize=(6, 6))
        plt.imshow(img_distorcida_resized)

        # Calcular as métricas
        ssim_score = calculate_ssim(original_img, img_distorcida_resized) if 'SSIM' in metrics_for_each[nome_distorcao] else "N/A"
        psnr_value = calculate_psnr(original_img, img_distorcida_resized) if 'PSNR' in metrics_for_each[nome_distorcao] else "N/A"
        entropy_value = calculate_entropy(img_distorcida_resized) if 'Entropia' in metrics_for_each[nome_distorcao] else "N/A"
        class_name, confidence, distorted_probs = classify(img_distorcida_resized)
        prob_diff = calculate_probability_difference(original_probs, distorted_probs) if distorted_probs is not None and 'Diferença nas Probabilidades' in metrics_for_each[nome_distorcao] else "N/A"

        # Verificar se os valores são números e aplicar formatação, caso contrário, exibir 'N/A'
        ssim_str = f"{ssim_score:.2f}%" if isinstance(ssim_score, (int, float)) else ssim_score
        psnr_str = f"{psnr_value:.2f}" if isinstance(psnr_value, (int, float)) else psnr_value
        entropy_str = f"{entropy_value:.2f}" if isinstance(entropy_value, (int, float)) else entropy_value
        prob_diff_str = f"{prob_diff:.4f}" if isinstance(prob_diff, (int, float)) else prob_diff

        # Exibir imagem com as métricas
        plt.title(f"{nome_distorcao}\nSSIM: {ssim_str} | PSNR: {psnr_str}\n"
                  f"Classe: {class_name}, Confiança: {confidence:.2f}\n"
                  f"Entropia: {entropy_str} | Diferença nas Prob: {prob_diff_str}")
        plt.axis('off')
        plt.show()

# Caminho para a imagem original
img_path = 'imagenette2/train/n02102040/n02102040_7886.JPEG'

# Exibir as distorções e suas métricas
display_individual_distortions(open_img(img_path), img_path)
