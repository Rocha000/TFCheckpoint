from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage.metrics import structural_similarity
from scipy.stats import entropy
import numpy as np
import cv2


MODEL = ResNet50(weights="imagenet")
# Funções de distorção
def jpeg(img, quality):
    _, x = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(x, cv2.IMREAD_COLOR)

def resize(img, w, h):
    orig_h, orig_w = img.shape[:2]
    x = cv2.resize(img, (w, h))
    return cv2.resize(x, (orig_w, orig_h))

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def crop_as_zoom(img, zoom_factor=2):
    # Dimensões da imagem original
    h, w = img.shape[:2]

    # Calcula a área de corte central com base no fator de zoom
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2

    # Realiza o corte central para simular o zoom
    cropped_img = img[start_y:start_y + new_h, start_x:start_x + new_w]

    # Redimensiona para o tamanho original
    return cv2.resize(cropped_img, (w, h))


# Funções de métricas
def ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2) * 100

def calculate_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return entropy(hist, base=2)


def classify_to_data(img):
    classify = []
    try:
        x = cv2.resize(img, (224, 224))
        x = x[:, :, ::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds)[0]
        for c in classes:
            classify.append([c[1], c[0], c[2] * 100])

    except Exception as e:
        print("Classification failed.")

    return classify

def open_img(path):
    return cv2.imread(path)  