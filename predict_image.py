import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model_builder import Model_Classification
from config import HIDDEN_UNITS

def process_and_predict_image(image_path: str, model_path: str = "modelo_entrenado.pth"):
    # 1. Cargar la imagen y convertirla a escala de grises
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return
    
    # 2. Transformar la imagen para que coincida con los datos de entrenamiento (8x8)
    # Los datos originales (digits) son 8x8 pixeles, con fondo oscuro (0) y trazo claro (hasta 16)
    preprocess = transforms.Compose([
        transforms.Resize((8, 8)), # Redimensionar a 8x8
    ])
    
    img_resized = preprocess(img)
    
    # Convertir a numpy array y luego ajustar la escala (de 0-255 a 0-16 aprox)
    # Dependiendo de tu imagen real, quizás necesites invertir los colores (255 - pixel)
    # si tu imagen tiene fondo blanco y el dígito negro.
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Si la imagen tiene fondo blanco y letras negras, invierte los colores:
    # img_array = 255.0 - img_array 
    
    # Escalar a rango 0-16 (como en tu dataset train.csv)
    img_array = (img_array / 255.0) * 16.0
    
    # 3. Aplanar la imagen (de 8x8 a un vector de 64 posiciones)
    img_flattened = img_array.flatten()
    
    # 4. Convertir a Tensor de PyTorch y agregar dimensión de lote (batch size = 1)
    # Forma esperada: (1, 64)
    input_tensor = torch.from_numpy(img_flattened).unsqueeze(0)
    
    # 5. Cargar el modelo
    model = Model_Classification(input_shape=64, 
                                 hidden_units=HIDDEN_UNITS, 
                                 output_features=10)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 6. Predecir
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        
    print(f"Predicción final para '{image_path}': Clase {pred_class.item()}")
    print(f"Probabilidades: \n{probs}")

if __name__ == "__main__":
    import sys
    # Se ejecuta pasándole la ruta de la imagen: python predict_image.py mi_numero.png
    if len(sys.argv) > 1:
        ruta_imagen = sys.argv[1]
        process_and_predict_image(ruta_imagen)
    else:
        print("Por favor, pasa la ruta de una imagen como argumento.")
        print("Uso: python predict_image.py ruta/a/tu/imagen.jpg")
