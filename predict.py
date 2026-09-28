import torch
from model_builder import Model_Classification
from config import HIDDEN_UNITS

def predict(input_data: torch.Tensor, model_path: str = "modelo_entrenado.pth"):
    """
    Realiza predicciones utilizando un modelo ya entrenado.
    
    Args:
    input_data - Un tensor de PyTorch con forma (N, 64) 
                 donde N es el número de muestras.
    model_path - Ruta donde se encuentra guardado el modelo.
                 
    Retorna:
    probs - Las probabilidades de las clases.
    pred_classes - La clase predicha con mayor probabilidad.
    """
    
    # 1. Instanciar el modelo con la misma arquitectura con la que se entrenó
    model = Model_Classification(input_shape=64, 
                                 hidden_units=HIDDEN_UNITS, 
                                 output_features=10)
    
    # 2. Cargar los pesos entrenados
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 3. Poner el modelo en modo de evaluación
    model.eval() 
    
    # 4. Realizar la predicción sin calcular gradientes (más eficiente)
    with torch.inference_mode():
        logits = model(input_data)
        
        # Calcular las probabilidades utilizando Softmax
        probs = torch.softmax(logits, dim=1)
        
        # Obtener la clase predicha (la de mayor probabilidad)
        pred_classes = torch.argmax(probs, dim=1)
        
    return probs, pred_classes


if __name__ == "__main__":
    # ---- Ejemplo de uso ----
    print("Generando datos de prueba (1 muestra de 64 características)...")
    
    # Generamos un tensor con valores aleatorios simulando una muestra de entrada
    sample_data = torch.rand(1, 64)
    
    try:
        probabilidades, clase_predicha = predict(sample_data)
        
        print("\n=== Resultados de la Predicción ===")
        print(f"Probabilidades por clase:\n{probabilidades}")
        print(f"Clase con mayor probabilidad: {clase_predicha.item()}")
        
    except FileNotFoundError:
        print("\n[Error] No se encontró el archivo 'modelo_entrenado.pth'.")
        print("¡Asegúrate de ejecutar primero 'python train.py' para entrenar y guardar el modelo!")
