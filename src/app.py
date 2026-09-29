from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# 1. Configuración de página
st.set_page_config(
    page_title="Clasificador de Animales",
    page_icon="🎨",
    layout="centered"
)

# 2. Definición de clases en el mismo orden alfabético que generó ImageFolder
CLASS_NAMES = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
translate = {"cane": "perro", "cavallo": "caballo", "elefante": "elefante", "farfalla": "mariposa", "gallina": "gallina", "gatto": "gato", "mucca": "vaca", "pecora": "oveja", "ragno": "araña", "scoiattolo": "ardilla"}


# 3. Transformación requerida para inferencia (idéntica a eval_transform)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_trained_model(filepath):
    # 1. Instanciar la arquitectura EfficientNet-B0 que usaste en train.py
    model = efficientnet_b0()
    
    # 2. Modificar la capa de salida para que coincida con tus 10 clases
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=10) 
    
    # 3. Cargar el diccionario de pesos en el modelo
    state_dict = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    
    # 4. Mover al dispositivo y poner en modo de evaluación
    model.to(device)
    model.eval()
    
    return model


# --- Interfaz de usuario ---
st.title("Clasificador de Animales")
st.write("Sube una imagen para predecir a qué animal corresponde.")

# Selector del archivo de pesos del modelo
weights_file = Path(r"C:\Users\rober\Downloads\Clasificador_de_Animales\models\model.pth")  # Cambia por la ruta de tu modelo guardado

if not weights_file.exists():
    st.warning(f"No se encontró el archivo de modelo `{weights_file}`. Asegúrate de haber ejecutado el entrenamiento y guardado los pesos.")
else:
    model = load_trained_model(str(weights_file))

    uploaded_file = st.file_uploader(
        "Selecciona una imagen (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Cargar y mostrar la imagen subida
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", use_container_width=True)

        # Preprocesar imagen
        tensor_img = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)  # Añadir dimensión de batch: [1, 3, 224, 224]

        # Inferencia
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

            # Obtener clase ganadora
            pred_idx = torch.argmax(probabilities).item()

            pred_class = translate[CLASS_NAMES[pred_idx]]
            pred_confidence = probabilities[pred_idx].item() * 100

            # Mostrar resultado destacado
            st.success(f"**Predicción:** {pred_class.capitalize()} ({pred_confidence:.2f}% de certeza)")

            # Gráfico de barras con todas las probabilidades
            st.subheader("Distribución de probabilidades:")

            # Modificación: aplicar translate a cada elemento iterado
            prob_dict = {
                translate[CLASS_NAMES[i]].capitalize(): float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            }
            st.bar_chart(prob_dict)
