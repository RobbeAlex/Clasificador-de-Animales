"""
clean_dataset.py
Escanea la carpeta de imágenes y reporta o elimina archivos corruptos/truncados.
"""
from pathlib import Path
from PIL import Image

data_dir = Path(r"C:\Users\rober\.gemini\antigravity\scratch\ai-code-auditor\Deep_Learning_CUGDL\07_pytorch_modular_intro\personal\raw-img")

print("Analizando imágenes en busca de archivos dañados...")
corrupted_files = []

for file_path in data_dir.rglob("*"):
    if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        try:
            with Image.open(file_path) as img:
                # Forzar lectura completa de píxeles
                img.verify()
            
            # Segunda comprobación de conversión
            with Image.open(file_path) as img:
                img.convert("RGB").load()

        except Exception as e:
            print(f"Archivo dañado encontrado: {file_path} ({e})")
            corrupted_files.append(file_path)

if corrupted_files:
    print(f"\nSe encontraron {len(corrupted_files)} archivos dañados.")
    eliminar = input("¿Deseas eliminarlos automáticamente? (s/n): ")
    if eliminar.lower() == "s":
        for f in corrupted_files:
            f.unlink()
        print("Archivos dañados eliminados.")
else:
    print("Todas las imágenes están en perfecto estado.")