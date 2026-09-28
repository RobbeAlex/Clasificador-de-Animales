"""
dataset.py
Carga imágenes desde carpetas (train y test) y genera particiones
de train, valid y test dividiendo train mediante torch.utils.data.random_split.
"""

"""
dataset.py
Carga imágenes desde una única carpeta y genera particiones
de train, valid y test dividiendo la data dinámicamente mediante torch.utils.data.random_split.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms


class TransformedSubset(Dataset):
    """
    Aplica una transformación específica a un subconjunto obtenido por random_split.
    Permite separar transformaciones de entrenamiento, validación y prueba.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def create_dataloaders(
    data_dir: str = r"C:\Users\rober\.gemini\antigravity\scratch\ai-code-auditor\Deep_Learning_CUGDL\07_pytorch_modular_intro\personal\raw-img",
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
    batch_size: int = 32,
    image_size: tuple = (224, 224),
    num_workers: int = 0,
    seed: int = 42
):
    """
    Carga todos los datos desde un único directorio en disco, los divide en
    (train_subset, val_subset, test_subset) según los ratios indicados y
    retorna los tres DataLoaders junto con las clases.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de datos en: {data_path}")

    # Transformaciones con Data Augmentation para entrenamiento
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformaciones deterministas para validación y prueba
    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Cargamos el dataset completo sin transformar aún la imagen en tensor
    raw_data = datasets.ImageFolder(root=str(data_path))
    class_names = raw_data.classes

    # 2. Cálculo de tamaños para la partición en tres conjuntos
    total_samples = len(raw_data)
    train_size = int(train_split_ratio * total_samples)
    val_size = int(val_split_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # 3. División aleatoria de todo el dataset con semilla fija para reproducibilidad
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        raw_data,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 4. Asignamos la transformación correspondiente a cada partición
    train_data = TransformedSubset(train_subset, transform=train_transform)
    val_data = TransformedSubset(val_subset, transform=eval_transform)
    test_data = TransformedSubset(test_subset, transform=eval_transform)

    # 5. Creación de los DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names


if __name__ == "__main__":
    print("Probando particionamiento dinámico en 3 partes con random_split...")
    # Se ajustan los ratios para obtener por ejemplo: 70% train, 15% val, 15% test
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_split_ratio=0.7,
        val_split_ratio=0.15,
        batch_size=16
    )

    print(f"\nClases encontradas ({len(class_names)}): {class_names}")
    print(f"Muestras totales en el disco: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
    print(f"Muestras en Entrenamiento: {len(train_loader.dataset)}")
    print(f"Muestras en Validación:    {len(val_loader.dataset)}")
    print(f"Muestras en Prueba:        {len(test_loader.dataset)}")

    # Comprobación del primer batch
    images, labels = next(iter(train_loader))
    print(f"\nDimensiones del batch: {images.shape}")
    print(f"Etiquetas del batch: {labels}")