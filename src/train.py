from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from dataset import create_dataloaders


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")

    # 1. Cargar DataLoaders inyectando explícitamente el directorio
    print("Cargando datos...")
    data_directory = r"C:\Users\rober\.gemini\antigravity\scratch\ai-code-auditor\Deep_Learning_CUGDL\07_pytorch_modular_intro\personal\raw-img"

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_directory,
        batch_size=16,
        image_size=(224, 224)
    )
    num_classes = len(class_names)
    print(f"Entrenando con {num_classes} clases: {class_names}")

    # 2. Configurar arquitectura con pesos preentrenados (Transfer Learning)
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Congelar capas base extractor de características
    for param in model.features.parameters():
        param.requires_grad = False

    # Modificar la capa de salida para que coincida con las 10 clases detectadas
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    model.to(device)

    # 3. Función de pérdida y optimizador
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # 4. Ciclo de entrenamiento
    epochs = 5
    print(f"\nIniciando entrenamiento por {epochs} épocas...")

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_acc += (preds == labels).sum().item() / len(labels)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validación
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_acc += (preds == labels).sum().item() / len(labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(
            f"Época {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%"
        )

    # 5. Evaluación final independiente con el conjunto de prueba
    print("\nEvaluando modelo final en el conjunto de prueba (Test)...")
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            test_acc += (preds == labels).sum().item() / len(labels)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Resultados de Prueba | Test Loss: {test_loss:.4f} - Test Acc: {test_acc * 100:.2f}%")

    # 6. Guardado seguro usando state_dict en lugar del modelo completo
    output_path = Path("model.pth")
    torch.save(model.state_dict(), output_path)
    print(f"\nEntrenamiento finalizado. Pesos del modelo guardados en: {output_path.resolve()}")


if __name__ == "__main__":
    train()
