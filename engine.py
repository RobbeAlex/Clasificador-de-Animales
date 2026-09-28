# Logica detras del entrenamiento

# Entrenamiento de la red
import torch

def train_step(model: torch.nn.Module,
    X,
    y,
    loss_fn: torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    device: torch.device):
    """
    Esta función entrena un modelo de pytorch en una sola época.

    Args:
    model - Modelo de pytorch (red neuronal)
    X - Datos (características)
    y - Datos (target)
    loss - Función de pérdida
    optimizer - método de optimización de la función de pérdida
    device - hardware utilizado

    Return
    probs - probabilidades predichas
    """

    model.train()
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()

    # 1. Forwar propagation
    y_pred = model(X)

    # 2. Calcular perdida
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    probs = torch.softmax(y_pred, dim=1)
    return probs.detach()
   
def test_step(model,
    X,
    y,
    loss_fn: torch.nn.Module,
    device: torch.device):
    
    """
    Esta función entrena un modelo de pytorch en una sola época.

    Args:
    model - Modelo de pytorch (red neuronal)
    X - Datos (características)
    y - Datos (target)
    loss_fn - Función de pérdida
    optimizer - método de optimización de la función de pérdida
    device - hardware utilizado

    Return
    probs - probabilidades predichas
    """


    model.eval()
    X, y = X.to(device), y.to(device)

    with torch.inference_mode():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        probs = torch.softmax(y_pred, dim=1)

    return probs

#def test_step:
#    return
