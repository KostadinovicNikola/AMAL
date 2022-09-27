import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context

nb_data = 100

# Les données supervisées
x = torch.randn(nb_data, 13, dtype=torch.float64)
y = torch.randn(nb_data, 3, dtype=torch.float64)

print("Shape x:", x.shape)
print("Shape y:", y.shape)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, dtype=torch.float64)
b = torch.randn(3, dtype=torch.float64)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    ctx_linear = Context()
    yhat = Linear.forward(ctx_linear, x, w, b)

    ctx_mse = Context()
    # `loss` doit correspondre au coût MSE calculé à cette itération
    loss = MSE.forward(ctx_mse, yhat, y)
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss.mean(), n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)

    yhat_back, _ = MSE.backward(ctx_mse, loss)
    print(yhat_back)
    _, grad_w, grad_b = Linear.backward(ctx_linear, yhat_back.double())

    ##  TODO:  Mise à jour des paramètres du modèle

    w -= epsilon * grad_w / nb_data
    b -= epsilon * grad_b / nb_data

