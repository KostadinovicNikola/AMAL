# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        return torch.mean((yhat-y)**2, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        return grad_output * 2 * (yhat-y) / yhat.shape[0], grad_output * -2 * (yhat - y) / yhat.shape[0]
        

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE
class Linear(Function):
    """Début d'implémentation de la fonction Linear"""
    @staticmethod
    def forward(ctx, X, W, B):
        ctx.save_for_backward(X, W, B)
        return X @ W + B

    def backward(ctx, grad_output):
        X, W, B = ctx.saved_tensors
        # On renvoit les dérivées partielles par rapport à X, W et b
        return grad_output @ W.T, X.T @ grad_output, grad_output.T @ torch.ones(grad_output.shape[0]).double()

        
## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply