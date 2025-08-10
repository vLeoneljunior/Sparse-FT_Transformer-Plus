import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
import sys
import os

# Chemin absolu pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_utils import prepare_california_data, get_data_splits
from rtdl_revisiting_models.bin.ft_transformer_plr import TransformerPLR
from num_embedding_factory import get_num_embedding

# =========================
# Préparation des données California Housing
# =========================
X, y = prepare_california_data()
data = get_data_splits(X, y)

X_train_tensor = data['X_train']
X_val_tensor = data['X_val']
X_test_tensor = data['X_test']
y_train_tensor = data['y_train']
y_val_tensor = data['y_val']
y_test_tensor = data['y_test']
n_features = data['n_features']

# =========================
# Configuration du modèle FT-Transformer avec P-LR
# =========================
model_config = {
    "d_numerical": n_features,
    "categories": None,
    "token_bias": True,
    "n_layers": 3,
    "d_token": 192,
    "n_heads": 8,
    "d_ffn_factor": 1.333,
    "attention_dropout": 0.2,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "activation": "relu",
    "prenormalization": True,
    "initialization": "kaiming",
    "kv_compression": None,
    "kv_compression_sharing": None,
    "d_out": 1,
}

print("=== Paramètres de configuration ===")
for k, v in model_config.items():
    print(f"{k}: {v}")

model = TransformerPLR(**model_config)

# Ajouter l'embedding P-LR
num_embedding = get_num_embedding(
    embedding_type="P-LR",
    X_train=X_train_tensor.numpy(),  # Convertir en numpy pour get_num_embedding
    d_embedding=model_config["d_token"]
)
model.tokenizer.num_tokenizer = num_embedding

# =========================
# Boucle d'entraînement avec validation
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()
n_epochs = 20
batch_size = 64

n_parameters = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres du modèle: {n_parameters}")

best_val_loss = float('inf')
best_epoch = -1
best_model_state = None

start_time = time.time()
for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0.0
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]
        optimizer.zero_grad()
        output = model(batch_x, None)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= X_train_tensor.size(0)

    # Validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, X_val_tensor.size(0), batch_size):
            batch_x = X_val_tensor[i:i+batch_size]
            batch_y = y_val_tensor[i:i+batch_size]
            preds = model(batch_x, None)
            val_preds.append(preds)
    val_preds = torch.cat(val_preds)
    val_loss = loss_fn(val_preds, y_val_tensor)
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_model_state = model.state_dict()
        print(f"<<< Nouveau meilleur modèle (Val Loss: {val_loss:.4f})")

end_time = time.time()
train_time = end_time - start_time
print(f"Temps d'entraînement total: {train_time:.2f} secondes")
print(f"Meilleur modèle à l'époque {best_epoch} avec Val Loss: {best_val_loss:.4f}")

# =========================
# Évaluation sur le test set
# =========================
if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor, None)
    test_loss = loss_fn(preds, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# =========================
# Interprétabilité : importance des features
# =========================
# Hook pour suivre les cartes d'attention
class AttentionHook:
    def __init__(self):
        self.attention_maps = []
    
    def __call__(self, module, input, output):
        # output est un tuple (sortie, attention_weights)
        self.attention_maps.append(output[1].detach().cpu())

hook = AttentionHook()
handles = []
for layer in model.layers:
    handles.append(layer['attention'].register_forward_hook(hook))

model.eval()
with torch.no_grad():
    # Passer tout le dataset d'entraînement par batch pour collecter les cartes d'attention
    for i in range(0, X_train_tensor.size(0), batch_size):
        batch_x = X_train_tensor[i:i+batch_size]
        model(batch_x, None)

if hook.attention_maps:
    # Concaténer toutes les cartes d'attention collectées
    attention_maps = torch.cat(hook.attention_maps, dim=0)
    
    # Calcul de la moyenne globale (toutes les observations, têtes et blocs)
    average_attention_map = attention_maps.mean(0)
    
    # Token CLS en première position
    average_cls_attention_map = average_attention_map[0]
    
    # Importance des features (tokens 1 à n_features+1)
    feature_importance = average_cls_attention_map[1:1+n_features]
    
    print("\nImportance des caractéristiques (via attention du token CLS):")
    for i, imp in enumerate(feature_importance):
        print(f"Feature {i}: {imp.item():.4f}")
else:
    print("Aucune carte d'attention collectée.")

# Nettoyer les hooks
for handle in handles:
    handle.remove()