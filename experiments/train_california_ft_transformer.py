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
# Ajouter le répertoire du script (experiments) au PYTHONPATH pour permettre 'import data_utils'
sys.path.append(current_dir)

from data_utils import prepare_california_data, get_data_splits
import rtdl_lib as rtdl

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
# Configuration du modèle FT-Transformer
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

# Construire le modèle via rtdl_lib en respectant les paramètres principaux
# On mappe n_layers -> n_blocks et calcule ffn_d_hidden depuis d_token * d_ffn_factor
ffn_d_hidden = int(model_config["d_token"] * model_config["d_ffn_factor"])
model = rtdl.FTTransformer.make_baseline(
    n_num_features=n_features,
    cat_cardinalities=None,
    d_token=model_config["d_token"],
    n_blocks=model_config["n_layers"],
    attention_dropout=model_config["attention_dropout"],
    ffn_d_hidden=ffn_d_hidden,
    ffn_dropout=model_config["ffn_dropout"],
    residual_dropout=model_config["residual_dropout"],
    d_out=model_config["d_out"],
)

# =========================
# Boucle d'entraînement avec validation
# =========================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()
n_epochs = 50
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
        # Squeeze targets when d_out == 1 so shapes match model output (batch,)
        batch_y_squeezed = (
            batch_y.squeeze(-1)
            if batch_y.dim() > 1 and batch_y.shape[-1] == 1
            else batch_y
        )
        loss = loss_fn(output.squeeze(-1), batch_y_squeezed)
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
    val_targets = (
        y_val_tensor.squeeze(-1)
        if y_val_tensor.dim() > 1 and y_val_tensor.shape[-1] == 1
        else y_val_tensor
    )
    val_loss = loss_fn(val_preds.squeeze(-1), val_targets)
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
    test_targets = (
        y_test_tensor.squeeze(-1)
        if y_test_tensor.dim() > 1 and y_test_tensor.shape[-1] == 1
        else y_test_tensor
    )
    test_loss = loss_fn(preds.squeeze(-1), test_targets)
    print(f"Test Loss: {test_loss.item():.4f}")

# =========================
# Interprétabilité : importance des features
# =========================
# Hook pour suivre les cartes d'attention (compatible avec RTDL et FTT+)
class AttentionHook:
    def __init__(self):
        self.attention_maps = []

    def __call__(self, module, input, output):
        try:
            att = output[1]['attention_probs']
            self.attention_maps.append(att.detach().cpu())
        except Exception:
            return

hook = AttentionHook()
handles = []
# Attacher les hooks sur les blocs du transformer (rtdl_lib expose transformer.blocks)
for layer in model.transformer.blocks:
    handles.append(layer['attention'].register_forward_hook(hook))

model.eval()
with torch.inference_mode():
    # Passer le dataset de test par batch pour collecter les cartes d'attention
    for i in range(0, X_test_tensor.size(0), batch_size):
        batch_x = X_test_tensor[i:i+batch_size]
        model(batch_x, None)

# Concaténer toutes les cartes d'attention collectées
if hook.attention_maps:
    attention_maps = torch.cat(hook.attention_maps, dim=0)

    # Calcul de la moyenne globale (toutes les observations, têtes et blocs)
    average_attention_map = attention_maps.mean(0)

    # Position du token CLS : FTTransformer ajoute le token CLS à la fin => index -1
    average_cls_attention_map = average_attention_map[-1]

    # Importance des features : tous les tokens sauf le CLS final
    feature_importance = average_cls_attention_map[:-1].cpu().numpy()

    import scipy.stats
    feature_ranks = scipy.stats.rankdata(-feature_importance)
    feature_indices_sorted_by_importance = np.argsort(-feature_importance)

    print("Rangs des caractéristiques (1 = plus important):", feature_ranks)
    print("Indices triés par importance:", feature_indices_sorted_by_importance)

    # Sauvegarde des résultats
    import csv
    os.makedirs('results', exist_ok=True)
    np.save(os.path.join('results', 'feature_importance.npy'), feature_importance)
    np.save(os.path.join('results', 'feature_ranks.npy'), feature_ranks)
    csv_path = os.path.join('results', 'feature_importance_and_ranks.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_index', 'importance', 'rank'])
        for idx, imp, rank in zip(range(len(feature_importance)), feature_importance, feature_ranks):
            writer.writerow([int(idx), float(imp), int(rank)])
else:
    print("Aucune carte d'attention collectée.")

# Nettoyer les hooks
for handle in handles:
    handle.remove()