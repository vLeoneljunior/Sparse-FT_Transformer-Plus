import torch
import torch.nn as nn
import time
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sparse_ftt_plus import InterpretableFTTPlus
from data_utils import prepare_california_data, get_data_splits

# Préparation des données California Housing
X, y = prepare_california_data()
data = get_data_splits(X, y)
X_train_tensor, X_val_tensor, X_test_tensor = data['X_train'], data['X_val'], data['X_test']
y_train_tensor, y_val_tensor, y_test_tensor = data['y_train'], data['y_val'], data['y_test']
n_features = data['n_features']
feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# Configuration du modèle
model_config = {
    "n_num_features": n_features,
    "cat_cardinalities": None,  # Pas de variables catégoriques pour California Housing
    "d_token": 192,
    "n_blocks": 3,
    "n_heads": 8,
    "d_ffn_factor": 1.333,
    "attention_dropout": 0.2,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "d_out": 1,
    "attention_initialization": "kaiming",
    "attention_normalization": "LayerNorm",
    "ffn_activation": "ReGLU",
    "ffn_normalization": "LayerNorm",
    "prenormalization": True,
    "num_tokenizer": True,
    "num_tokenizer_type": "LR",
}

# Validation de num_tokenizer_type
if model_config["num_tokenizer"] and model_config["num_tokenizer_type"] is None:
    raise ValueError("num_tokenizer_type must be specified when num_tokenizer=True")

# Calcul de ffn_d_hidden
ffn_d_hidden = int(model_config["d_token"] * model_config["d_ffn_factor"])

# Création du modèle
model = InterpretableFTTPlus.make_baseline(
    n_num_features=model_config["n_num_features"],
    cat_cardinalities=model_config["cat_cardinalities"],
    d_token=model_config["d_token"],
    n_blocks=model_config["n_blocks"],
    n_heads=model_config["n_heads"],
    attention_dropout=model_config["attention_dropout"],
    ffn_d_hidden=ffn_d_hidden,
    ffn_dropout=model_config["ffn_dropout"],
    residual_dropout=model_config["residual_dropout"],
    d_out=model_config["d_out"],
    attention_initialization=model_config["attention_initialization"],
    attention_normalization=model_config["attention_normalization"],
    ffn_activation=model_config["ffn_activation"],
    ffn_normalization=model_config["ffn_normalization"],
    prenormalization=model_config["prenormalization"],
    num_tokenizer=model_config["num_tokenizer"],
    num_tokenizer_type=model_config["num_tokenizer_type"],
)
n_parameters = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres du modèle: {n_parameters}")

# Entraînement
optimizer = torch.optim.AdamW(model.optimization_param_groups(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()
n_epochs = 50
batch_size = 64

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
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        optimizer.zero_grad()
        output = model(batch_x, x_cat=None).squeeze(-1)
        loss = loss_fn(output, batch_y.squeeze(-1))
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
            val_preds.append(model(batch_x, x_cat=None).squeeze(-1))
    val_preds = torch.cat(val_preds)
    val_loss = loss_fn(val_preds, y_val_tensor.squeeze(-1))
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_model_state = model.state_dict()
        torch.save(best_model_state, "best_model.pth")
        print(f"<<< Nouveau meilleur modèle (Val Loss: {val_loss:.4f})")

print(f"Temps d'entraînement: {(time.time() - start_time):.2f} secondes")
print(f"Meilleur modèle à l'époque {best_epoch} avec Val Loss: {best_val_loss:.4f}")

# Évaluation sur le test set
if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor, x_cat=None).squeeze(-1)
    y_true = y_test_tensor.squeeze(-1)
    test_loss = loss_fn(preds, y_true).item()
    test_r2 = r2_score(y_true.cpu().numpy(), preds.cpu().numpy())
    test_mae = mean_absolute_error(y_true.cpu().numpy(), preds.cpu().numpy())
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

# Importance des features
result = model.get_cls_importance(X_test_tensor, x_cat=None, feature_names=feature_names, batch_size=batch_size)
print("\nImportance des features (via attention du token CLS):")
for name, imp in result.items():
    if not name.startswith('_'):
        print(f"{name}: {imp:.4f}")
print("Fichiers sauvegardés:", result.get('_saved_paths', {}))