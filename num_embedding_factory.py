"""
Factory pour créer différents types d'embeddings numériques basé sur rtdl_num_embeddings.
Sortie toujours: (batch_size, n_features, d_embedding).
"""

import torch
import torch.nn as nn
import numpy as np
from rtdl_num_embeddings import (
    LinearEmbeddings,
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
    compute_bins,
)


def get_num_embedding(
    embedding_type: str,
    X_train=None,
    d_embedding: int = None,
    y_train=None,
    n_bins: int = 5,
    d_periodic_embedding: int = None,
    sigma: float = 0.01,
    n_features: int = None,
    **kwargs
):
    """
    Retourne un module d'embedding numérique basé sur rtdl_num_embeddings.
    X_train est désormais optionnel ; si absent, `n_features` doit être fourni.
    Les embeddings basés sur quantiles/tree nécessitent néanmoins X_train.
    """
    # Conversion des données si fournies
    if X_train is not None:
        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        elif isinstance(X_train, torch.Tensor):
            X_train = X_train.float()

    if y_train is not None:
        if isinstance(y_train, np.ndarray):
            y_train = torch.tensor(y_train)
        elif isinstance(y_train, torch.Tensor):
            y_train = y_train

    # Déduire n_features si possible
    if X_train is not None and n_features is None:
        n_features = X_train.shape[1]
    if n_features is None:
        raise ValueError("n_features ou X_train doit être fourni pour construire les embeddings.")

    # === SIMPLE EMBEDDINGS ===
    if embedding_type == "L":
        return LinearEmbeddings(n_features, d_embedding)
    
    elif embedding_type == "LR":
        return LinearReLUEmbeddings(n_features, d_embedding)
    
    elif embedding_type == "LR-LR":
        return nn.Sequential(
            LinearReLUEmbeddings(n_features, d_embedding),
            # Projection finale pour maintenir (B, F, d_embedding)
            nn.Linear(d_embedding, d_embedding)
        )
    
    # === PIECEWISE-LINEAR EMBEDDINGS (Quantile) ===
    elif embedding_type.startswith("Q"):
        # Calcul des bins basés sur les quantiles
        bins = compute_bins(X_train, n_bins=n_bins)
        
        if embedding_type == "Q":
            return PiecewiseLinearEmbeddings(
                bins, 
                d_embedding, 
                activation=False, 
                version="B"
            )
        
        elif embedding_type == "Q-L":
            total_bins = sum(len(b) - 1 for b in bins)
            return nn.Sequential(
                PiecewiseLinearEncoding(bins),
                nn.Linear(total_bins, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding))
            )
        
        elif embedding_type == "Q-LR":
            total_bins = sum(len(b) - 1 for b in bins)
            return nn.Sequential(
                PiecewiseLinearEncoding(bins),
                nn.Linear(total_bins, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding)),
                nn.ReLU()
            )
        
        elif embedding_type == "Q-LR-LR":
            total_bins = sum(len(b) - 1 for b in bins)
            return nn.Sequential(
                PiecewiseLinearEncoding(bins),
                nn.Linear(total_bins, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding)),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(n_features * d_embedding, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding)),
                nn.ReLU()
            )
    
    # === PIECEWISE-LINEAR EMBEDDINGS (Tree-based) ===
    elif embedding_type.startswith("T"):
        if y_train is None:
            raise ValueError("y_train requis pour les embeddings tree-based")
        
        # Calcul des bins basés sur les arbres de décision
        bins = compute_bins(
            X_train,
            n_bins=n_bins,
            tree_kwargs={
                'min_samples_leaf': max(64, len(X_train) // 50),
                'min_impurity_decrease': 1e-4,
                'max_depth': 5
            },
            y=y_train,
            regression=False  # Classification par défaut
        )
        
        if embedding_type == "T":
            return PiecewiseLinearEmbeddings(
                bins, 
                d_embedding, 
                activation=False, 
                version="B"
            )
        
        elif embedding_type == "T-L":
            total_bins = sum(len(b) - 1 for b in bins)
            return nn.Sequential(
                PiecewiseLinearEncoding(bins),
                nn.Linear(total_bins, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding))
            )
        
        elif embedding_type == "T-LR":
            total_bins = sum(len(b) - 1 for b in bins)
            return nn.Sequential(
                PiecewiseLinearEncoding(bins),
                nn.Linear(total_bins, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding)),
                nn.ReLU()
            )
        
        elif embedding_type == "T-LR-LR":
            total_bins = sum(len(b) - 1 for b in bins)
            return nn.Sequential(
                PiecewiseLinearEncoding(bins),
                nn.Linear(total_bins, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding)),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(n_features * d_embedding, n_features * d_embedding),
                nn.Unflatten(1, (n_features, d_embedding)),
                nn.ReLU()
            )
    
    # === PERIODIC EMBEDDINGS ===
    elif embedding_type.startswith("P"):
        if d_periodic_embedding is None:
            d_periodic_embedding = d_embedding
        
        if embedding_type == "P":
            return PeriodicEmbeddings(
                n_features, 
                d_periodic_embedding, 
                lite=True,
                frequency_init_scale=sigma
            )
        
        elif embedding_type == "P-L":
            pe = PeriodicEmbeddings(
                n_features, 
                d_periodic_embedding, 
                lite=True,
                frequency_init_scale=sigma
            )
            # Projection vers d_embedding si différent
            if d_periodic_embedding != d_embedding:
                return nn.Sequential(
                    pe,
                    nn.Linear(d_periodic_embedding, d_embedding)
                )
            else:
                return pe
        
        elif embedding_type == "P-LR":
            pe = PeriodicEmbeddings(
                n_features, 
                d_periodic_embedding, 
                lite=True,
                frequency_init_scale=sigma
            )
            return nn.Sequential(
                pe,
                nn.Linear(d_periodic_embedding, d_embedding),
                nn.ReLU()
            )
        
        elif embedding_type == "P-LR-LR":
            pe = PeriodicEmbeddings(
                n_features, 
                d_periodic_embedding, 
                lite=True,
                frequency_init_scale=sigma
            )
            return nn.Sequential(
                pe,
                nn.Linear(d_periodic_embedding, d_embedding),
                nn.ReLU(),
                nn.Linear(d_embedding, d_embedding),
                nn.ReLU()
            )
    
    else:
        raise ValueError(f"Type d'embedding inconnu: {embedding_type}")


# === CLASSE UTILITAIRE POUR USAGE FACILE ===
class NumericalEmbedder(nn.Module):
    """
    Wrapper pour utilisation facile des embeddings numériques.
    
    Usage:
        embedder = NumericalEmbedder("P-LR", X_train, d_embedding=32)
        x = torch.randn(batch_size, n_features)
        embedded = embedder(x)  # Shape: (batch_size, n_features, 32)
    """
    
    def __init__(
        self,
        embedding_type: str,
        X_train=None,
        d_embedding: int = None,
        n_features: int = None,
        **kwargs
    ):
        super().__init__()
        self.embedding_type = embedding_type

        # Déduire n_features si possible
        if X_train is not None and hasattr(X_train, "shape"):
            self.n_features = X_train.shape[1]
        elif n_features is not None:
            self.n_features = n_features
        else:
            raise ValueError("X_train ou n_features doit être fourni pour NumericalEmbedder.")

        self.d_embedding = d_embedding

        # Créer l'embedding (transmettre n_features pour compatibilité)
        self.embedder = get_num_embedding(
            embedding_type=embedding_type,
            X_train=X_train,
            d_embedding=d_embedding,
            n_features=self.n_features,
            **kwargs
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, n_features)
        
        Returns:
            Tensor de shape (batch_size, n_features, d_embedding)
        """
        return self.embedder(x)
    
    def get_output_shape(self):
        """Retourne la shape de sortie sans la dimension batch."""
        return (self.n_features, self.d_embedding)
    
    def get_flattened_size(self):
        """Retourne la taille après aplatissement."""
        return self.n_features * self.d_embedding
