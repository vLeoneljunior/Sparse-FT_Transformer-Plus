"""Interpretable FT-Transformer (sparse + shared-V)

Ce module implémente une variante interprétable du FT-Transformer adaptée aux données
tabulaires. Principes clés (résumé) :
- Chaque variable (catégorielle ou numérique) est tokenisée via FeatureTokenizer en un embedding
  de dimension d_token. Les embeddings sont assemblés en séquence et préfixés par un token [CLS]
  qui agrège l'information pour la prédiction finale.
- L'attention multi-tête utilisée remplace softmax par sparsemax, produisant des distributions
  creuses et concentrant l'attention sur un petit sous-ensemble de features.
- Les têtes partagent la même projection V (W_V commune) afin d'éliminer la distorsion due à
  différentes transformations V_h ; ainsi, la seule source de variabilité entre têtes est la
  matrice d'attention elle-même.
- Les cartes d'attention sont moyennées sur les têtes pour obtenir une matrice unique
  avg_attention (seq_len × seq_len). La ligne correspondant au token [CLS] fournit des scores
  d'importance intrinsèques pour les features (normalisés par ligne, somme = 1). Grâce à
  sparsemax ces scores sont souvent creux et directement exploitables.

Fonctionnalités exposées :
- InterpretableTransformerBlock : bloc Transformer utilisant l'attention interprétable.
- InterpretableFTTPlus : modèle complet (tokenizer + blocs + head) et utilitaire
  get_cls_importance() qui collecte, moyenne et sauvegarde les importances par feature.

"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import scipy.stats
import os
import csv
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from .attention import InterpretableMultiHeadAttention
from rtdl_lib.modules import FeatureTokenizer, CLSToken, _make_nn_module

class AttentionHook:
    """Collecte les cartes d'attention pour l'interprétabilité."""
    def __init__(self) -> None:
        self.attention_maps: List[Tensor] = []

    def __call__(self, module, input, output):
        att = output[1]['attention_probs']
        self.attention_maps.append(att.detach().cpu())

    def clear(self) -> None:
        self.attention_maps.clear()

class InterpretableTransformerBlock(nn.Module):
    """Bloc Transformer avec attention interprétable (sparsemax + V partagé)."""
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
    ) -> None:
        super().__init__()
        self.prenormalization = prenormalization
        self.attention = InterpretableMultiHeadAttention(
            d_model=d_token, n_heads=n_heads, dropout=attention_dropout, initialization=attention_initialization
        )
        self.attention_normalization = _make_nn_module(attention_normalization, d_token)
        self.ffn_normalization = _make_nn_module(ffn_normalization, d_token)
        from rtdl_lib.modules import Transformer
        self.ffn = Transformer.FFN(
            d_token=d_token, d_hidden=ffn_d_hidden, bias_first=True, bias_second=True,
            dropout=ffn_dropout, activation=ffn_activation
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else None
        self.ffn_residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else None

    def apply_normalization(self, x: Tensor, stage: str) -> Tensor:
        if self.prenormalization:
            return self.attention_normalization(x) if stage == "attention" else self.ffn_normalization(x)
        return x

    def add_residual(self, x: Tensor, residual: Tensor, stage: str) -> Tensor:
        if stage == "attention" and self.attention_residual_dropout:
            residual = self.attention_residual_dropout(residual)
        elif stage == "ffn" and self.ffn_residual_dropout:
            residual = self.ffn_residual_dropout(residual)
        x = x + residual
        if not self.prenormalization:
            x = self.attention_normalization(x) if stage == "attention" else self.ffn_normalization(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.apply_normalization(x, "attention")
        att_out, _ = self.attention(x_residual)
        x = self.add_residual(x, att_out, "attention")
        x_residual = self.apply_normalization(x, "ffn")
        x = self.add_residual(x, self.ffn(x_residual), "ffn")
        return x

class InterpretableFTTPlus(nn.Module):
    """FT-Transformer interprétable avec attention sparse et V partagé."""
    def __init__(
        self,
        n_num_features: int,
        d_token: int,
        n_blocks: int,
        n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        head_activation: str,
        head_normalization: str,
        d_out: int,
        num_tokenizer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(n_num_features=n_num_features, cat_cardinalities=None, d_token=d_token)
        if num_tokenizer:
            self.feature_tokenizer.num_tokenizer = num_tokenizer
        self.cls_token = CLSToken(d_token, self.feature_tokenizer.initialization)
        self.blocks = nn.ModuleList([
            InterpretableTransformerBlock(
                d_token=d_token, n_heads=n_heads, attention_dropout=attention_dropout,
                attention_initialization=attention_initialization, attention_normalization=attention_normalization,
                ffn_d_hidden=ffn_d_hidden, ffn_dropout=ffn_dropout, ffn_activation=ffn_activation,
                ffn_normalization=ffn_normalization, residual_dropout=residual_dropout, prenormalization=prenormalization
            ) for _ in range(n_blocks)
        ])
        from rtdl_lib.modules import Transformer
        self.head = Transformer.Head(
            d_in=d_token, d_out=d_out, bias=True, activation=head_activation,
            normalization=head_normalization if prenormalization else "Identity"
        )
        self.prenormalization = prenormalization

    @classmethod
    def get_baseline_config(cls) -> Dict[str, Any]:
        return {
            "n_heads": 8,
            "attention_initialization": "kaiming",
            "attention_normalization": "LayerNorm",
            "ffn_activation": "ReGLU",
            "ffn_normalization": "LayerNorm",
            "prenormalization": True,
            "head_activation": "ReLU",
            "head_normalization": "LayerNorm",
        }

    @classmethod
    def make_baseline(
        cls,
        n_num_features: int,
        d_token: int,
        n_blocks: int,
        n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
        d_out: int,
        attention_initialization: str = "kaiming",
        attention_normalization: str = "LayerNorm",
        ffn_activation: str = "ReGLU",
        ffn_normalization: str = "LayerNorm",
        prenormalization: bool = True,
        head_activation: str = "ReLU",
        head_normalization: str = "LayerNorm",
        num_tokenizer: Optional[nn.Module] = None,
        num_tokenizer_type: Optional[str] = "LR",
    ) -> "InterpretableFTTPlus":
        config = cls.get_baseline_config()
        config.update({
            "n_num_features": n_num_features, "d_token": d_token, "n_blocks": n_blocks, "n_heads": n_heads,
            "attention_dropout": attention_dropout, "ffn_d_hidden": ffn_d_hidden, "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout, "d_out": d_out, "attention_initialization": attention_initialization,
            "attention_normalization": attention_normalization, "ffn_activation": ffn_activation,
            "ffn_normalization": ffn_normalization, "prenormalization": prenormalization,
            "head_activation": head_activation, "head_normalization": head_normalization, "num_tokenizer": num_tokenizer
        })
        model = cls(**config)
        if num_tokenizer is None and num_tokenizer_type:
            from num_embedding_factory import get_num_embedding
            
            model.feature_tokenizer.num_tokenizer = get_num_embedding(
                embedding_type=num_tokenizer_type,
                n_features=n_num_features,  # Passer n_num_features
                d_embedding=d_token,
            )
            
        return model

    def forward(self, x_num: Tensor) -> Tensor:
        x = self.feature_tokenizer(x_num, None)
        x = self.cls_token(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def get_cls_importance(self, x_num: Tensor, feature_names: Optional[List[str]] = None, batch_size: int = 64) -> Dict[str, Any]:
        """Extrait et sauvegarde l'importance des features à partir des cartes d'attention.

        Détails et conventions :
        - Cette méthode attache un hook forward à chaque module d'attention pour collecter la
          sortie meta renvoyée par InterpretableMultiHeadAttention, attendu sous la forme
          (output_tensor, {"attention_probs": avg_attention}) où avg_attention a la forme
          (batch_size, seq_len, seq_len) et correspond à la moyenne des probabilités d'attention
          sur les têtes (grâce à sparsemax et au partage de V).
        - Convention d'indexation dans cette implémentation : le token [CLS] est ajouté par
          CLSToken à la FIN de la séquence. Par conséquent, la ligne correspondant au token CLS
          est prise ici comme le dernier index (-1).
        - Extraction : on lit la ligne CLS et on exclut la colonne correspondant au token CLS lui-même
          pour obtenir les importances des J features -> average_attention_map[CLS_index, :-1].
        - Remarque d'interprétabilité : les scores proviennent directement de sparsemax et sont
          normalisés par ligne (somme = 1).

        Retour :
        Dictionnaire contenant :
         - key: nom de la feature (ou feature_i) -> importance (float)
         - _sorted_indices : indices triés par importance décroissante
         - _ranks_array : rangs (1 = plus important)
         - _saved_paths : chemins absolus des fichiers sauvegardés (.npy et .csv)
        """
        hook = AttentionHook()
        handles = [block.attention.register_forward_hook(hook) for block in self.blocks]
        try:
            self.eval()
            with torch.inference_mode():
                for i in range(0, x_num.size(0), batch_size):
                    batch_x_num = x_num[i : i + batch_size]
                    _ = self(batch_x_num)

                if not hook.attention_maps:
                    print("Aucune carte d'attention collectée.")
                    return {}

                # attention_maps : (n_collections, batch, seq_len, seq_len)
                attention_maps = torch.cat(hook.attention_maps, dim=0)
                # moyenne sur collections et batchs -> (seq_len, seq_len)
                average_attention_map = attention_maps.mean(dim=0)

                # Ici, le CLS est le dernier token (convention CLSToken ajoutant à la fin).
                # Si votre CLSToken préfixe au début, remplacer -1 par 0.
                cls_index = -1
                feature_importance = average_attention_map[cls_index, :-1].cpu().numpy()

                feature_ranks = scipy.stats.rankdata(-feature_importance)
                feature_indices_sorted = np.argsort(-feature_importance)

                os.makedirs("results", exist_ok=True)
                np.save("results/feature_importance.npy", feature_importance)
                np.save("results/feature_ranks.npy", feature_ranks)

                with open("results/feature_importance_and_ranks.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["feature_index", "importance", "rank"])
                    for idx, imp, rank in zip(range(len(feature_importance)), feature_importance, feature_ranks):
                        writer.writerow([idx, imp, rank])

                result = {feature_names[i] if feature_names else f"feature_{i}": float(imp) for i, imp in enumerate(feature_importance)}
                result["_sorted_indices"] = feature_indices_sorted.tolist()
                result["_ranks_array"] = feature_ranks.tolist()
                result["_saved_paths"] = {
                    "npy_importance": os.path.abspath("results/feature_importance.npy"),
                    "npy_ranks": os.path.abspath("results/feature_ranks.npy"),
                    "csv": os.path.abspath("results/feature_importance_and_ranks.csv")
                }
                return result
        finally:
            for handle in handles:
                handle.remove()
            hook.clear()

    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        NO_WD_NAMES = ["feature_tokenizer", "normalization", ".bias"]
        return [
            {"params": [p for n, p in self.named_parameters() if all(s not in n for s in NO_WD_NAMES)]},
            {"params": [p for n, p in self.named_parameters() if any(s in n for s in NO_WD_NAMES)], "weight_decay": 0.0}
        ]

    def make_default_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(self.optimization_param_groups(), lr=1e-4, weight_decay=1e-5)