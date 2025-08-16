"""
Interpretable FT-Transformer (sparse + shared-V)

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
from typing import Any, Dict, List, Optional, Tuple, Union
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
        cat_cardinalities: Optional[List[int]] = None,
        num_tokenizer: bool = False,
        num_tokenizer_type: Optional[str] = "LR",
    ) -> None:
        super().__init__()
        self.cat_cardinalities = cat_cardinalities  # Stocker cat_cardinalities
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
        )
        if num_tokenizer:
            if num_tokenizer_type is None:
                raise ValueError("num_tokenizer_type must be specified when num_tokenizer=True")
            from num_embedding_factory import get_num_embedding
            self.feature_tokenizer.num_tokenizer = get_num_embedding(
                embedding_type=num_tokenizer_type,
                n_features=n_num_features,
                d_embedding=d_token,
            )
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
        cat_cardinalities: Optional[List[int]] = None,
        attention_initialization: str = "kaiming",
        attention_normalization: str = "LayerNorm",
        ffn_activation: str = "ReGLU",
        ffn_normalization: str = "LayerNorm",
        prenormalization: bool = True,
        head_activation: str = "ReLU",
        head_normalization: str = "LayerNorm",
        num_tokenizer: bool = False,  # False -> Use default NumericalFeatureTokenizer from rtdl_lib
                                      # True -> Use custom numeric embedding via get_num_embedding
        num_tokenizer_type: Optional[str] = "LR",
    ) -> "InterpretableFTTPlus":
        config = cls.get_baseline_config()
        config.update({
            "n_num_features": n_num_features,
            "d_token": d_token,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "attention_dropout": attention_dropout,
            "ffn_d_hidden": ffn_d_hidden,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
            "d_out": d_out,
            "cat_cardinalities": cat_cardinalities,
            "attention_initialization": attention_initialization,
            "attention_normalization": attention_normalization,
            "ffn_activation": ffn_activation,
            "ffn_normalization": ffn_normalization,
            "prenormalization": prenormalization,
            "head_activation": head_activation,
            "head_normalization": head_normalization,
            "num_tokenizer": num_tokenizer,
            "num_tokenizer_type": num_tokenizer_type,
        })
        return cls(**config)

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor] = None) -> Tensor:
        """Forward compatible with both numerical and categorical inputs.

        Accepts:
            x_num: numerical features tensor or None if absent
            x_cat: categorical features tensor or None if absent

        The FeatureTokenizer will internally use any tokenizer present
        (num_tokenizer and/or cat_tokenizer).
        """
        assert (x_cat is None) == (self.cat_cardinalities is None), \
            "x_cat must be None if cat_cardinalities is None, and vice versa"
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.cls_token(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def get_cls_importance(
        self,
        x_num: Optional[Tensor],
        x_cat: Optional[Tensor] = None,
        feature_names: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Extrait et sauvegarde l'importance des features à partir des cartes d'attention.

        Arguments:
            x_num: Tensor des features numériques (peut être None si pas de numériques)
            x_cat: Tensor des features catégorielles (peut être None si pas de catégorielles)
            feature_names: liste optionnelle de noms de features
            batch_size: taille de batch pour l'extraction

        Note:
            Le token CLS est ajouté à la FIN de la séquence (convention CLSToken).
        """
        assert (x_cat is None) == (self.cat_cardinalities is None), \
            "x_cat must be None if cat_cardinalities is None, and vice versa"
        hook = AttentionHook()
        handles = [block.attention.register_forward_hook(hook) for block in self.blocks]
        try:
            self.eval()
            with torch.inference_mode():
                n_samples = x_num.size(0) if x_num is not None else x_cat.size(0)
                for i in range(0, n_samples, batch_size):
                    batch_x_num = None if x_num is None else x_num[i : i + batch_size]
                    batch_x_cat = None if x_cat is None else x_cat[i : i + batch_size]
                    _ = self(batch_x_num, batch_x_cat)

                if not hook.attention_maps:
                    print("Aucune carte d'attention collectée.")
                    return {}

                # attention_maps : (n_collections, batch, seq_len, seq_len)
                attention_maps = torch.cat(hook.attention_maps, dim=0)
                # moyenne sur collections et batchs -> (seq_len, seq_len)
                average_attention_map = attention_maps.mean(dim=0)

                # Ici, le CLS est le dernier token (convention CLSToken ajoutant à la fin).
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