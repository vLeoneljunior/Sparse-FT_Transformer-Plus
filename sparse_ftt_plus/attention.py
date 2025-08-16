import torch
import torch.nn as nn
from torch import Tensor
import math
from sparsemax import Sparsemax
from typing import Tuple, Dict

"""
Module d'attention interprétable pour le Feature Tokenizer Transformer (FTT).

Principe et justification (résumé) :
- Problème : avec softmax et V propres à chaque tête, les poids d'attention sont toujours denses
  (chaque position reçoit un poids non nul) et les contributions effectives sont difficiles à interpréter
  car V_h varie selon la tête.
- Solution apportée ici :
  1) Remplacement de softmax par sparsemax → des probabilités creuses (beaucoup de zéros), ce qui
     concentre l'attention sur un sous-ensemble limité de features et améliore l'explicabilité.
  2) Partage de la projection V entre toutes les têtes (W_V commun) → la variabilité entre têtes
     ne vient plus que des matrices d'attention A_h. Ainsi, un poids d'attention A_h(i,j) comparable
     entre têtes signifie une contribution comparable de la feature j, car V est identique.
- Agrégation : les cartes d'attention par tête sont moyennées pour obtenir une matrice unique
  tilde A ∈ R^{seq_len × seq_len}. La ligne correspondant au token [CLS] contient
  directement les importances normalisées des features vis-à-vis de la prédiction.
- Extraction d'importance : lire la première ligne de la matrice moyennée renvoie les scores
  {tilde A(0, j)}_j déjà normalisés (somme = 1 par ligne) → utilisable comme score d'importance
  intrinsèque au modèle (échantillon / batch / dataset).

Ce module implémente ces principes en gardant une API simple compatible avec l'utilisation
dans un bloc Transformer standard.
"""

sparsemax = Sparsemax(dim=-1)

class MultiheadAttention(nn.Module):
    """Attention multi-tête de base pour projections Q, K, V.

    Cette classe fournit les projections linéaires W_q, W_k, W_v et un éventuel W_out.
    Elle expose également _split_to_heads qui convertit une représentation (B, T, D)
    en (B * H, T, D_head) pour le calcul tête-par-tête.
    """
    def __init__(self, d_token: int, n_heads: int, dropout: float = 0.0, initialization: str = "kaiming"):
        super().__init__()
        assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']

        self.n_heads = n_heads
        self.d_token = d_token
        self.W_q = nn.Linear(d_token, d_token, bias=True)
        self.W_k = nn.Linear(d_token, d_token, bias=True)
        self.W_v = nn.Linear(d_token, d_token, bias=True)
        self.W_out = nn.Linear(d_token, d_token, bias=True) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier':
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None and self.W_out.bias is not None:
            nn.init.zeros_(self.W_out.bias)

    def _split_to_heads(self, x: Tensor) -> Tensor:
        """Scinde le tenseur en têtes pour calculs indépendants par tête.

        Entrée : (batch_size, seq_len, d_token)
        Sortie : (batch_size * n_heads, seq_len, d_head)
        """
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads, n_tokens, d_head)

class InterpretableMultiHeadAttention(nn.Module):
    """Attention multi-tête interprétable avec sparsemax et V partagé.

    Comportement :
    - Calcule Q, K, V à partir de la même entrée X.
    - Applique sparsemax sur les logits QK^T / sqrt(d_head) pour obtenir des probabilités creuses
      tête-par-tête.
    - Construit V partagé en moyennant les composantes V sur les têtes (évite les distorsions dues à V_h).
    - Réplique V partagé pour chaque tête et calcule la sortie attentionnée.
    - Retourne également la matrice d'attention moyenne sur les têtes pour permettre l'interprétabilité.
    
    Formats :
    - x : (batch_size, seq_len, d_model)
    - Retour : (x_out, {"attention_probs": avg_attention}) où avg_attention a la forme (batch_size, seq_len, seq_len)
      et peut être utilisée pour extraire l'importance du token [CLS] via avg_attention[:, 0, 1:].
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, initialization: str = "kaiming"):
        super().__init__()
        self.n_heads = n_heads
        self.base_mha = MultiheadAttention(d_token=d_model, n_heads=n_heads, dropout=dropout, initialization=initialization)
        self.dropout = self.base_mha.dropout

    def _average_attention_probs(self, attention_probs: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Moyenne les probabilités d'attention sur les têtes.

        attention_probs : (batch_size * n_heads, seq_len, seq_len)
        Retour : (batch_size, seq_len, seq_len) moyenné sur l'axe tête.
        """
        ap = attention_probs.view(batch_size, self.n_heads, seq_len, seq_len)
        return ap.mean(dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass avec sparsemax et V partagé.

        Args:
            x: Tensor d'entrée de forme (batch_size, seq_len, d_model).

        Returns:
            Tuple[Tensor, Dict] :
              - output : (batch_size, seq_len, d_model) après concat/réprojection des têtes.
              - meta dict contenant "attention_probs" : matrice moyennée (batch_size, seq_len, seq_len).
        
        Notes d'interprétabilité :
        - Pour obtenir l'importance des features pour la prédiction (token [CLS]), lire la ligne 0 de
          la matrice renvoyée : avg_attention[:, 0, :] (ou avg_attention[0, 1:] si on omet le token CLS lui-même).
        - Les valeurs sont normalisées par ligne (somme = 1) et, grâce à sparsemax, sont souvent creuses,
          facilitant l'identification des features réellement influentes.
        """
        batch_size, seq_len, d_model = x.shape

        # Projections Q, K, V
        q = self.base_mha.W_q(x)  # (batch_size, seq_len, d_model)
        k = self.base_mha.W_k(x)
        v = self.base_mha.W_v(x)

        # Reshape pour têtes
        qh = self.base_mha._split_to_heads(q)  # (batch_size * n_heads, seq_len, d_head)
        kh = self.base_mha._split_to_heads(k)
        vh_full = self.base_mha._split_to_heads(v)

        d_head = vh_full.shape[-1]

        # Calcul des logits d'attention
        attention_logits = qh @ kh.transpose(1, 2) / math.sqrt(d_head)  # (batch_size * n_heads, seq_len, seq_len)

        # Sparsemax pour probabilités creuses (tête-par-tête)
        attention_probs = sparsemax(attention_logits)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        # V partagé : on moyenner les composantes V sur l'axe tête pour obtenir W_V commun implicite
        v_resh = v.reshape(batch_size, seq_len, self.n_heads, d_head)
        v_shared = v_resh.mean(dim=2)  # (batch_size, seq_len, d_head)
        # Répliquer V partagé pour chaque tête afin d'aligner les dimensions pour le produit attention*V
        v_shared_rep = v_shared.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        vh = v_shared_rep.transpose(1, 2).reshape(batch_size * self.n_heads, seq_len, d_head)

        # Sortie d'attention
        x_out = attention_probs @ vh  # (batch_size * n_heads, seq_len, d_head)
        x_out = (
            x_out.reshape(batch_size, self.n_heads, seq_len, d_head)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.n_heads * d_head)
        )
        if self.base_mha.W_out is not None:
            x_out = self.base_mha.W_out(x_out)

        # Moyenne des probabilités d'attention sur les têtes -> utilité interprétabilité
        avg_attention = self._average_attention_probs(attention_probs, batch_size, seq_len)
        return x_out, {"attention_probs": avg_attention}

    def get_attention_weights(self, x: Tensor) -> Tensor:
        """Retourne les probabilités d'attention moyennes sans gradients.

        Utile pour inspection / visualisation (heatmap, barplot des importances).
        """
        with torch.no_grad():
            _, meta = self.forward(x)
        return meta["attention_probs"]