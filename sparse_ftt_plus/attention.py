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

Améliorations computationnelles intégrées :
- V partagé calculé directement via une projection unique au lieu de moyenner post-projection
- Scaling pré-intégré dans les poids Q pour éviter la division répétitive
- Opérations de reshape et transpose réduites au minimum
- Memory layout optimisé pour réduire les copies de tenseurs
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
        self.d_head = d_token // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)  # Pré-calcul du facteur de scaling
        
        self.W_q = nn.Linear(d_token, d_token, bias=True)
        self.W_k = nn.Linear(d_token, d_token, bias=True)
        self.W_v = nn.Linear(d_token, d_token, bias=True)
        self.W_out = nn.Linear(d_token, d_token, bias=True) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Initialisation avec scaling intégré pour Q
        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier':
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            elif initialization == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        # Intégration du scaling dans les poids de Q
        with torch.no_grad():
            self.W_q.weight.mul_(self.scale)
            if self.W_q.bias is not None:
                self.W_q.bias.mul_(self.scale)
                
        if self.W_out is not None and self.W_out.bias is not None:
            nn.init.zeros_(self.W_out.bias)

    def _split_to_heads(self, x: Tensor) -> Tensor:
        """Scinde le tenseur en têtes pour calculs indépendants par tête.

        Entrée : (batch_size, seq_len, d_token)
        Sortie : (batch_size * n_heads, seq_len, d_head)
        """
        batch_size, n_tokens, d = x.shape
        return x.view(batch_size * self.n_heads, n_tokens, self.d_head)

class InterpretableMultiHeadAttention(nn.Module):
    """Attention multi-tête interprétable avec sparsemax et V partagé.

    Comportement :
    - Calcule Q, K, V à partir de la même entrée X.
    - Applique sparsemax sur les logits QK^T / sqrt(d_head) pour obtenir des probabilités creuses
      tête-par-tête.
    - Utilise un V partagé calculé directement via une projection unique vers d_head.
    - Applique l'attention et retourne la matrice d'attention moyenne pour l'interprétabilité.
    
    Formats :
    - x : (batch_size, seq_len, d_model)
    - Retour : (x_out, {"attention_probs": avg_attention}) où avg_attention a la forme (batch_size, seq_len, seq_len)
      et doit être utilisée en tenant compte que le token [CLS] est ajouté EN FIN de la séquence :
      pour obtenir l'importance du token [CLS] utilisez avg_attention[:, -1, :-1] (cls_index = -1).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, initialization: str = "kaiming"):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        
        # Projections Q, K classiques
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        
        # V partagé : projection directe vers d_head au lieu de moyenner après projection complète
        self.W_v_shared = nn.Linear(d_model, self.d_head, bias=True)
        
        self.W_out = nn.Linear(d_model, d_model, bias=True) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Pré-calcul du facteur de scaling
        scale = 1.0 / math.sqrt(self.d_head)
        
        # Initialisation
        for m in [self.W_q, self.W_k, self.W_v_shared]:
            if initialization == 'xavier':
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            elif initialization == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Intégration du scaling dans les poids de Q pour éviter la division à chaque forward
        with torch.no_grad():
            self.W_q.weight.mul_(scale)
            if self.W_q.bias is not None:
                self.W_q.bias.mul_(scale)

    def _average_attention_probs(self, attention_probs: Tensor) -> Tensor:
        """Moyenne les probabilités d'attention sur les têtes.

        attention_probs : (batch_size, n_heads, seq_len, seq_len)
        Retour : (batch_size, seq_len, seq_len) moyenné sur l'axe tête.
        """
        return attention_probs.mean(dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass avec sparsemax et V partagé calculé efficacement.

        Args:
            x: Tensor d'entrée de forme (batch_size, seq_len, d_model).

        Returns:
            Tuple[Tensor, Dict] :
              - output : (batch_size, seq_len, d_model) après concat/réprojection des têtes.
              - meta dict contenant "attention_probs" : matrice moyennée (batch_size, seq_len, seq_len).
        
        Notes d'interprétabilité :
        - Pour obtenir l'importance des features pour la prédiction (token [CLS]), lire la DERNIÈRE ligne de
          la matrice renvoyée : avg_attention[:, -1, :] (ou avg_attention[:, -1, :-1] si on omet le token CLS lui-même).
        - Les valeurs sont normalisées par ligne (somme = 1) et, grâce à sparsemax, sont souvent creuses,
          facilitant l'identification des features réellement influentes.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Projections - Q (avec scaling pré-intégré), K, V_shared (direct)
        q = self.W_q(x)  # (batch_size, seq_len, d_model) - scaling déjà appliqué
        k = self.W_k(x)  # (batch_size, seq_len, d_model)
        v_shared = self.W_v_shared(x)  # (batch_size, seq_len, d_head) - projection directe !
        
        # Reshape pour têtes (Q, K seulement)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        
        # Calcul des logits d'attention (pas de division par sqrt(d_head) car intégré dans W_q)
        attention_logits = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
        # Sparsemax par tête
        attention_probs = sparsemax(attention_logits)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        
        # Application de l'attention au V partagé
        # v_shared: (B, T, d_head), on l'expand pour toutes les têtes
        v_expanded = v_shared.unsqueeze(1).expand(batch_size, self.n_heads, seq_len, self.d_head)  # (B, H, T, d_head)
        
        # Sortie d'attention
        x_out = torch.matmul(attention_probs, v_expanded)  # (B, H, T, d_head)
        
        # Concaténation des têtes
        x_out = x_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        if self.W_out is not None:
            x_out = self.W_out(x_out)
        
        # Moyenne des probabilités d'attention pour interprétabilité
        avg_attention = self._average_attention_probs(attention_probs)
        
        return x_out, {"attention_probs": avg_attention}

    def get_attention_weights(self, x: Tensor) -> Tensor:
        """Retourne les probabilités d'attention moyennes sans gradients.

        Utile pour inspection / visualisation (heatmap, barplot des importances).
        """
        with torch.no_grad():
            _, meta = self.forward(x)
        return meta["attention_probs"]