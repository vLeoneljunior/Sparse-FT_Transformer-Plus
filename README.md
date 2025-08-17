# Sparse FTT+ : Feature Tokenizer Transformer avec Attention Sparse pour Données Tabulaires

---

## 1. Introduction

Sparse FTT+ (Sparse Feature Tokenizer Transformer Plus) est une architecture conçue pour l'apprentissage sur données tabulaires, combinant performance et interprétabilité. Cette variante du FT-Transformer utilise une attention sparse basée sur la fonction `sparsemax` pour réduire les interactions non pertinentes entre features, identifier explicitement les features les plus influents, et offrir une transparence accrue des décisions du modèle. Sparse FTT+ est particulièrement adapté aux applications nécessitant une explication claire des prédictions, comme dans la banque, l'assurance, ou la santé.

---

## 2. Description Technique de Sparse FTT+

Sparse FTT+ est une version interprétable du FT-Transformer qui intègre une attention sparse avec `sparsemax` et un poids de valeur (V) partagé entre les têtes pour une interprétabilité directe. Le modèle traite les données tabulaires (numériques et catégoriques) comme une séquence de tokens, en utilisant un token `[CLS]` pour agréger l'information et produire des prédictions.

### 2.1 Schéma global du forward pass

<div align="center">
  <img src="images/FT_Transformer architecture.png" alt="Architecture globale du Sparse FTT+ appliqué aux données tabulaires" width="500"/>
  <br>
  <b>Architecture globale du Sparse FTT+ appliqué aux données tabulaires</b>
</div>

### 2.2 Étapes détaillées

#### 2.2.1 Tokenisation des features

Le `FeatureTokenizer` (basé sur `rtdl`) encode les variables numériques et catégoriques en vecteurs denses de dimension `d_token` :
- **Variables numériques** : Transformées via une projection linéaire (par défaut) ou une transformation personnalisée (par exemple, linéaire suivie d'une ReLU).
- **Variables catégoriques** : Encodées en embeddings appris en fonction des cardinalités des catégories.

La tokenisation transforme les données brutes en une séquence de vecteurs uniformes que le Transformer peut traiter.

<div align="center">
  <img src="images/Illustration%20d'un%20Feature%20Tokenizer.png" alt="Illustration du processus de tokenisation des variables brutes en vecteurs denses" width="500"/>
  <br>
  <b>Illustration du processus de tokenisation des variables brutes en vecteurs denses</b>
</div>

#### 2.2.2 Ajout du token CLS

Un token spécial `[CLS]`, appris, est ajouté à la fin de la séquence pour agréger les informations des features et servir de base pour la prédiction finale.

#### 2.2.3 Passage dans les blocs Transformer

Chaque bloc applique successivement une attention sparse interprétable, un FFN, et une normalisation résiduelle.

##### Interpretable Multi-Head Attention

- **Attention sparse** : Utilise `sparsemax` (au lieu de `softmax`) pour produire des poids d'attention creuses, concentrés sur un sous-ensemble restreint de features, améliorant l'interprétabilité.
- **Valeur partagée (V)** : Une seule matrice de valeur (V) est partagée entre toutes les têtes, éliminant les distortions dues à des transformations V différentes et rendant les scores d'attention directement comparables.
- **Moyenne des scores d'attention** : Calculée sur les têtes pour refléter l'importance réelle de chaque feature.

<div align="center">
  <img src="images/Scaled Dot-Product Attention.png" alt="Scaled Dot-Product Attention avec sparsemax pour Sparse FTT+" width="300"/>
  <br>
  <b>Scaled Dot-Product Attention avec sparsemax pour Sparse FTT+</b>
</div>

<div align="center">
  <img src="images/Interpretable Multi-Head Attention.png" alt="Illustration de l'Interpretable Multi-Head Attention avec V partagé" width="500"/>
  <br>
  <b>Interpretable Multi-Head Attention : V partagé et sparsemax pour une interprétabilité directe</b>
</div>

##### Feed-Forward Network (FFN)

- Transformation non-linéaire appliquée à chaque token, avec une dimension cachée configurable pour équilibrer capacité et coût computationnel.

##### Normalisation & Résidualité

- **LayerNorm** : Appliquée avant ou après les blocs pour stabiliser les gradients.
- **Connexions résiduelles** : Pour faciliter l'entraînement en permettant un flux direct des gradients.

<div align="center">
  <img src="images/One Transformer layer.png" alt="Vue d'ensemble d'un bloc Transformer adapté aux données tabulaires (Sparse FTT+)" width="300"/>
  <br>
  <b>Vue d'ensemble d'un bloc Transformer adapté aux données tabulaires (Sparse FTT+)</b>
</div>

#### 2.2.4 Head de classification

La prédiction finale est obtenue à partir du token `[CLS]` via une couche linéaire avec activation.

#### 2.2.5 Interprétabilité

- **Importance des features** : Les scores sont extraits directement de la matrice d'attention CLS→features, normalisés et souvent creuses grâce à `sparsemax`.
- **Avantage de sparsemax** : Les poids nuls pour les features non pertinentes rendent les scores d'importance plus interprétables et concentrés sur les features déterminantes.
- **Valeur partagée** : Garantit que les scores d'attention reflètent directement les contributions des features sans distortion.

---

## 3. Structure du Code

```
sparse_ftt_plus/
    attention.py         # Implémentation de l'attention sparse interprétable (sparsemax, V partagé)
    model.py             # Architecture Sparse FTT+ (FeatureTokenizer, CLS, blocs Transformer, head)
```

---

## 4. Objectifs de cette Étude

- Comprendre et expliquer les décisions des modèles tabulaires : enjeu crucial en entreprise (banque, assurance, santé).
- Allier performance et transparence : lever le « black box effect » des réseaux profonds via une attention sparse.
- Proposer une architecture réutilisable : code modulaire pour l'analyse de données tabulaires.

---

## 5. Références

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*.
- Gorishniy, Y., Rubachev, I., & Babenko, A. (2021). *On Embeddings for Numerical Features in Tabular Deep Learning*.
- Martins, A. F. T., & Astudillo, R. F. (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*.
- Lim, B., Arik, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.

---

## Auteur

**Léonel VODOUNOU**  
Sparse FTT+ – (Feature Tokenizer-Transformer interprétable pour données tabulaires)  
2025