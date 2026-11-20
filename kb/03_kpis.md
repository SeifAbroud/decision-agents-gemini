# Définitions des indicateurs (KPIs)

- Ventes 28 j : somme des ventes sur les 28 derniers jours connus.
- Croissance 28 j (%) : ventes 28 j / ventes des 28 jours précédents - 1.
- Croissance annuelle (%) : ventes 28 j / mêmes 28 jours l'année précédente - 1 (décalage de 364 jours pour garder les jours de semaine).
- Écart au cluster (%) : ventes 28 j du magasin / médiane des ventes 28 j des magasins du même cluster - 1. Un écart très négatif signale un magasin sous-performant par rapport à des magasins comparables.
- Ventes par transaction : ventes 28 j / transactions 28 j (taille moyenne du panier).
- Classe ABC : A = familles qui font 80 % des ventes, B = 15 % suivants, C = 5 % restants.
- Volatilité (CV) : écart-type / moyenne des ventes quotidiennes sur 91 jours.
- Effet promo (%) : médiane du rapport ventes en promotion / moyenne hors promotion pour le même magasin, la même famille et le même jour de semaine, sur 365 jours.
- Incertitude de prévision (%) : (p90 - p10) / prévision. Plus elle est grande, plus il faut de stock de sécurité.
- Prévision 16 j : somme des prévisions journalières (modèle LightGBM global) sur l'horizon Kaggle de 16 jours, calibrée pour corriger le biais du modèle sur les totaux.
