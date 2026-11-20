# Rapport d'aide à la décision – Favorita
Généré le 2026-11-18 21:55 · données 2016-01-01 → 2017-08-15 · prévision 2017-08-16 → 2017-08-31 · moteur : hors-ligne (règles)

## 1. Synthèse exécutive
Sur les 28 derniers jours (jusqu'au 2017-08-15), les ventes évoluent de -1.3 % par rapport à la période précédente. Le système a produit 4 recommandations, dont 4 validées par l'agent critique.

1. Renforcer le réapprovisionnement de LINGERIE
2. Réduire les commandes de HOME AND KITCHEN II
3. Concentrer les promotions sur LIQUOR,WINE,BEER

## 2. Indicateurs clés (28 derniers jours)
- Ventes : 2,257,805 (-1.3 % vs 28 j précédents, +7.0 % sur un an)
- Ventes moyennes par jour : 80,636
- Transactions : 210,579
- Effet jours de paie : +5.2 %
- Lignes en promotion : 10.6 %

## 3. Constats de l'analyste
- Ventes des 28 derniers jours : 2257805.1 (-1.3 % vs les 28 jours précédents, 7.0 % sur un an) [E1].
- Les jours de paie font varier les ventes de 5.2 % [E1].
- CLEANING est la première famille avec 28.83 % des ventes [E5].
- Le magasin 31 (Manta) est à -80.2 % de la médiane de son cluster [E4].
- La promotion a le plus d'effet sur LIQUOR,WINE,BEER (+33.3 %) [E7].

## 4. Prévisions (16 jours)
- Meilleur modèle en validation : LightGBM (global) (RMSLE 0.45) [E10].
- Prévision 16 jours : 1295420.6, soit 1.6 % vs les 16 derniers jours [E11].
- Plus forte hausse attendue : LINGERIE (3.6 %) [E12].

| Modèle | RMSLE | WAPE % |
|---|---|---|
| LightGBM (global) | 0.4503 | 39.53 |
| Naive saisonnier (J-21) | 0.6484 | 57.26 |
| Moyenne mobile 28 j | 0.4901 | 42.9 |

| Famille | Prévision 16 j | Derniers 16 j | Variation % |
|---|---|---|---|
| CLEANING | 376,052 | 366,479 | +2.6 |
| DELI | 261,274 | 256,877 | +1.7 |
| PRODUCE | 202,327 | 198,745 | +1.8 |
| AUTOMOTIVE | 75,746 | 76,173 | -0.6 |
| EGGS | 65,491 | 64,516 | +1.5 |
| SEAFOOD | 38,948 | 38,720 | +0.6 |
| BEAUTY | 27,500 | 27,478 | +0.1 |
| BREAD/BAKERY | 27,005 | 26,614 | +1.5 |

## 5. Recommandations
### R1 – Renforcer le réapprovisionnement de LINGERIE  [VALIDEE]
- Action : Augmenter les commandes de LINGERIE pour les 16 prochains jours.
- Justification : La prévision est de 4620.7 contre 4460.7 sur les 16 derniers jours, soit 3.6 %.
- Impact attendu : Moins de ruptures sur la famille en hausse.
- Priorité : haute · preuves : E12
- Contrôle automatique : 3 chiffres vérifiés, OK

### R2 – Réduire les commandes de HOME AND KITCHEN II  [VALIDEE]
- Action : Ajuster à la baisse les commandes de HOME AND KITCHEN II pour éviter le surstock.
- Justification : Prévision 8199.7 contre 8290.4 récemment (-1.1 %).
- Impact attendu : Moins de stock immobilisé et de pertes.
- Priorité : moyenne · preuves : E12
- Contrôle automatique : 3 chiffres vérifiés, OK

### R3 – Concentrer les promotions sur LIQUOR,WINE,BEER  [VALIDEE]
- Action : Réallouer une partie du budget promotionnel de SEAFOOD vers LIQUOR,WINE,BEER.
- Justification : L'effet promo est de 33.3 % sur LIQUOR,WINE,BEER contre -5.5 % sur SEAFOOD.
- Impact attendu : Plus de volume pour le même budget promotionnel.
- Priorité : moyenne · preuves : E7
- Contrôle automatique : 2 chiffres vérifiés, OK

### R4 – Diagnostiquer le magasin 31  [VALIDEE]
- Action : Audit assortiment, ruptures et affluence du magasin 31 (Manta).
- Justification : Ventes 28 j de 12139.1 contre une médiane de cluster de 61373.8 (-80.2 %).
- Impact attendu : Identifier les causes du retard sur les magasins comparables.
- Priorité : haute · preuves : E4
- Contrôle automatique : 4 chiffres vérifiés, OK

## 6. Annexe – preuves (rejouables)
- **E1** `overview({})`
- **E2** `rank_stores({"metric": "growth_28d_pct", "n": 3})`
- **E3** `rank_stores({"metric": "growth_28d_pct", "n": 3, "ascending": true})`
- **E4** `underperforming_stores({"n": 3})`
- **E5** `rank_families({"metric": "share_pct", "n": 5})`
- **E6** `rank_families({"metric": "growth_28d_pct", "n": 3, "ascending": true})`
- **E7** `promo_effect({"n": 3})`
- **E8** `holiday_effects({})`
- **E9** `stockout_signals({"n": 3})`
- **E10** `model_quality({})`
- **E11** `forecast_overview({})`
- **E12** `forecast_movers({"level": "family", "n": 4})`
- **E13** `forecast_movers({"level": "store", "n": 3})`
- **E14** `uncertain_families({"n": 3})`

_Coût LLM de ce rapport : 0 appels, 0 tokens entrée, 0 tokens sortie ≈ 0.0 $_