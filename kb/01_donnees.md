# Dictionnaire de données (Favorita – Store Sales)

## train.csv / test.csv
- date : jour de vente.
- store_nbr : identifiant du magasin (54 magasins).
- family : famille de produits (33 familles, ex. GROCERY I, BEVERAGES, PRODUCE).
- sales : ventes totales de la famille dans le magasin ce jour-là. Ce sont des volumes (unités), pas un chiffre d'affaires ; des valeurs fractionnaires existent (produits vendus au kilo). Il n'y a pas de prix dans ce jeu de données.
- onpromotion : nombre d'articles de la famille en promotion ce jour-là.

## stores.csv
- city, state : localisation du magasin.
- type : type de magasin (A à E).
- cluster : groupe de magasins similaires. Comparer un magasin à la médiane de son cluster est plus juste que de le comparer à tous les magasins.

## transactions.csv
- transactions : nombre de tickets par magasin et par jour (tous rayons confondus). Ventes / transactions donne un proxy de la taille du panier.

## oil.csv
- dcoilwtico : prix quotidien du pétrole. L'Équateur dépend fortement du pétrole : son prix influence l'économie et la consommation.

## holidays_events.csv
- type : Holiday, Transfer, Additional, Bridge, Work Day, Event.
- locale : National, Regional (state), Local (city).
- transferred = True : le jour férié officiel a été déplacé ; le jour réellement chômé est la ligne de type Transfer.
