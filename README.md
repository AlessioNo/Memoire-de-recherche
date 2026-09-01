# Mémoire Alessio Nocera

Prédiction des rendements boursiers par apprentissage automatique (régression linéaire,
Elastic Net, LightGBM, Random Forest), et combinaison de leurs prédictions en un
portefeuille d'ensemble

## Principe : les calculs sont dans `scripts/`, les notebooks n'affichent que les résultats

Le projet sépare strictement **calculer** et **montrer** :

- **les fichiers de `scripts/`** font tout le travail lourd (nettoyage, construction du
  panel, entraînement des 4 modèles sur les 2 horizons) et écrivent leurs résultats sur
  disque. On les lance **à la main**, avec `python`, depuis la racine du projet.
- **`notebooks/02` à `07`** ne calculent plus rien : ils **relisent** ces résultats et les
  affichent (tableaux, graphiques, commentaires). Ils sont légers et ré-exécutables à
  volonté (`Run All` en quelques secondes).

Concrètement, le cycle de travail est toujours le même :

```bash
# 1. modifier un paramètre dans config.py
# 2. relancer le(s) script(s) concerné(s)
python scripts/construction_panel.py
python scripts/entrainer/lineaire_h1.py
# 3. ouvrir le notebook correspondant et Run All pour voir le résultat
```

Les notebooks `01` (exploration), `08` (portefeuilles) et `09` (comparaison des
expériences) n'ont pas de script associé : ils sont légers ou purement en lecture, et
n'ont pas eu besoin d'être scindés. Le notebook `08` fait exception à la règle « les
notebooks ne calculent rien » : il construit les portefeuilles (parties B et C), mais
uniquement à partir des prédictions déjà sauvegardées — il ne ré-entraîne jamais rien.

## Données

Le dossier `data/raw/` n'est pas versionné et doit être rempli à la main avant le premier
lancement, avec les 3 fichiers sources :

- `datashare.parquet` — les 94 caractéristiques d'entreprise candidates (univers GKX)
- `StockReturn.parquet` — rendements mensuels par titre
- `MacroData.parquet` — les 8 prédicteurs macroéconomiques
- `F-F_Research_Data_5_Factors_2x3.csv` — MKT-RF, SMB, HML, RMW, CMA (Ken French)
- `F-F_Momentum_Factor.csv` — MOM (Ken French)

Les deux derniers se téléchargent depuis la Data Library de Ken French
(`F-F_Research_Data_5_Factors_2x3_CSV.zip` et `F-F_Momentum_Factor_CSV.zip`) et se
dézippent tels quels. ⚠️ Ne pas les ouvrir puis les ré-enregistrer avec un tableur : celui-ci
reformate la première colonne, et le bloc mensuel n'est alors plus reconnu au nettoyage.
Contrairement aux trois premiers, ils **ne servent jamais à prédire** — ce sont des séries
d'évaluation, consommées uniquement par le notebook 08 (alphas et comparaison de Sharpe).

Le notebook `01_exploration.ipynb` sert uniquement à inspecter ces fichiers bruts, sans
jamais les modifier. Le nettoyage proprement dit commence à l'étape 02.

## Structure du projet

```
memoire/
├── config.py                  PARAMÈTRES, et rien d'autre (aucun chemin)
├── chemins.py                 tous les chemins de sortie, sous forme de fonctions
│
├── fenetres.py                fenêtres glissantes/extensives, R²_oos, embargo
├── horizon.py                 cible composée sur H mois + contexte d'horizon
├── journal.py                 journal des expériences (déduplication par clé)
├── rapports.py                pont scripts → notebooks (valeurs + tableaux)
├── portefeuilles.py           déciles, long-short, mesures de performance
├── ensemble.py                combinaison de plusieurs modèles
│
├── entrainement/              LE code d'entraînement, écrit une seule fois
│   ├── boucle.py                le protocole commun aux 4 modèles × 2 horizons
│   ├── specs.py                 ce qui distingue chaque modèle (4 classes)
│   └── analyses.py              Fama-MacBeth, stabilité, importances, SHAP
│
├── scripts/                   ce qui se lance à la main
│   ├── nettoyage_donnees.py     étape 02
│   ├── construction_panel.py    étape 03
│   ├── analyse_par_taille.py    étape 10
│   ├── figure_fenetres.py       figure du protocole de fenêtres
│   └── entrainer/               LES 8 ENTRAÎNEMENTS, indépendants les uns des autres
│       ├── lineaire_h1.py           lineaire_h12.py
│       ├── elastic_net_h1.py        elastic_net_h12.py
│       ├── lightgbm_h1.py           lightgbm_h12.py
│       └── random_forest_h1.py      random_forest_h12.py
│
├── notebooks/                 01 à 11 — ils AFFICHENT, ils ne calculent rien
├── data/{raw,interim,processed}/
├── modeles/                   modèles entraînés (.joblib / .pkl)
└── outputs/                   résultats, figures, journal, rapports/
```

### Les 8 fichiers d'entraînement

Chacun fait six lignes. Tout le calcul est dans `entrainement/boucle.py` :

```python
# scripts/entrainer/elastic_net_h12.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from entrainement import boucle, specs

if __name__ == "__main__":
    boucle.lancer(specs.ELASTIC_NET, horizon=12)
```

Ils sont **indépendants** : aucun n'a besoin qu'un autre ait tourné, et l'ordre n'a pas
d'importance. Lance-les un par un, ou tous d'affilée :

```bash
for f in scripts/entrainer/*.py; do python "$f"; done
```

### Où est passé le code d'entraînement ?

Les quatre modèles suivent exactement le même protocole : mêmes fenêtres, même recherche
d'hyperparamètres sur la validation de chaque fenêtre, même R²_oos poolé, mêmes fichiers de
sortie, même ligne au journal. Ce protocole est écrit **une seule fois**, dans
`entrainement/boucle.py`.

Ce qui distingue un modèle des trois autres tient en une poignée de points, déclarés dans
`entrainement/specs.py` : la grille d'hyperparamètres, la façon de construire l'estimateur,
la façon de l'ajuster (LightGBM a besoin d'un `eval_set`), les colonnes de diagnostic, et
les analyses annexes.

**Conséquence pratique.** Le critère de sélection des hyperparamètres est écrit à un seul
endroit (`boucle.py`, ligne `index_meilleur = ...`) : le changer change les quatre modèles
et les deux horizons d'un coup, sans risque d'oubli.

### Ajouter un cinquième modèle

1. une classe dans `entrainement/specs.py`, plus une entrée dans son registre `TOUS` ;
2. deux fichiers de six lignes dans `scripts/entrainer/`.

Rien d'autre : les chemins de sortie sont engendrés par `chemins.py`, la boucle est
générique, et `config.py` n'a aucune ligne à recevoir.

### L'horizon est un argument, pas une copie du code

`horizon=1` et `horizon=12` empruntent **littéralement le même chemin de code**. Seuls la
cible, l'embargo aux frontières des blocs et le suffixe des fichiers changent, et les trois
sont dérivés de ce seul argument :

```python
with horizon.contexte(12):
    ...   # config.CIBLE == 'excess_return_12m', embargo == 12 mois
# ici, tout est revenu à l'état d'origine — même si le bloc a levé une exception
```

⚠️ Ce `with` remplace l'ancienne fonction `activer_mode_horizon()`, qui basculait ces deux
valeurs **définitivement**. Tant que la bascule n'était pas annulée, tout code exécuté
ensuite dans le même processus — un notebook, un test, un second modèle enchaîné —
travaillait silencieusement sur la mauvaise cible.

## Comment fonctionne le code : `config.py` au centre

`config.py` ne contient plus que des **paramètres**. Change une valeur là, une seule fois :
tous les scripts et tous les notebooks qui l'importent reçoivent le changement, sans qu'une
valeur soit jamais recopiée d'un fichier à l'autre.

Les **chemins** vivent dans `chemins.py`, sous forme de fonctions plutôt que de constantes
écrites une par une :

```python
chemins.predictions('lightgbm')                 # outputs/predictions_lightgbm.parquet
chemins.predictions('lightgbm', horizon=12)     # outputs/predictions_lightgbm_h12.parquet
chemins.resultats('elastic_net', par_fenetre=True)
chemins.fichiers_modele('random_forest', 12)    # les 4 chemins d'un coup
```

Un cinquième modèle ne demande donc aucune ligne dans `chemins.py`.

ℹ️ **Compatibilité.** Les anciens noms `config.FICHIER_PREDICTIONS_LIGHTGBM`,
`config.fichiers_horizon('lightgbm')`, etc. existent toujours : ils sont **engendrés** à
partir de `chemins.py` (bas de `config.py`), donc impossible qu'un nom de fichier diverge
entre les deux formes. Les notebooks 01 à 11 fonctionnent sans modification. Dans du code
neuf, préfère la forme fonctionnelle.

## Le pont scripts → notebooks : `rapports.py`

Les gros résultats du projet ont chacun leur fichier attitré dans `config.py`
(`panel_pret_modelisation.parquet`, `resultats_*.parquet`, `predictions_*.parquet`,
`significativite_*`, `importance_*`...). `rapports.py` s'occupe de **tout le reste** : les
compteurs et petits tableaux de diagnostic (nombre de lignes retirées à chaque filtre, taux
de valeurs manquantes, coefficients de la dernière fenêtre, durée d'entraînement...) qui
n'étaient auparavant qu'imprimés au fil des cellules, et qui seraient donc perdus une fois
le calcul déplacé dans un script.

Chaque script écrit un rapport dans `outputs/rapports/` :

```
outputs/rapports/<nom>.json            # les VALEURS (compteurs, listes, textes)
outputs/rapports/<nom>/<clé>.parquet   # les TABLEAUX (DataFrame / Series)
```

Côté script : `rap.valeur('n_lignes', 12345)` / `rap.table('taux_missing', serie)`.
Côté notebook : `rap = rapports.charger('02_nettoyage')` puis `rap.valeur('n_lignes')` /
`rap.table('taux_missing')`.

Si un notebook est ouvert alors que son script n'a jamais tourné, `rapports.charger` lève
une **erreur explicite indiquant la commande exacte à lancer**, plutôt qu'un
`FileNotFoundError` illisible.

## Premier lancement du projet

1. Installer les dépendances : `pip install -r requirements.txt` (Python 3.13 recommandé).
2. Placer les 3 fichiers bruts dans `data/raw/` (voir section Données).
3. Depuis la **racine** du projet, lancer les scripts dans l'ordre :

```bash
python scripts/nettoyage_donnees.py
python scripts/construction_panel.py
python scripts/entrainer/lineaire_h1.py
python scripts/entrainer/elastic_net_h1.py
python scripts/entrainer/lightgbm_h1.py         # long
python scripts/entrainer/random_forest_h1.py    # le plus long
```

Ou, pour tout enchaîner :

```bash
for f in scripts/entrainer/*_h1.py; do python "$f"; done
```

Chaque script crée automatiquement `data/interim/`, `data/processed/`, `modeles/`,
`outputs/` et `outputs/rapports/` s'ils n'existent pas encore.

4. Ouvrir les notebooks `02` à `07` (`Run All`) pour visualiser les résultats, puis `08`
   (portefeuilles long-short **et portefeuille combiné**) et `09` (comparaison des
   expériences).

## Quel script relancer après quel changement dans `config.py` ?

Les scripts se lancent à la main : ce tableau remplace l'ancienne détection automatique.

| Paramètre modifié | Scripts à relancer |
|---|---|
| `ANNEE_DEBUT`, `CARACTERISTIQUES`, `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` | `nettoyage_donnees`, puis `construction_panel`, puis les 8 `entrainer/*` (**tout**, en chaîne) |
| `SEUIL_PERCENTILE_TAILLE`, `SEUIL_PERCENTILE_LIQUIDITE` | `construction_panel`, puis les 8 `entrainer/*` |
| `PREDICTEURS`, `TYPE_FENETRE`, `ANNEE_DEBUT_ENTRAINEMENT`, `ANNEES_*`, `REDUCTION_VALIDATION_PAR_FENETRE`, `FENETRE_DEBUT_REDUCTION_VALIDATION` | les 8 `entrainer/*` (02 et 03 restent intacts) |
| `GRILLE_*_ELASTIC_NET`, `MAX_ITER_ELASTIC_NET` | `entrainer/elastic_net_h1` et `_h12` |
| `GRILLE_*_LIGHTGBM`, `STOPPING_ROUNDS_LIGHTGBM` | `entrainer/lightgbm_h1` et `_h12` |
| `GRILLE_*_RANDOM_FOREST`, `MAX_SAMPLES_RANDOM_FOREST` | `entrainer/random_forest_h1` et `_h12` |
| `N_JOBS_RANDOM_FOREST`, `OOB_SCORE_RANDOM_FOREST`, `GARDER_CANDIDATS_RANDOM_FOREST`, `PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO` | `entrainer/random_forest_*`, et **sans** créer de nouvelle expérience (ils ne changent pas les prédictions) |
| `NB_DECILES` | aucun script — relancer le notebook 08, puis `analyse_par_taille` |
| `MODE_GROUPES_TAILLE` et ses seuils/noms, `MULTIPLICATEUR_MVEL1_EN_DOLLARS`, `NOM_GROUPE_UNIVERS_COMPLET` | **`analyse_par_taille` seulement**. Ni ré-entraînement, ni reconstruction du panel : les groupes sont dérivés de `mvel1_brut` à chaque exécution de l'étape 10 |
| `MODELES_ENSEMBLE`, `METHODE_PONDERATION_ENSEMBLE`, `POIDS_ENSEMBLE`, `FENETRE_PONDERATION_ENSEMBLE_MOIS`, `MOIS_MINIMUM_PONDERATION_ENSEMBLE`, `POIDS_ENSEMBLE_POSITIFS` | aucun script — relancer le notebook 08 (partie C) |

Dans tous les cas, le notebook 08 est à ré-exécuter après tout ré-entraînement (il dépend
des **quatre** modèles à la fois), et le notebook 09 après le 08.

## L'analyse par taille (étape 10 / notebook 10)

L'étape 10 répond à une question que le notebook 08 ne pose pas : **où** le modèle
fonctionne-t-il ? Elle reprend les **mêmes** prédictions (`outputs/predictions_*.parquet`) et
les **ré-évalue séparément** sur les petites et les grandes capitalisations.

⚠️ **Aucun ré-entraînement, et aucune modification des étapes 04 à 07** : les 4 modèles
restent entraînés *et* évalués sur l'univers complet. C'est le protocole de Gu, Kelly & Xiu
(2020). Cette analyse ne crée donc **aucune ligne au journal des expériences** et ne touche à
aucun fichier lu par les notebooks 04 à 09 — d'où son notebook dédié (10) plutôt qu'une
section du notebook 09, qui compare des *expériences* entre elles.

Comment ça s'articule :

- l'**étape 03** (partie B, section B.2bis) écrit **une seule** colonne descriptive dans
  `panel_pret_modelisation.parquet` : `mvel1_brut`, la capitalisation **avant** le rank
  transform (après quoi `mvel1` vit dans [−1, 1] et n'est plus lisible en dollars). Les
  étapes 04 à 07 l'ignorent, puisqu'elles sélectionnent explicitement `config.PREDICTEURS` ;
- le **découpage en groupes** n'est *pas* stocké : il est dérivé de `mvel1_brut` par
  l'étape 10, à chaque exécution. Changer `MODE_GROUPES_TAILLE` ne demande donc **que** de
  relancer l'étape 10, et aucune désynchronisation n'est possible. Cinq découpages :
  `'mediane'`, `'terciles'`, `'quintiles'`, `'personnalise'` (percentiles sur mesure) et
  `'dollars'` (seuils absolus en dollars, avec zone tampon optionnelle) ;
- `portefeuilles.py` contient la logique des déciles, du long-short et des mesures de
  performance, **partagée** entre le notebook 08 (univers complet) et l'étape 10 (par
  segment) — elle vivait auparavant dans les cellules du notebook 08 ;
- `scripts/analyse_par_taille.py` calcule tout, `notebooks/10_analyse_par_taille.ipynb`
  affiche **tous les groupes en même temps** (jamais un groupe par exécution).

```bash
python scripts/construction_panel.py --partie-b-seulement   # écrit mvel1_brut (une seule fois)
python scripts/analyse_par_taille.py                        # découpe et ré-évalue
# puis ouvrir notebooks/10_analyse_par_taille.ipynb
```

Pour changer de découpage ensuite, seule la deuxième commande est à relancer.

⚠️ **Le `R²_oos` n'est pas comparable d'un groupe à l'autre** (son dénominateur est la
variance des rendements du sous-univers, et les petites capitalisations sont plus volatiles).
La comparaison entre groupes se fait sur le **rank-IC** et le **Sharpe** du long-short — voir
les trois avertissements méthodologiques en tête du notebook 10.

## Deux dates de départ à ne pas confondre

- `ANNEE_DEBUT` (section « Nettoyage ») filtre la **base de données** dès l'étape 02. Les
  années antérieures n'existent plus nulle part ensuite — la changer oblige à tout relancer
  depuis 02.
- `ANNEE_DEBUT_ENTRAINEMENT` (section « Fenêtres ») ne filtre que le panel utilisé pour
  **entraîner les modèles**. Les données restent dans `panel_pret_modelisation.parquet`,
  elles sont simplement ignorées au moment de construire les fenêtres. Tester un démarrage
  plus tardif ne demande donc que de relancer 04/05/06.

⚠️ Retarder `ANNEE_DEBUT_ENTRAINEMENT` raccourcit d'autant la période disponible : il faut
ajuster à la main `ANNEES_TRAIN_INITIAL` et/ou `ANNEES_VALIDATION`, sinon il reste moins
d'années de test (voire aucune fenêtre — `fenetres.py` lève alors une erreur explicite).

## Faire rétrécir la validation au fil des fenêtres (optionnel)

Par défaut, la validation garde la même taille à toutes les fenêtres. Trois paramètres de
`config.py` permettent de la faire **rétrécir** progressivement, au profit de
l'entraînement :

| Paramètre | Rôle |
|---|---|
| `REDUCTION_VALIDATION_PAR_FENETRE` | nb d'années retirées à la validation à chaque nouvelle fenêtre (`0` = désactivé, comportement d'origine) |
| `FENETRE_DEBUT_REDUCTION_VALIDATION` | numéro de la **première** fenêtre raccourcie (`1` = dès la deuxième fenêtre) |
| `ANNEES_VALIDATION_MINIMUM` | plancher : la validation ne descend jamais en dessous |

Exemple avec `ANNEES_TRAIN_INITIAL = 10`, `ANNEES_VALIDATION = 10`,
`ANNEES_TEST_PAR_FENETRE = 1`, une réduction de 1 an dès la fenêtre 1, à partir de 1980 :

```
fenêtre 0 : train [1980-1989] | validation [1990-1999] (10 ans) | test [2000]
fenêtre 1 : train [1980-1991] | validation [1992-2000] ( 9 ans) | test [2001]   (expanding)
     ou   : train [1982-1991] | validation [1992-2000] ( 9 ans) | test [2001]   (rolling)
fenêtre 2 : train [1980-1993] | validation [1994-2001] ( 8 ans) | test [2002]   (expanding)
```

⚠️ Ce qui reste **fixe**, c'est le test : il avance toujours de `ANNEES_TEST_PAR_FENETRE`
d'une fenêtre à l'autre, sans trou ni chevauchement — sinon la série hors-échantillon mise
bout à bout (donc le R²_oos final et les portefeuilles du notebook 08) n'aurait plus de
sens. La validation est donc calée sur sa **fin** (elle s'arrête juste avant le test), et le
train occupe tout ce qui reste devant elle. L'année libérée par la validation passe au
**train** : en `expanding` il grandit de `ANNEES_TEST_PAR_FENETRE +
REDUCTION_VALIDATION_PAR_FENETRE` par fenêtre (12 ans à la fenêtre 1 ci-dessus, au lieu de
11 sans réduction) ; en `rolling` il garde sa taille fixe et glisse d'autant.

Le découpage réellement obtenu se vérifie dans le tableau `resume_fenetres` (rapports
04/05/06, affiché en tête des notebooks — colonnes `n_annees_train` et
`n_annees_validation`) et sur la figure `python scripts/figure_fenetres.py`.

Ces trois paramètres n'entrent dans la signature d'une expérience (`cle_experience`,
notebook 09) **que** lorsque la réduction est active : les expériences déjà au journal,
lancées avant l'ajout de cette option, gardent donc exactement la même clé.

## Le portefeuille combiné (notebook 08, partie C)

Plutôt que de choisir *un* modèle, la partie C du notebook 08 **combine** les prédictions de
plusieurs d'entre eux en une prédiction unique, puis en fait un portefeuille long-short
évalué exactement comme les autres. L'idée vient de Bates & Granger (1969) : deux modèles qui
se trompent sur des choses **différentes** se complètent, et leur combinaison a une erreur
plus faible que chacun pris isolément — la même logique que la diversification d'un
portefeuille, appliquée aux prévisions.

Tout se règle dans `config.py`, section « Portefeuille COMBINÉ » ; la logique vit dans
`ensemble.py`.

| Méthode (`METHODE_PONDERATION_ENSEMBLE`) | Poids | Principe |
|---|---|---|
| `manuelle` | constants | ceux fixés dans `POIDS_ENSEMBLE` |
| `egale` | constants | 1/N — la référence à battre |
| `r2_validation` | constants | ∝ au R²_oos de **validation** de chaque modèle |
| `inverse_variance` | variables | ∝ 1/EQM, ré-estimés chaque mois (Bates & Granger, 1969) |
| `moindres_carres` | variables | poids minimisant l'erreur de la combinaison, ré-estimés chaque mois (Granger & Ramanathan, 1984 ; *stacking*) |

⚠️ **Aucune fuite de données** : les poids appliqués au mois *t* ne sont jamais estimés sur
le mois *t* ni sur un mois postérieur — uniquement sur les mois strictement antérieurs (au
plus `FENETRE_PONDERATION_ENSEMBLE_MOIS`). Les premiers mois, faute d'historique
(`MOIS_MINIMUM_PONDERATION_ENSEMBLE`), retombent sur des poids égaux. Des poids optimaux
calculés sur toute la période de test donneraient un Sharpe flatteur et irréalisable :
c'est l'erreur classique de cet exercice.

ℹ️ Ne sois pas surpris si `egale` bat les méthodes estimées : c'est le « forecast
combination puzzle » (Smith & Wallis, 2009), un résultat empirique très robuste — les poids
estimés sont eux-mêmes bruités. Compare systématiquement à `egale` avant de conclure.

Chaque configuration d'ensemble est enregistrée comme une **expérience à part entière**
(nom `Ensemble`, clé incluant les `cle_experience` des modèles sources) : elle apparaît donc
au notebook 09 à côté des modèles individuels, avec son R²_oos et ses mesures de
portefeuille. Ses colonnes `r2_oos_train` / `r2_oos_validation` valent `NaN` — l'ensemble
n'a pas de phase d'entraînement propre.

Sorties : `outputs/predictions_ensemble.parquet` et `outputs/poids_ensemble_par_mois.parquet`
(les poids effectivement appliqués chaque mois, à tracer pour vérifier qu'ils ne partent pas
dans tous les sens).

## Voir toutes les combinaisons d'hyperparamètres testées

Les scripts 05 et 06 enregistrent le score de **chaque** combinaison de leur grille, sur le
train et sur la validation, pour **chaque** fenêtre (tableau `grille_complete` du rapport).
Les notebooks 05 et 06 l'affichent en section 3bis : une heatmap du R²_oos de validation,
une heatmap de l'écart `R²_train − R²_validation` (diagnostic de sur-apprentissage), et les
tableaux correspondants.

Ces tableaux ne couvrent que le **dernier lancement** de chaque script — la comparaison
entre plusieurs lancements reste le rôle du notebook 09.

La sélection elle-même se fait toujours sur le meilleur R²_oos de validation. Pour la baser
sur l'écart train-validation, il faut changer la ligne `index_meilleur = ...`
(`entrainer/elastic_net_h1.py`) ou `meilleur_index = ...` (`entrainer/lightgbm_h1.py`).

## Options de relance partielle

Chaque script sauvegarde son rapport **au fur et à mesure** : un plantage tardif ne fait
jamais perdre les étapes déjà terminées. Quelques options en ligne de commande évitent de
tout refaire :

```bash
# rejouer seulement la partie B de l'étape 03 (filtres, imputation, rank transform)
# à partir de panel_final.parquet, sans refaire la fusion
python scripts/construction_panel.py --partie-b-seulement

# sauter la significativité Fama-MacBeth (section 6bis du notebook 04)
python scripts/entrainer/lineaire_h1.py --sans-fama-macbeth

# sauter le calcul SHAP (section 6bis des notebooks 06 et 07)
python scripts/entrainer/lightgbm_h1.py --sans-shap
python scripts/entrainer/random_forest_h1.py --sans-shap
```

⚠️ Sur le Random Forest, SHAP est nettement plus lent que sur LightGBM (arbres bien plus
gros) : `--sans-shap` est une option à utiliser sans hésiter.

### Random Forest : temps de calcul contre mémoire

`GARDER_CANDIDATS_RANDOM_FOREST` (`config.py`) décide de **comment** la forêt gagnante de
chaque fenêtre est récupérée, une fois la grille parcourue. Les résultats sont **identiques**
dans les deux cas (`random_state = 0` est fixé) — seuls le temps et la mémoire changent.

| | `True` (défaut) | `False` |
|---|---|---|
| Forêts de la grille | gardées en mémoire | jetées, seuls les scores sont conservés |
| Gagnante | reprise telle quelle | reconstruite et **ré-entraînée** |
| Entraînements, grille de *G* × *F* fenêtres | *G × F* | *(G+1) × F* |
| RAM pendant une fenêtre | toute la grille | une seule forêt |

Mesuré sur une grille de 4 combinaisons : `False` coûte **+25 %** de temps.

**Comment estimer la mémoire.** Les feuilles se partagent les lignes du train sans
recouvrement, donc leur nombre est plafonné à la fois par `2^max_depth` et par
`n_train / (2 × min_samples_leaf)` — c'est le **plus petit des deux** qui s'applique. Un arbre
à *F* feuilles a `2F − 1` nœuds, et scikit-learn stocke 64 octets par nœud (8 tableaux de 8
octets). D'où, pour un train de 1,2 M de lignes et 300 arbres :

| `max_depth` | `min_samples_leaf` | RAM par forêt |
|---|---|---|
| 8 | 1000 | ~10 Mo |
| `None` | 1000 | ~23 Mo |
| `None` | 100 | ~230 Mo |
| `None` | 10 | ~2,3 Go |

⚠️ La mémoire est proportionnelle à `n_train / min_samples_leaf` : **diviser
`min_samples_leaf` par 10 la multiplie par 10**, et allonger la période d'entraînement
l'augmente aussi. C'est ce paramètre-là qui décide, pas la taille du panel en soi.

Tu n'as rien à surveiller : avec `True`, le script **estime la mémoire avant de commencer** et
bascule seul sur le comportement `False` (en te le disant à l'écran, et dans le notebook 07)
si l'estimation dépasse `PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO`.

## Garder la trace de chaque expérience (notebook 09)

Les scripts 04/05/06 **écrasent** leurs fichiers de résultats
(`outputs/resultats_*.parquet`) à chaque exécution : un seul jeu de résultats à la fois,
celui du dernier lancement (voulu, pour que le notebook 08 lise toujours « le » dernier
modèle entraîné sans ambiguïté). Un ancien résultat serait donc normalement perdu dès qu'on
relance le même modèle avec d'autres paramètres — c'est ce que le système journal /
notebook 09 résout.

**Journal des expériences** (`outputs/journal_experiences.parquet`, écrit par `journal.py`,
jamais écrasé) : à chaque exécution d'un script 04/05/06, une ligne y est ajoutée avec les
paramètres généraux (lus dans `config.py` au moment de l'appel) et spécifiques du
lancement, plus les R²_oos obtenus. Une **clé unique** (hash des paramètres) déduplique
automatiquement : relancer deux fois exactement la même expérience n'ajoute jamais de
doublon.

**Historique de performance des portefeuilles**
(`outputs/historique_performance_portefeuilles.parquet`, écrit par `journal.py` depuis le
notebook 08) fonctionne sur le même principe : à chaque exécution de 08, les mesures
(Sharpe, Sortino, drawdown...) des expériences pas encore vues y sont ajoutées, taguées par
la même clé que dans le journal, sans jamais toucher aux lignes déjà présentes.

Le notebook 09 regroupe ensuite le journal par `(modèle, paramètres spécifiques)` — un
tableau par groupe, une ligne par combinaison de paramètres généraux testée — puis enrichit
chaque ligne avec les mesures de portefeuille de l'historique, en les reliant par cette clé
commune. Une expérience apparaît avec `NaN` sur les colonnes de portefeuille uniquement si
le notebook 08 n'a **jamais encore** tourné avec ses prédictions sur disque ; sinon elle
garde ses mesures même après avoir été supplantée par un lancement plus récent du même
modèle. Ça permet de comparer, par exemple, `expanding` vs `rolling`, ou plusieurs grilles
d'hyperparamètres Elastic Net, sans jamais perdre un résultat passé.

## Le projet, étape par étape

**01 — Exploration** (notebook seul)
Premier coup d'œil aux 3 fichiers bruts, sans rien modifier.
- Entrée : `data/raw/*` — Sortie : —

**02 — Nettoyage des données**
Partie A caractéristiques (+ filtre automatique des candidates trop incomplètes, section
A.3bis), Partie B rendements, Partie C macro, Partie D facteurs FF5 + MOM — 4 nettoyages
indépendants. La partie D remet les facteurs de Ken French en décimal (ils sont publiés en
pourcentage) et fusionne les deux CSV en un seul fichier mensuel.
- Script : `scripts/nettoyage_donnees.py`
- Entrée : `data/raw/*`
- Sortie : `data/interim/*` (+ `caracteristiques_retenues.json`) + rapport `02_nettoyage`

**03 — Construction du panel**
Partie A fusion des 3 fichiers nettoyés, Partie B préparation pour la modélisation
(filtres taille/liquidité, imputation, winsorizing, rank transform).
- Script : `scripts/construction_panel.py`
- Entrée : `data/interim/*`
- Sortie : `data/processed/*` + rapport `03_panel`

**04 — Modèle : régression linéaire**
Benchmark simple, sans hyperparamètre, ré-entraîné à chaque fenêtre ; significativité des
variables par Fama-MacBeth (section 6bis).
- Script : `scripts/entrainer/lineaire_h1.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/regression_lineaire.joblib`,
  `outputs/predictions_regression_lineaire.parquet`,
  `outputs/resultats_regression_lineaire*.parquet` (+ `cle_experience`),
  `outputs/significativite_regression_lineaire.parquet`, + 1 ligne dans
  `outputs/journal_experiences.parquet` + rapport `04_regression_lineaire`

**05 — Modèle : Elastic Net**
Linéaire régularisé, hyperparamètres re-choisis sur la `validation` de chaque fenêtre ;
stabilité de sélection des variables (section 6bis).
- Script : `scripts/entrainer/elastic_net_h1.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/elastic_net.joblib`, `outputs/predictions_elastic_net.parquet`,
  `outputs/resultats_elastic_net*.parquet` (+ `cle_experience`),
  `outputs/importance_elastic_net.parquet`, + 1 ligne dans le journal + rapport
  `05_elastic_net`

**06 — Modèle : LightGBM**
Gradient boosting, arrêt anticipé sur la `validation` de chaque fenêtre, grille
d'hyperparamètres (nombre de feuilles, learning rate, minimum d'observations par feuille,
budget maximal d'arbres) ; importance gain/split + valeurs SHAP (section 6bis).
- Script : `scripts/entrainer/lightgbm_h1.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/lightgbm.joblib`, `outputs/predictions_lightgbm.parquet`,
  `outputs/resultats_lightgbm*.parquet` (+ `cle_experience`),
  `outputs/importance_lightgbm.parquet`, + 1 ligne dans le journal + rapport `06_lightgbm`
  (valeurs SHAP incluses)

**07 — Modèle : Random Forest**
Bagging : N arbres profonds **indépendants**, chacun sur un échantillon bootstrap et une
fraction des prédicteurs à chaque nœud. Réduit la **variance** là où LightGBM réduit le
**biais** — d'où leur complémentarité, exploitée en partie C du notebook 08. Pas d'arrêt
anticipé possible (les arbres sont indépendants) : la régularisation passe par `max_depth`,
`min_samples_leaf` et `max_features`. Fournit en prime un **R²_oos out-of-bag**, une
évaluation hors-échantillon gratuite qu'aucun des trois autres modèles ne permet.
- Script : `scripts/entrainer/random_forest_h1.py`
- Entrée : `panel_pret_modelisation.parquet`
- Sortie : `modeles/random_forest.joblib`, `outputs/predictions_random_forest.parquet`,
  `outputs/resultats_random_forest*.parquet` (+ `cle_experience`),
  `outputs/importance_random_forest.parquet`, + 1 ligne dans le journal + rapport
  `07_random_forest` (valeurs SHAP incluses)

**08 — Évaluation finale (dernier lancement)** (notebook seul)
Partie A comparaison du R²_oos (pooled + évolution par fenêtre), Partie B portefeuilles
long-short par décile (Sharpe, Sortino, drawdown...) à partir des prédictions déjà
sauvegardées, taguées avec `cle_experience` et **ajoutées** (sans rien écraser) à
l'historique cumulatif pour le notebook 09, **Partie C portefeuille combiné** (voir la
section dédiée plus haut), **Partie D évaluation factorielle FF5 + MOM**, Synthèse R²_oos vs
Sharpe — toujours le **dernier** modèle entraîné de chaque type.
- Entrée : `outputs/resultats_*.parquet`, `outputs/predictions_*.parquet`,
  `data/interim/facteurs_clean.parquet`
- Sortie : `outputs/*`, dont `outputs/performance_portefeuilles.parquet` (instantané),
  `outputs/historique_performance_portefeuilles.parquet` (cumulatif),
  `outputs/predictions_ensemble.parquet`, `outputs/poids_ensemble_par_mois.parquet`,
  `outputs/alphas_portefeuilles.parquet` et
  `outputs/comparaison_modeles_vs_facteurs.parquet`

La **partie D** répond à deux questions qu'il ne faut pas confondre. L'**alpha** demande si
le long-short rapporte ce que les facteurs n'expliquent pas : régression
`r_LS,t = α + β' F_t + ε_t`, écarts-types Newey-West, modèles emboîtés CAPM → FF3 → FF5 →
FF5+MOM. La **comparaison** demande si le long-short fait mieux qu'un portefeuille construit
avec les facteurs eux-mêmes : Sharpe contre Sharpe, dans un tableau unique où les quatre
modèles, l'ensemble, le 1/N factoriel et chaque facteur isolé occupent des lignes
comparables. Un alpha positif significatif accompagné d'un Sharpe inférieur à celui du
momentum seul est cohérent : rendement orthogonal et rendement total sont deux grandeurs
différentes. Toute la logique vit dans `facteurs.py` (racine), les paramètres dans
`config.py`, section « Évaluation factorielle ».

⚠️ Ces sorties ne passent **pas** par le journal des expériences. Elles ne dépendent
d'aucune prédiction nouvelle, et ajouter des colonnes d'alpha à
`journal.COLONNES_PORTEFEUILLE` rendrait l'historique déjà écrit inconcaténable avec les
lignes futures. Même traitement que `performance_constructions.parquet`.

**09 — Comparaison des expériences** (notebook seul)
Compare **tous** les lancements passés de 04/05/06/07 entre eux (R²_oos, temps
d'entraînement), ainsi que chaque portefeuille combiné enregistré par le notebook 08,
regroupés par modèle et hyperparamètres spécifiques, enrichis des mesures de portefeuille
de l'historique cumulatif du notebook 08 — ne ré-entraîne rien.
- Entrée : `outputs/journal_experiences.parquet`,
  `outputs/historique_performance_portefeuilles.parquet`
- Sortie : affichage seulement, rien de sauvegardé

## L'horizon de prédiction long (`*_h12` / notebook 11)

Piste **parallèle** : au lieu du rendement excédentaire du mois suivant, on prédit celui des
12 mois suivants, composé — Π(1+R) − Π(1+Rf).

⚠️ **Elle s'ajoute, elle ne remplace rien.** Les étapes 04 à 07 et les notebooks 04 à 10 ne
sont ni modifiés ni relancés : leurs sorties ne sont jamais écrasées, celles de cette piste
portant le suffixe `_h12`.

**L'horizon n'est qu'un argument.** Les fichiers `*_h12.py` appellent exactement le même
`entrainement/boucle.py` que les `*_h1.py`, avec `horizon=12` : seuls la cible, l'embargo et
le suffixe des fichiers changent. Une correction apportée à la boucle profite donc
automatiquement aux deux pistes — ce qui n'était pas le cas quand elles vivaient dans huit
fichiers séparés.

Les quatre modèles sont **indépendants** : on en entraîne un à la fois, et le notebook 11
s'ouvre avec n'importe quel sous-ensemble déjà exécuté.

```bash
python scripts/construction_panel.py            # écrit les DEUX cibles dans le panel
python scripts/entrainer/lineaire_h12.py        # puis les 3 autres quand tu veux
# puis ouvrir notebooks/11_horizon_12_mois.ipynb
```

Choix méthodologiques, tous documentés dans `config.py` et rappelés en tête du notebook 11 :

- **cible excédentaire** : rendements et taux sans risque capitalisés séparément puis
  soustraits — seul choix cohérent avec `excess_return` ;
- **embargo de 12 mois** aux frontières train/validation/test (`fenetres.appliquer_embargo`) :
  sans lui, la cible des derniers mois du train porterait sur des rendements de la validation ;
- **radiations conservées** par liquidation au dernier rendement observé puis placement au
  taux sans risque (Shumway 1997) — les écarter produirait un biais de survie sévère ;
- **trous au milieu écartés** (12 dates par trou) et **censure de fin d'échantillon écartée
  uniformément**, y compris pour les titres radiés ;
- **portefeuilles à cohortes chevauchantes** (Jegadeesh-Titman) et **rebalancement annuel**,
  calculés tous les deux ;
- **pas de Fama-MacBeth** : les observations se chevauchent sur 11 mois, la règle de lags
  Newey-West de l'étape 04 donnerait des t-stats largement surévaluées.

⚠️ Le `R²_oos` **n'est pas comparable entre les deux horizons** (dénominateurs différents,
facteur ~15 sur la variance) : la comparaison passe par le **rank-IC** et le **Sharpe**.
