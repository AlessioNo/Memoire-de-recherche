# Prévision des rendements boursiers par apprentissage automatique

Réplication et extension de Gu, Kelly & Xiu (2020) sur un panel d'actions américaines :
quatre modèles (régression linéaire, Elastic Net, LightGBM, Random Forest) sont entraînés
en *walk-forward* pour prédire le rendement excédentaire mensuel de chaque titre, puis
évalués par leur R²hors-échantillon et par des portefeuilles long-short construits à
partir de leurs prédictions.

Le projet comporte aussi trois extensions : un **portefeuille combiné** (moyenne pondérée
des prédictions des quatre modèles), une **analyse par segment de capitalisation**
(où le signal fonctionne-t-il ?) et un **horizon de prédiction à 12 mois** mené en
parallèle de l'horizon à 1 mois.

---

## 1. Principe d'organisation : calculer ≠ afficher

La séparation est stricte, et c'est la clé pour comprendre le dépôt :

- **`scripts/`** fait tout le travail lourd (nettoyage, construction du panel,
  entraînement des 4 modèles × 2 horizons, analyse par taille) et **écrit ses résultats sur
  disque** (`data/`, `modeles/`, `outputs/`). Ces scripts se lancent à la main, avec
  `python`, depuis la racine du projet.
- **`notebooks/`** ne calcule (presque) rien : les notebooks **relisent** ces fichiers et
  les affichent — tableaux, figures, commentaires. Ils sont légers et ré-exécutables en
  quelques secondes (`Run All`).
- **modules à la racine** (`config.py`, `fenetres.py`, `portefeuilles.py`…) : les
  paramètres et la logique partagée, importés aussi bien par les scripts que par les
  notebooks.

Le cycle de travail est donc toujours le même :

```bash
# 1. modifier un paramètre dans config.py
# 2. relancer le ou les scripts concernés (cf. tableau §8)
python scripts/construction_panel.py
python scripts/entrainer/lightgbm_h1.py
# 3. ouvrir le notebook correspondant et Run All pour voir le résultat
```

Deux exceptions à « les notebooks ne calculent rien » : le notebook `08` construit les
portefeuilles et le portefeuille combiné (à partir des prédictions déjà sauvegardées — il
ne ré-entraîne jamais rien), et le notebook `09` agrège le journal des expériences.

---

## 2. Données

`data/raw/` n'est pas versionné : il faut y déposer trois fichiers avant le premier
lancement.

| Fichier | Contenu |
|---|---|
| `datashare.parquet` | caractéristiques d'entreprise mensuelles (univers candidat GKX, 94 colonnes) |
| `StockReturn.parquet` | rendements mensuels par titre |
| `MacroData.parquet` | 8 prédicteurs macroéconomiques |

Le notebook `01_exploration.ipynb` ne sert qu'à inspecter ces fichiers bruts, sans jamais
les modifier. Le traitement commence à l'étape 02.

**Prédicteurs.** `config.CARACTERISTIQUES` déclare la liste candidate (30 caractéristiques
par défaut : momentum, taille/liquidité, risque, valorisation, investissement,
rentabilité, qualité des résultats). L'étape 02 en écarte automatiquement celles dont le
taux de valeurs manquantes dépasse `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` après
`ANNEE_DEBUT`, et écrit le sous-ensemble survivant dans
`data/interim/caracteristiques_retenues.json`. C'est ce sous-ensemble que `PREDICTEURS`
reprend par défaut ; plusieurs recettes alternatives (avec macro, sans telle variable,
univers complet…) sont préparées en commentaire dans `config.py`.

---

## 3. Installation et premier lancement

```bash
pip install -r requirements.txt            # Python 3.13 recommandé
# déposer les 3 fichiers bruts dans data/raw/

python scripts/nettoyage_donnees.py        # étape 02
python scripts/construction_panel.py       # étape 03
python scripts/entrainer/lineaire_h1.py    # étape 04
python scripts/entrainer/elastic_net_h1.py # étape 05
python scripts/entrainer/lightgbm_h1.py    # étape 06  (long)
python scripts/entrainer/random_forest_h1.py  # étape 07  (le plus long)
```

Ou, pour tout enchaîner : `for f in scripts/entrainer/*_h1.py; do python "$f"; done`.

Les dossiers `data/interim/`, `data/processed/`, `modeles/`, `outputs/` et
`outputs/rapports/` sont créés automatiquement s'ils n'existent pas.

Il ne reste qu'à ouvrir les notebooks `02` à `07` (`Run All`) pour visualiser chaque étape,
puis `08` (portefeuilles) et `09` (comparaison des expériences).

---

## 4. Structure du dépôt

```
.
├── config.py                  PARAMÈTRES, et rien d'autre
├── chemins.py                 tous les chemins de sortie, sous forme de fonctions
│
├── fenetres.py                fenêtres glissantes/extensives, R²_oos, embargo
├── horizon.py                 cible composée sur H mois + contexte d'horizon
├── portefeuilles.py           déciles, long-short, mesures de performance
├── ensemble.py                combinaison des prédictions de plusieurs modèles
├── journal.py                 journal des expériences (déduplication par clé)
├── rapports.py                pont scripts → notebooks (valeurs + tableaux)
│
├── entrainement/              le code d'entraînement, écrit une seule fois
│   ├── boucle.py                le protocole commun aux 4 modèles × 2 horizons
│   ├── specs.py                 ce qui distingue chaque modèle (4 classes)
│   └── analyses.py              Fama-MacBeth, stabilité, importances, SHAP
│
├── scripts/                   ce qui se lance à la main
│   ├── nettoyage_donnees.py     étape 02
│   ├── construction_panel.py    étape 03
│   ├── analyse_par_taille.py    étape 10
│   ├── figure_fenetres.py       figure du protocole de fenêtres
│   └── entrainer/               8 entraînements indépendants les uns des autres
│       ├── lineaire_h1.py           lineaire_h12.py
│       ├── elastic_net_h1.py        elastic_net_h12.py
│       ├── lightgbm_h1.py           lightgbm_h12.py
│       └── random_forest_h1.py      random_forest_h12.py
│
├── notebooks/                 01 à 11 — ils affichent, ils ne calculent rien
├── data/{raw,interim,processed}/
├── modeles/                   modèles entraînés (.joblib / .pkl)
└── outputs/                   résultats, figures, journal, rapports/
```

### `config.py` : le seul endroit où l'on change une valeur

`config.py` ne contient que des **paramètres**, jamais de logique ni de chemin. Une valeur
modifiée là est reçue par tous les scripts et tous les notebooks qui importent le module :
aucune valeur n'est jamais recopiée d'un fichier à l'autre. Les paramètres y sont classés
en deux familles, reprises telles quelles par le journal des expériences :

- **généraux** — ils changent le résultat des 4 modèles de la même façon (prédicteurs, mode
  de fenêtres, seuils de filtrage de l'univers) ;
- **spécifiques** — ils ne concernent qu'un modèle (la grille d'alpha de l'Elastic Net n'a
  aucun sens pour LightGBM).

### `chemins.py` : les chemins comme fonctions

Tous les fichiers produits suivent une règle unique,
`outputs/<rôle>_<modèle><suffixe_horizon>.parquet`, écrite une seule fois :

```python
chemins.predictions('lightgbm')                  # outputs/predictions_lightgbm.parquet
chemins.predictions('lightgbm', horizon=12)      # outputs/predictions_lightgbm_h12.parquet
chemins.resultats('elastic_net', par_fenetre=True)
chemins.fichiers_modele('random_forest', 12)     # les 4 chemins d'un coup
```

Ces mêmes chemins restent accessibles sous les noms constants
`config.FICHIER_PREDICTIONS_LIGHTGBM`, `config.fichiers_horizon('lightgbm')`, etc., qui
sont **engendrés** à partir de `chemins.py` (bas de `config.py`) : les deux formes ne
peuvent pas diverger. Dans du code neuf, la forme fonctionnelle est préférable.

### Les 8 fichiers d'entraînement

Chaque fichier de `scripts/entrainer/` fait six lignes :

```python
# scripts/entrainer/elastic_net_h12.py
from entrainement import boucle, specs

if __name__ == "__main__":
    boucle.lancer(specs.ELASTIC_NET, horizon=12)
```

Les quatre modèles suivent en effet exactement le même protocole — mêmes fenêtres, même
recherche d'hyperparamètres sur la validation de chaque fenêtre, même R²_oos poolé, mêmes
fichiers de sortie, même ligne au journal — et ce protocole vit **une seule fois**, dans
`entrainement/boucle.py`. Ce qui distingue un modèle des autres (grille
d'hyperparamètres, construction de l'estimateur, ajustement, colonnes de diagnostic,
analyses annexes) est déclaré dans `entrainement/specs.py`.

Conséquence pratique : le critère de sélection des hyperparamètres est écrit à un seul
endroit (`boucle.py`, ligne `index_meilleur = ...`) ; le changer change les quatre modèles
et les deux horizons d'un coup.

Ces huit scripts sont **indépendants** : aucun n'a besoin qu'un autre ait tourné, et
l'ordre est libre.

**Ajouter un cinquième modèle** demande donc uniquement : une classe dans `specs.py` plus
une entrée dans son registre `TOUS`, et deux fichiers de six lignes dans
`scripts/entrainer/`. Ni `chemins.py` ni `config.py` n'ont de ligne à recevoir.

### `rapports.py` : le pont scripts → notebooks

Les gros résultats ont leur fichier attitré (`panel_pret_modelisation.parquet`,
`resultats_*.parquet`, `predictions_*.parquet`…). `rapports.py` s'occupe de tout le reste :
les compteurs et petits tableaux de diagnostic (lignes retirées par chaque filtre, taux de
valeurs manquantes, coefficients de la dernière fenêtre, durée d'entraînement…) qu'un
notebook doit pouvoir afficher sans refaire le calcul.

```
outputs/rapports/<nom>.json            # les VALEURS (compteurs, listes, textes)
outputs/rapports/<nom>/<clé>.parquet   # les TABLEAUX (DataFrame / Series)
```

Côté script : `rap.valeur('n_lignes', 12345)` / `rap.table('taux_missing', serie)`.
Côté notebook : `rap = rapports.charger('02_nettoyage')` puis `rap.valeur('n_lignes')`.
Si un notebook est ouvert alors que son script n'a jamais tourné, `rapports.charger` lève
une erreur explicite indiquant la commande exacte à lancer.

---

## 5. Le protocole d'évaluation : fenêtres walk-forward

Plutôt qu'un découpage train/validation/test unique, chaque modèle est ré-entraîné sur une
succession de fenêtres qui avancent dans le temps (`fenetres.py`) :

```
fenêtre 0 : train [1980-1997] | validation [1998-2009] | test [2010]
fenêtre 1 : train [1980-1998] | validation [1999-2010] | test [2011]   (expanding)
     ou   : train [1981-1998] | validation [1999-2010] | test [2011]   (rolling)
fenêtre 2 : train [1980-1999] | validation [2000-2011] | test [2012]   (expanding)
```

- `TYPE_FENETRE = "expanding"` : le train garde son point de départ et grandit chaque année
  (choix de GKX). `"rolling"` (défaut du dépôt) : il garde une taille fixe et glisse,
  oubliant les années les plus anciennes.
- Les hyperparamètres sont re-choisis sur la **validation de chaque fenêtre**, jamais sur le
  test.
- Les prédictions de tous les tests, mis bout à bout, forment la série hors-échantillon
  qui donne le R²_oos poolé et alimente les portefeuilles.
- La figure du protocole se génère avec `python scripts/figure_fenetres.py`, à partir des
  valeurs réelles de `config.py` : elle ne peut pas se désynchroniser.

**Deux dates de départ à ne pas confondre.** `ANNEE_DEBUT` filtre la base de données dès
l'étape 02 (les années antérieures n'existent plus nulle part ensuite : la changer oblige à
tout relancer depuis 02). `ANNEE_DEBUT_ENTRAINEMENT` ne filtre que le panel utilisé pour
entraîner : les données restent sur disque, elles sont ignorées au moment de construire les
fenêtres, et seules les étapes 04 à 07 sont à relancer.

**Option : validation qui rétrécit.** `REDUCTION_VALIDATION_PAR_FENETRE` (0 = désactivé),
`FENETRE_DEBUT_REDUCTION_VALIDATION` et `ANNEES_VALIDATION_MINIMUM` permettent de retirer
progressivement des années à la validation au profit du train. Ce qui reste **fixe**, c'est
le test : il avance toujours de `ANNEES_TEST_PAR_FENETRE`, sans trou ni chevauchement — la
validation est donc calée sur sa fin, et le train occupe tout ce qui reste devant elle. Le
découpage réellement obtenu se vérifie dans le tableau `resume_fenetres` des rapports 04 à
07 (colonnes `n_annees_train`, `n_annees_validation`).

---

## 6. Le projet, étape par étape

**01 — Exploration** *(notebook seul)*
Premier coup d'œil aux 3 fichiers bruts, sans rien modifier.

**02 — Nettoyage** — `scripts/nettoyage_donnees.py`
Trois parties indépendantes : A caractéristiques (dont le filtre automatique des
candidates trop incomplètes), B rendements, C macro.
`data/raw/*` → `data/interim/*` + `caracteristiques_retenues.json` + rapport `02_nettoyage`.

**03 — Construction du panel** — `scripts/construction_panel.py`
A : fusion des 3 fichiers nettoyés et calcul de la cible `excess_return` (et de la cible
composée à 12 mois). B : préparation pour la modélisation — filtres taille et liquidité
mois par mois, imputation, winsorizing, rank transform dans [−1, 1]. B écrit aussi une
colonne purement descriptive, `mvel1_brut` (capitalisation avant rank transform), utilisée
par la seule étape 10 et jamais par les modèles.
`data/interim/*` → `data/processed/*` + rapport `03_panel`.

**04 — Régression linéaire** — `scripts/entrainer/lineaire_h1.py`
Benchmark sans hyperparamètre, ré-estimé à chaque fenêtre ; significativité des variables
par Fama-MacBeth. Sortie : `outputs/significativite_regression_lineaire.parquet`.

**05 — Elastic Net** — `scripts/entrainer/elastic_net_h1.py`
Linéaire régularisé, grille d'alpha/l1_ratio re-choisie à chaque fenêtre ; stabilité de
sélection des variables.

**06 — LightGBM** — `scripts/entrainer/lightgbm_h1.py`
Gradient boosting avec arrêt anticipé sur la validation de chaque fenêtre ; importance
gain/split et valeurs SHAP.

**07 — Random Forest** — `scripts/entrainer/random_forest_h1.py`
Bagging : arbres profonds indépendants, chacun sur un échantillon bootstrap et une
fraction des prédicteurs à chaque nœud. Réduit la **variance** là où LightGBM réduit le
**biais**, d'où leur complémentarité dans le portefeuille combiné. Pas d'arrêt anticipé
possible : la régularisation passe par `max_depth`, `min_samples_leaf`, `max_features`.
Fournit en prime un R²_oos *out-of-bag*.

Chacune des étapes 04 à 07 produit les mêmes cinq sorties : le modèle de la dernière
fenêtre (`modeles/`), `predictions_<modèle>.parquet`, `resultats_<modèle>.parquet` et sa
version `_par_fenetre`, un tableau d'importance, plus une ligne au journal des expériences
et un rapport pour le notebook correspondant.

**08 — Évaluation et portefeuilles** *(notebook seul)*
A : comparaison des R²_oos (poolé et par fenêtre). B : portefeuilles long-short par décile
— Sharpe, Sortino, drawdown… — construits à partir des prédictions sur disque, plus des
constructions alternatives (long only, pondération par capitalisation). C : portefeuille
combiné (§7). Chaque résultat est taguée de sa `cle_experience` et **ajoutée** à
l'historique cumulatif, sans rien écraser.

**09 — Comparaison des expériences** *(notebook seul)*
Compare tous les lancements passés entre eux, sans rien recalculer (§9).

**10 — Analyse par taille** — `scripts/analyse_par_taille.py` (§10)

**11 — Horizon 12 mois** *(notebook seul)* (§11)

---

## 7. Le portefeuille combiné (`ensemble.py`, notebook 08 partie C)

Plutôt que de choisir *un* modèle, la partie C combine les prédictions de plusieurs d'entre
eux en une prédiction unique, puis en fait un portefeuille long-short évalué exactement
comme les autres. L'idée vient de Bates & Granger (1969) : deux modèles qui se trompent sur
des choses **différentes** se complètent, et leur combinaison a une erreur plus faible que
chacun pris isolément — la diversification d'un portefeuille, appliquée aux prévisions.

| `METHODE_PONDERATION_ENSEMBLE` | Poids | Principe |
|---|---|---|
| `manuelle` | constants | ceux fixés dans `POIDS_ENSEMBLE` |
| `egale` | constants | 1/N — la référence à battre |
| `r2_validation` | constants | ∝ au R²_oos de **validation** de chaque modèle |
| `inverse_variance` | variables | ∝ 1/EQM, ré-estimés chaque mois (Bates & Granger, 1969) |
| `moindres_carres` | variables | poids minimisant l'erreur de la combinaison, ré-estimés chaque mois (Granger & Ramanathan, 1984 ; *stacking*) |

⚠️ **Aucune fuite de données** : les poids appliqués au mois *t* ne sont estimés que sur des
mois strictement antérieurs (au plus `FENETRE_PONDERATION_ENSEMBLE_MOIS`). Les premiers
mois, faute d'historique (`MOIS_MINIMUM_PONDERATION_ENSEMBLE`), retombent sur des poids
égaux. Des poids optimaux calculés sur toute la période de test donneraient un Sharpe
flatteur et irréalisable : c'est l'erreur classique de cet exercice.

ℹ️ Ne pas s'étonner si `egale` bat les méthodes estimées : c'est le *forecast combination
puzzle* (Smith & Wallis, 2009), un résultat empirique robuste — les poids estimés sont
eux-mêmes bruités. Toujours comparer à `egale` avant de conclure.

Chaque configuration d'ensemble est enregistrée comme une **expérience à part entière**
(nom `Ensemble`, clé incluant celles des modèles sources) et apparaît donc au notebook 09 à
côté des modèles individuels. Ses colonnes `r2_oos_train` / `r2_oos_validation` valent
`NaN` : l'ensemble n'a pas de phase d'entraînement propre.

Sorties : `outputs/predictions_ensemble.parquet` et `outputs/poids_ensemble_par_mois.parquet`
(les poids effectivement appliqués chaque mois — à tracer pour vérifier qu'ils ne partent
pas dans tous les sens).

---

## 8. Quel script relancer après quel changement dans `config.py` ?

| Paramètre modifié | Scripts à relancer |
|---|---|
| `ANNEE_DEBUT`, `CARACTERISTIQUES`, `SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES` | `nettoyage_donnees`, puis `construction_panel`, puis les 8 `entrainer/*` (**tout**, en chaîne) |
| `SEUIL_PERCENTILE_TAILLE`, `SEUIL_PERCENTILE_LIQUIDITE` | `construction_panel`, puis les 8 `entrainer/*` |
| `PREDICTEURS`, `TYPE_FENETRE`, `ANNEE_DEBUT_ENTRAINEMENT`, `ANNEES_*`, `REDUCTION_VALIDATION_PAR_FENETRE`, `FENETRE_DEBUT_REDUCTION_VALIDATION` | les 8 `entrainer/*` (02 et 03 restent intacts) |
| `GRILLE_*_ELASTIC_NET`, `MAX_ITER_ELASTIC_NET` | `entrainer/elastic_net_h1` et `_h12` |
| `GRILLE_*_LIGHTGBM`, `STOPPING_ROUNDS_LIGHTGBM` | `entrainer/lightgbm_h1` et `_h12` |
| `GRILLE_*_RANDOM_FOREST`, `MAX_SAMPLES_RANDOM_FOREST` | `entrainer/random_forest_h1` et `_h12` |
| `N_JOBS_RANDOM_FOREST`, `OOB_SCORE_RANDOM_FOREST`, `GARDER_CANDIDATS_RANDOM_FOREST`, `PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO` | `entrainer/random_forest_*` — sans créer de nouvelle expérience (ils ne changent pas les prédictions) |
| `NB_DECILES` | aucun script — relancer le notebook 08, puis `analyse_par_taille` |
| `MODE_GROUPES_TAILLE` et ses seuils/noms, `MULTIPLICATEUR_MVEL1_EN_DOLLARS`, `NOM_GROUPE_UNIVERS_COMPLET` | `analyse_par_taille` seulement |
| `MODELES_ENSEMBLE`, `METHODE_PONDERATION_ENSEMBLE`, `POIDS_ENSEMBLE`, `FENETRE_PONDERATION_ENSEMBLE_MOIS`, `MOIS_MINIMUM_PONDERATION_ENSEMBLE`, `POIDS_ENSEMBLE_POSITIFS` | aucun script — relancer le notebook 08 (partie C) |

Dans tous les cas : le notebook 08 est à ré-exécuter après tout ré-entraînement (il dépend
des quatre modèles à la fois), et le 09 après le 08.

### Relances partielles

Chaque script sauvegarde son rapport au fur et à mesure : un plantage tardif ne fait jamais
perdre les étapes déjà terminées. Quelques options évitent de tout refaire :

```bash
# rejouer seulement la partie B de l'étape 03 (filtres, imputation, rank transform)
python scripts/construction_panel.py --partie-b-seulement

# sauter la significativité Fama-MacBeth
python scripts/entrainer/lineaire_h1.py --sans-fama-macbeth

# sauter le calcul SHAP
python scripts/entrainer/lightgbm_h1.py --sans-shap
python scripts/entrainer/random_forest_h1.py --sans-shap
```

⚠️ Sur le Random Forest, SHAP est nettement plus lent que sur LightGBM (arbres bien plus
gros) : `--sans-shap` est à utiliser sans hésiter.

### Random Forest : temps de calcul contre mémoire

`GARDER_CANDIDATS_RANDOM_FOREST` décide comment la forêt gagnante de chaque fenêtre est
récupérée une fois la grille parcourue. Les résultats sont **identiques** dans les deux cas
(`random_state = 0`) — seuls le temps et la mémoire changent.

| | `True` (défaut) | `False` |
|---|---|---|
| Forêts de la grille | gardées en mémoire | jetées, seuls les scores sont conservés |
| Gagnante | reprise telle quelle | reconstruite et ré-entraînée |
| Entraînements (grille *G* × *F* fenêtres) | *G × F* | *(G+1) × F* |
| RAM pendant une fenêtre | toute la grille | une seule forêt |

Mesuré sur une grille de 4 combinaisons, `False` coûte environ +25 % de temps. La mémoire
d'une forêt est proportionnelle à `n_train / min_samples_leaf` : diviser `min_samples_leaf`
par 10 la multiplie par 10. Rien à surveiller pour autant — avec `True`, le script estime
la mémoire avant de commencer et bascule seul sur `False` (en le signalant) si l'estimation
dépasse `PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO`.

---

## 9. Le journal des expériences (`journal.py`, notebook 09)

Les scripts d'entraînement **écrasent** leurs fichiers de résultats à chaque exécution : un
seul jeu à la fois, celui du dernier lancement — c'est voulu, pour que le notebook 08 lise
toujours « le » dernier modèle entraîné sans ambiguïté. Le journal résout le problème que
cela pose.

- **`outputs/journal_experiences.parquet`** n'est jamais écrasé : chaque entraînement y
  ajoute une ligne avec ses paramètres généraux (lus dans `config.py` au moment de
  l'appel), ses paramètres spécifiques et ses R²_oos. Une **clé unique** (hash des
  paramètres) déduplique : relancer deux fois exactement la même expérience ne crée jamais
  de doublon.
- **`outputs/historique_performance_portefeuilles.parquet`** suit le même principe depuis
  le notebook 08 : les mesures de portefeuille des expériences pas encore vues y sont
  ajoutées, taguées de la même clé, sans toucher aux lignes existantes.

Le notebook 09 regroupe le journal par (modèle, paramètres spécifiques) — un tableau par
groupe, une ligne par combinaison de paramètres généraux testée — et enrichit chaque ligne
des mesures de portefeuille correspondantes. On peut ainsi comparer `expanding` vs
`rolling`, ou plusieurs grilles d'hyperparamètres, sans jamais perdre un résultat passé.

---

## 10. L'analyse par taille (étape 10 / notebook 10)

L'étape 10 répond à une question que le notebook 08 ne pose pas : **où** le modèle
fonctionne-t-il ? Elle reprend les **mêmes** prédictions et les **ré-évalue séparément** sur
les petites et les grandes capitalisations.

⚠️ Aucun ré-entraînement : les 4 modèles restent entraînés *et* évalués sur l'univers
complet, conformément au protocole de Gu, Kelly & Xiu (2020) ; seule l'évaluation est
découpée. Cette analyse ne crée donc aucune ligne au journal et ne touche aucun fichier lu
par les notebooks 04 à 09 — d'où son notebook dédié.

Le découpage n'est **pas stocké** : il est dérivé de `mvel1_brut` à chaque exécution, selon
`MODE_GROUPES_TAILLE` (`'mediane'`, `'terciles'`, `'quintiles'`, `'personnalise'` ou
`'dollars'`). Changer de découpage ne demande donc que de relancer l'étape 10.

```bash
python scripts/construction_panel.py --partie-b-seulement   # écrit mvel1_brut (une fois)
python scripts/analyse_par_taille.py                        # découpe et ré-évalue
# puis ouvrir notebooks/10_analyse_par_taille.ipynb
```

Deux points de méthode :

- les **déciles sont recalculés à l'intérieur de chaque sous-univers**, mois par mois
  (`portefeuilles.py`) : réutiliser les déciles de l'univers complet donnerait des
  portefeuilles vides ou minuscules certains mois ;
- le **R²_oos n'est pas comparable d'un groupe à l'autre** (son dénominateur est la variance
  des rendements du sous-univers, et les petites capitalisations sont plus volatiles). La
  comparaison entre groupes passe par le **rank-IC** et le **Sharpe** du long-short.

---

## 11. L'horizon de prédiction à 12 mois (`*_h12`, notebook 11)

Piste **parallèle** : au lieu du rendement excédentaire du mois suivant, on prédit celui des
12 mois suivants, composé — Π(1+R) − Π(1+Rf), rendements et taux sans risque capitalisés
séparément puis soustraits (seul choix cohérent avec `excess_return`).

Elle **s'ajoute** au pipeline, elle ne le remplace pas : toutes ses sorties portent le
suffixe `_h12` et n'écrasent jamais celles de la piste à 1 mois. L'horizon n'est qu'un
argument — les fichiers `*_h12.py` appellent le même `entrainement/boucle.py` que les
`*_h1.py`, avec `horizon=12`, et seuls la cible, l'embargo et le suffixe des fichiers
changent :

```python
with horizon.contexte(12):
    ...   # config.CIBLE == 'excess_return_12m', embargo == 12 mois
# ici, tout est revenu à l'état d'origine — même si le bloc a levé une exception
```

```bash
python scripts/construction_panel.py        # écrit les DEUX cibles dans le panel
python scripts/entrainer/lineaire_h12.py    # puis les 3 autres, dans n'importe quel ordre
# puis ouvrir notebooks/11_horizon_12_mois.ipynb
```

Choix méthodologiques, documentés dans `config.py` et rappelés en tête du notebook 11 :

- **embargo de 12 mois** aux frontières train/validation/test
  (`fenetres.appliquer_embargo`) : sans lui, la cible des derniers mois du train porterait
  sur des rendements de la validation ;
- **radiations conservées**, par liquidation au dernier rendement observé puis placement au
  taux sans risque (Shumway, 1997) — les écarter produirait un biais de survie sévère ;
- **trous au milieu écartés** (12 dates par trou) et **censure de fin d'échantillon écartée
  uniformément**, y compris pour les titres radiés ;
- **portefeuilles à cohortes chevauchantes** (Jegadeesh-Titman) et **rebalancement annuel**,
  calculés tous les deux ;
- **pas de Fama-MacBeth** : les observations se chevauchent sur 11 mois, la règle de lags
  Newey-West de l'étape 04 donnerait des t-stats largement surévaluées.

⚠️ Le R²_oos **n'est pas comparable entre les deux horizons** (dénominateurs différents,
facteur ~15 sur la variance) : la comparaison passe par le **rank-IC** et le **Sharpe**.

---

