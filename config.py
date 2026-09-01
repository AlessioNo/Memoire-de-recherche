"""
Parametres du projet -- et RIEN QUE des parametres.

Modifie une valeur ICI, une seule fois : tous les scripts et tous les notebooks qui
importent ce module recoivent le changement, sans qu'aucune valeur ne soit jamais recopiee
d'un fichier a l'autre.

⚠️ Apres avoir change une valeur, il faut RELANCER A LA MAIN le ou les scripts concernes
(voir README.md, tableau "Quel script relancer apres quel changement ?"), puis re-executer
le notebook correspondant pour en visualiser le resultat.

Deux familles de parametres, reperees plus bas par un bandeau, et reprises telles quelles
par `journal.py` :
- Parametres GENERAUX  : ils changent le resultat des 4 modeles de la meme facon (choix des
  predicteurs, mode de fenetres, seuils de filtrage de l'univers).
- Parametres SPECIFIQUES : ils ne concernent qu'un seul modele (la grille d'alpha de
  l'Elastic Net n'a aucun sens pour LightGBM).

Ou sont passes les chemins ?
----------------------------
Les ~50 constantes FICHIER_* qui occupaient le haut de ce fichier vivent maintenant dans
`chemins.py`, sous forme de quelques fonctions (`chemins.predictions('lightgbm')`,
`chemins.resultats('lightgbm', horizon=12)`...). Elles restent accessibles ici sous leurs
anciens noms -- voir la section "Compatibilite" tout en bas : les notebooks 01 a 11
continuent de fonctionner sans la moindre modification.
"""

import json
from pathlib import Path

import numpy as np

import chemins

# ------------------------------------------------------------
# Dossiers et fonctions de chemins, re-exposes depuis chemins.py pour que
# `config.OUTPUTS_DIR`, `config.assurer_dossiers()` etc. continuent de fonctionner.
# ------------------------------------------------------------
RACINE = chemins.RACINE
DATA_RAW = chemins.DATA_RAW
DATA_INTERIM = chemins.DATA_INTERIM
DATA_PROCESSED = chemins.DATA_PROCESSED
MODELES_DIR = chemins.MODELES_DIR
OUTPUTS_DIR = chemins.OUTPUTS_DIR

assurer_dossiers = chemins.assurer_dossiers

# ============================================================
# Periode de depart (utilisee au notebook 02, parties A et C - elles DOIVENT
# rester coherentes entre elles, d'ou l'interet de ne le definir qu'ici)
# ============================================================
ANNEE_DEBUT = 1980


# ============================================================
# Univers CANDIDAT des 94 caracteristiques d'entreprise de Gu, Kelly & Xiu (2020).
#
# CARACTERISTIQUES est l'univers CANDIDAT (les 94 disponibles dans
# datashare.parquet) ; c'est le notebook 02 (partie A, section A.3bis) qui en
# retient automatiquement un SOUS-ENSEMBLE, sur la base du taux de valeurs
# manquantes de chacune apres ANNEE_DEBUT (voir
# SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES juste en dessous) -- plutot qu'une
# selection manuelle figee dans ce fichier.
#
# ============================================================
CARACTERISTIQUES = [
    # --- Tendances de prix / momentum (6) ---
    'mom1m',        # momentum 1 mois (reversion court terme)
    'mom6m',        # momentum 6 mois
    'mom12m',       # momentum 12 mois
    'chmom',        # variation du momentum 6 mois
    'indmom',       # momentum sectoriel
    'maxret',       # rendement quotidien maximum du mois

    # --- Taille et liquidite (7) ---
    'mvel1',        # taille (capitalisation) -- ⚠️ requis par le filtre de l'etape 03
    'dolvol',       # volume echange en dollars
    'turn',         # rotation des titres
    'std_turn',     # volatilite de la rotation
    'ill',          # illiquidite d'Amihud -- ⚠️ requis par le filtre de l'etape 03
    'zerotrade',    # jours sans echange
    'baspread',     # ecart bid-ask

    # --- Risque et volatilite (3) ---
    'retvol',       # volatilite des rendements
    'idiovol',      # volatilite idiosyncratique
    'beta',         # beta de marche

    # --- Valorisation (4) ---
    'bm',           # book-to-market
    'ep',           # benefices / prix
    'sp',           # ventes / prix
    'cfp',          # cash-flow / prix

    # --- Investissement et croissance (4) ---
    'agr',          # croissance des actifs
    'chcsho',       # variation du nombre d'actions
    'egr',          # croissance des capitaux propres
    'invest',       # investissements et stocks

    # --- Rentabilite (4) ---
    'gma',          # rentabilite brute (Novy-Marx)
    'operprof',     # rentabilite operationnelle (Fama-French 2015)
    'roaq',         # rentabilite des actifs
    'roeq',         # rentabilite des capitaux propres

    # --- Qualite des resultats (2) ---
    'acc',          # accruals (Sloan 1996)
    'chtx',         # variation de la charge fiscale
]

"""CARACTERISTIQUES = [
    "mvel1", "beta", "betasq", "chmom", "dolvol", "idiovol", "indmom",
    "mom1m", "mom6m", "mom12m", "mom36m", "pricedelay", "turn",
    "absacc", "acc", "age", "agr", "bm", "bm_ia", "cashdebt", "cashpr",
    "cfp", "cfp_ia", "chatoia", "chcsho", "chempia", "chinv", "chpmia",
    "convind", "currat", "depr", "divi", "divo", "dy", "egr", "ep",
    "gma", "grcapx", "grltnoa", "herf", "hire", "invest", "lev", "lgr",
    "mve_ia", "operprof", "orgcap", "pchcapx_ia", "pchcurrat", "pchdepr",
    "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect",
    "pchsale_pchxsga", "pchsaleinv", "pctacc", "ps", "quick", "rd",
    "rd_mve", "rd_sale", "realestate", "roic", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin", "sp", "tang", "tb",
    "aeavol", "cash", "chtx", "cinvest", "ear", "nincr", "roaq", "roavol",
    "roeq", "rsup", "stdacc", "stdcf", "ms", "baspread", "ill", "maxret",
    "retvol", "std_dolvol", "std_turn", "zerotrade",
]"""


# ============================================================
# Filtre des caracteristiques par taux de valeurs manquantes
#
# ⚠️ Parametre GENERAL (voir journal.py) : il conditionne quelles caracteristiques
# existent meme, donc il est repercute dans PREDICTEURS -- voir
# `charger_caracteristiques_retenues()` juste en dessous.
# ============================================================
SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES = 0.35
# Exclut toute caracteristique manquante sur plus de 35% des lignes (annee >= ANNEE_DEBUT).
# Augmente ce seuil pour etre plus permissif (garder plus de caracteristiques, plus
# imputees), diminue-le pour etre plus strict (panel plus leger, mais moins de
# caracteristiques). 0.30 est un compromis courant dans la litterature empirique --
# ajuste-le et compare le R2_oos (notebook 08) si tu veux challenger ce choix.


def charger_caracteristiques_retenues():
    """Renvoie la liste des caracteristiques REELLEMENT retenues apres le filtre de
    valeurs manquantes ci-dessus, telle que sauvegardee par le notebook 02 (partie A,
    section A.3bis) dans chemins.CARACTERISTIQUES_RETENUES.

    Repli (fallback) : si ce fichier n'existe pas encore (avant la toute premiere
    execution du notebook 02, ex. juste apres un clone du depot), renvoie l'univers
    CANDIDAT complet (`CARACTERISTIQUES`, 94 noms) -- pour que `import config` ne casse
    jamais, meme au tout premier lancement. Une fois le notebook 02 execute au moins
    une fois, cette fonction lit systematiquement le manifeste qu'il a produit.
    """
    if chemins.CARACTERISTIQUES_RETENUES.exists():
        with open(chemins.CARACTERISTIQUES_RETENUES) as f:
            manifeste = json.load(f)
        return manifeste['caracteristiques_retenues']
    return list(CARACTERISTIQUES)


# Calculee une fois a l'import de ce module : c'est CETTE liste (pas CARACTERISTIQUES,
# l'univers candidat complet) que reutilisent le notebook 03 (partie B) et les notebooks
# 04 a 06 pour construire le panel de modelisation et les predicteurs par defaut
# (PREDICTEURS, juste en dessous). Se met a jour automatiquement des que le notebook 02
# est ré-exécuté avec un nouveau SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES.
CARACTERISTIQUES_RETENUES = charger_caracteristiques_retenues()


# ============================================================
# Les 8 predicteurs macro de GKX (deja prefixes "macro_" pour eviter toute
# collision avec des caracteristiques d'entreprise du meme nom, ex: bm, ep)
# ============================================================
MACRO_PREDICTEURS = [
    "macro_dp", "macro_ep", "macro_bm", "macro_ntis",
    "macro_tbl", "macro_tms", "macro_dfy", "macro_svar",
]


# ============================================================
# Predicteurs utilises pour la modelisation (notebooks 04, 05, 06)
# Change UNIQUEMENT cette ligne : les 3 notebooks utilisent
# automatiquement le meme choix, pas besoin de toucher a rien d'autre.
#
# ⚠️ Parametre GENERAL (voir journal.py) : il est enregistre a chaque entrainement dans
# outputs/journal_experiences.parquet, comme TYPE_FENETRE ou les seuils de filtrage
# ci-dessous -- deux lancements avec un choix de predicteurs different apparaitront comme
# deux lignes distinctes au notebook 08, meme si tout le reste est identique.
#

# ============================================================
PREDICTEURS = CARACTERISTIQUES_RETENUES

"""[c for c in CARACTERISTIQUES_RETENUES + MACRO_PREDICTEURS if c not in [ "macro_ep", "macro_bm", "macro_ntis", "macro_tbl", "macro_tms", "macro_dfy"]]"""


# ------------------------------------------------------------
# Autres recettes possibles pour PREDICTEURS -- decommente UNE SEULE des lignes
# `PREDICTEURS = ...` ci-dessous (en la recopiant apres le bloc ci-dessus, ou en la
# substituant a la ligne active) selon ce que tu veux tester. Toutes utilisent
# CARACTERISTIQUES_RETENUES (le sous-ensemble qui a passe le filtre de valeurs
# manquantes), jamais CARACTERISTIQUES (l'univers candidat complet, potentiellement
# tres incomplet sur certaines colonnes) -- sauf la derniere, qui contourne le filtre
# volontairement.
#
# # Caracteristiques retenues seules (sans les 8 predicteurs macro)
# PREDICTEURS = CARACTERISTIQUES_RETENUES
#
# # Macro seules (8)
# PREDICTEURS = MACRO_PREDICTEURS
#
# # Caracteristiques retenues + macro, MOINS une ou plusieurs variables
# PREDICTEURS = [c for c in CARACTERISTIQUES_RETENUES + MACRO_PREDICTEURS if c not in ['mom1m', 'macro_dp']]
#
# # Caracteristiques retenues seules, MOINS une ou plusieurs variables
# PREDICTEURS = [c for c in CARACTERISTIQUES_RETENUES if c not in ['mom1m', 'mom6m']]
#
# # Macro seules, MOINS une ou plusieurs variables
# PREDICTEURS = [c for c in MACRO_PREDICTEURS if c not in ['macro_dp', 'macro_ep']]
#
# # Liste explicite, ecrite a la main (le sous-ensemble exact que tu veux)
# PREDICTEURS = ['mvel1', 'bm', 'ep', 'mom1m', 'mom6m', 'mom12m', 'ill', 'beta', 'turn', 'agr']
#
# # Univers CANDIDAT complet (94), en ignorant volontairement le filtre de valeurs
# # manquantes -- deconseille (voir SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES plus haut),
# # utile seulement pour mesurer l'effet du filtre lui-meme sur le R2_oos.
# PREDICTEURS = CARACTERISTIQUES + MACRO_PREDICTEURS

# ============================================================
# Nom de la variable cible, calculee au notebook 03 (partie A)
# ============================================================
CIBLE = "excess_return"


# ============================================================
# Filtres sur l'univers investissable (notebook 03, partie B, section B.2)
#
# Regroupes ici, a cote du reste des parametres du projet : change-les ICI, PUIS re-execute
# le notebook 03 (au moins la partie B) pour regenerer panel_pret_modelisation.parquet,
# avant de re-entrainer les modeles (04 a 06) sur ce nouvel univers.
#
# ⚠️ Parametres GENERAUX (voir journal.py) : contrairement a TYPE_FENETRE et consorts (qui
# n'affectent que la boucle d'entrainement de 04/05/06), ceux-ci affectent la construction
# meme du panel (notebook 03) -- mais ils sont neanmoins enregistres comme parametres
# generaux dans outputs/journal_experiences.parquet a chaque entrainement, pour garder une
# trace de l'univers utilise par chaque experience.
# ============================================================
SEUIL_PERCENTILE_TAILLE = 0.10
# Filtre taille (mvel1, notebook 03 section B.2.1) : exclut, CHAQUE MOIS, les titres SOUS ce
# percentile de capitalisation boursiere (0.50 = exclut les 2 quartiles inferieurs, la
# moitie la plus petite de l'univers chaque mois).

SEUIL_PERCENTILE_LIQUIDITE = 0.90
# Filtre liquidite (ill, notebook 03 section B.2.2) : exclut, CHAQUE MOIS, les titres
# AU-DESSUS de ce percentile d'illiquidite d'Amihud (0.90 = exclut le decile le plus illiquide).


# ============================================================
# Fenetres d'entrainement glissantes / extensives (notebooks 04 a 08)
#
# Chaque modele est ré-entrainé plusieurs fois,
# sur une succession de fenetres qui avancent dans le temps. Change les
# parametres ICI, les notebooks 04 a 08 recoivent automatiquement le
# changement (la construction des fenetres elle-meme est dans fenetres.py,
# a la racine du projet, a cote de ce fichier).
#
# ⚠️ Parametres GENERAUX (voir journal.py).
# ============================================================
TYPE_FENETRE = "rolling"
# "expanding" : le train GARDE son point de depart d'origine et grandit chaque
#               annee (il n'oublie jamais rien) -- c'est le choix de GKX (2020).
# "rolling"   : le train garde une taille FIXE (ANNEES_TRAIN_INITIAL annees) et
#               glisse dans le temps, en oubliant les annees les plus anciennes.

# Annee de DEBUT DE L'ENTRAINEMENT des modeles (etapes 04, 05, 06).
#
# ⚠️ A ne pas confondre avec ANNEE_DEBUT (section "Nettoyage") : ANNEE_DEBUT filtre la
# BASE DE DONNEES elle-meme des l'etape 02 (les annees anterieures n'existent nulle part
# ensuite) ; ANNEE_DEBUT_ENTRAINEMENT ne filtre QUE le panel utilise pour entrainer les
# modeles. Les donnees restent dans panel_pret_modelisation.parquet, elles sont
# simplement ignorees au moment de construire les fenetres.
#
# A quoi ca sert : tester si demarrer l'entrainement plus tard (donnees plus completes,
# regime de marche plus recent) ameliore ou degrade la performance, SANS avoir a
# re-executer les etapes 02 et 03 -- il suffit de relancer 04/05/06.
#
# ⚠️ Si tu retardes cette date, la periode disponible raccourcit d'autant : il faut
# ajuster A LA MAIN ANNEES_TRAIN_INITIAL et/ou ANNEES_VALIDATION ci-dessous, sinon il
# reste moins d'annees de test a la fin (voire aucune fenetre generee -- fenetres.py
# leve alors une erreur explicite).
#
# Doit etre >= ANNEE_DEBUT. Mettre la meme valeur que ANNEE_DEBUT = aucun filtre
# supplementaire (comportement d'origine).
ANNEE_DEBUT_ENTRAINEMENT = 1980

ANNEES_TRAIN_INITIAL = 18    # nb d'annees d'entrainement de la 1ere fenetre (taille FIXE du train si "rolling")
ANNEES_VALIDATION = 12       # nb d'annees de validation, glisse toujours juste apres le train
ANNEES_TEST_PAR_FENETRE = 1  # nb d'annees de test par fenetre avant de ré-entrainer (1 = ré-entrainement annuel, comme GKX)

# ⚠️ Augmenter ANNEES_TEST_PAR_FENETRE reduit le nombre de fenetres (donc le temps
# de calcul total, surtout pour le notebook 06 LightGBM) au prix d'un ré-entrainement
# moins frequent -- utile si scripts/etape06_modele_lightgbm.py est trop lent sur ton PC.


# ------------------------------------------------------------
# Reduction PROGRESSIVE de la validation au fil des fenetres (optionnel)
#
# Par defaut (REDUCTION_VALIDATION_PAR_FENETRE = 0), la validation garde la meme taille
# a toutes les fenetres : c'est le comportement d'origine du projet, rien ne change.
#
# Avec une reduction > 0, la validation RETRECIT d'autant d'annees a chaque nouvelle
# fenetre, a partir de la fenetre FENETRE_DEBUT_REDUCTION_VALIDATION, sans jamais
# descendre sous ANNEES_VALIDATION_MINIMUM. Exemple avec ANNEES_TRAIN_INITIAL = 10,
# ANNEES_VALIDATION = 10, ANNEES_TEST_PAR_FENETRE = 1, REDUCTION = 1, DEBUT = 1,
# a partir de 1980 :
#
#   fenetre 0 : train [1980-1989] | validation [1990-1999] (10 ans) | test [2000]
#   fenetre 1 : train [1980-1991] | validation [1992-2000] ( 9 ans) | test [2001]  (expanding)
#        ou   : train [1982-1991] | validation [1992-2000] ( 9 ans) | test [2001]  (rolling)
#   fenetre 2 : train [1980-1993] | validation [1994-2001] ( 8 ans) | test [2002]  (expanding)
#
# ⚠️ Ce qui reste FIXE, c'est le test : il avance toujours de ANNEES_TEST_PAR_FENETRE
# d'une fenetre a l'autre, sans trou ni chevauchement -- sinon la serie hors-echantillon
# mise bout a bout (et donc le R2_oos final et les portefeuilles du notebook 07) n'aurait
# plus de sens. La validation est donc calee sur sa FIN (elle s'arrete toujours juste
# avant le test) et le train occupe tout ce qui reste devant elle. L'annee liberee par la
# validation n'est pas perdue : elle passe au TRAIN. En "expanding" le train grandit donc
# de ANNEES_TEST_PAR_FENETRE + REDUCTION_VALIDATION_PAR_FENETRE par fenetre (12 ans a la
# fenetre 1 dans l'exemple ci-dessus, au lieu de 11 sans reduction) ; en "rolling" il
# garde sa taille fixe (ANNEES_TRAIN_INITIAL) et glisse d'autant.
#
# A quoi ca sert : consacrer de plus en plus de donnees a l'entrainement au fil du temps,
# tout en gardant une validation recente pour le choix des hyperparametres (05 et 06).
# ⚠️ Une validation courte rend ce choix plus bruite : compare le R2_oos au notebook 08
# avant de descendre trop bas.
#
# ⚠️ Parametres GENERAUX (voir journal.py) : ils ne sont enregistres dans le journal des
# experiences que lorsque la reduction est active (REDUCTION_VALIDATION_PAR_FENETRE > 0),
# pour que les cles des experiences deja lancees sans reduction restent inchangees.
# ------------------------------------------------------------
REDUCTION_VALIDATION_PAR_FENETRE = 1
# Nb d'annees retirees a la validation A CHAQUE nouvelle fenetre. 0 = desactive
# (validation de taille constante, comportement d'origine).

FENETRE_DEBUT_REDUCTION_VALIDATION = 1
# Numero de la PREMIERE fenetre raccourcie (les fenetres sont numerotees a partir de 0).
# 1 = la reduction commence des la DEUXIEME fenetre ; 0 = des la premiere ; 3 = la
# validation garde sa taille d'origine pour les fenetres 0, 1 et 2, puis retrecit.
# Sans effet si REDUCTION_VALIDATION_PAR_FENETRE = 0.
#
# Pour raisonner en ANNEE plutot qu'en numero de fenetre : la fenetre k teste l'annee
# ANNEE_DEBUT_ENTRAINEMENT + ANNEES_TRAIN_INITIAL + ANNEES_VALIDATION + k (avec
# ANNEES_TEST_PAR_FENETRE = 1), donc commencer a reduire a partir de l'annee de test A
# revient a mettre k = A - (ANNEE_DEBUT_ENTRAINEMENT + ANNEES_TRAIN_INITIAL + ANNEES_VALIDATION).
# Le tableau `resume_fenetres` (rapports 04/05/06, affiche aux notebooks) donne de toute
# facon la correspondance numero de fenetre <-> annee de test.

ANNEES_VALIDATION_MINIMUM = 4
# Plancher : la validation ne descend jamais sous ce nombre d'annees, meme si la
# reduction continue. Doit valoir au moins 1.


# ============================================================
# Hyperparametres SPECIFIQUES a l'Elastic Net (notebook 05 uniquement)
#
# Changes ICI, le notebook 05 les recoit automatiquement (recherche sur grille,
# ré-evaluee une fois par fenetre sur sa propre validation -- voir notebook 05, section 3).
# ⚠️ Parametres SPECIFIQUES (voir journal.py) : contrairement aux parametres generaux
# ci-dessus, ceux-la ne concernent QUE l'Elastic Net -- les enregistrer separement permet
# au notebook 08 de regrouper les runs en un seul tableau par grille testee.
# ============================================================
GRILLE_ALPHA_ELASTIC_NET =   [0.0001, 0.001]    #np.logspace(-5, -1, 5)        # force de regularisation : 1e-7 a 1e-1 (10 valeurs, echelle log)
GRILLE_L1_RATIO_ELASTIC_NET = [0.1,0.3, 0.5, 0.7, 0.9]    # equilibre Lasso (1.0) / Ridge (0.0) : 5 valeurs
MAX_ITER_ELASTIC_NET = 1000                                # nb max d'iterations de l'optimiseur, par modele candidat de la grille


# ============================================================
# Hyperparametres SPECIFIQUES a LightGBM (notebook 06 uniquement)
#
# Changes ICI, le notebook 06 les recoit automatiquement (recherche sur grille + arret
# anticipe, ré-evaluee une fois par fenetre sur sa propre validation -- voir notebook 06,
# section 3). Meme remarque que pour l'Elastic Net ci-dessus sur la separation
# generaux/specifiques, cruciale pour le regroupement des tableaux au notebook 08.
# ============================================================
GRILLE_NUM_LEAVES_LIGHTGBM = [3, 7, 15]                # nb max de feuilles par arbre : 5 valeurs
GRILLE_LEARNING_RATE_LIGHTGBM = [0.01]      # taux d'apprentissage : 1 valeur
GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM = [2000]    # nb min d'observations par feuille : 1 valeur
GRILLE_N_ESTIMATORS_LIGHTGBM = [2000]               # budget MAXIMUM d'arbres : 3 valeurs
# ⚠️ Interaction avec STOPPING_ROUNDS_LIGHTGBM ci-dessous : si l'arret anticipe se
# declenche AVANT le plus petit budget de la grille, toutes les valeurs de
# GRILLE_N_ESTIMATORS_LIGHTGBM donnent exactement le MEME modele (le budget ne mord
# jamais). La colonne 'nb_arbres_utilises' du tableau de grille (rapport '06_lightgbm',
# affiche au notebook 06) permet de le verifier d'un coup d'oeil : si elle est toujours
# strictement inferieure au plus petit budget, autant revenir a une seule valeur.
STOPPING_ROUNDS_LIGHTGBM = 200                            # arret si pas d'amelioration sur la validation depuis N arbres d'affilee


# ============================================================
# Hyperparametres SPECIFIQUES au Random Forest (etape 07 / notebook 07 uniquement)
#
# Meme logique que pour l'Elastic Net et LightGBM ci-dessus : grille ré-evaluee sur la
# validation de CHAQUE fenetre, parametres SPECIFIQUES (voir journal.py) donc separes des
# parametres generaux, pour que le notebook 09 regroupe correctement les lancements.
#
# ⚠️ Le Random Forest est, avec LightGBM, le plus lourd du projet -- mais pour une raison
# differente : LightGBM construit des arbres peu profonds les uns apres les autres, la
# foret construit N_ESTIMATORS arbres PROFONDS et independants. Le cout monte donc
# directement avec n_estimators x profondeur x nb de lignes du train. Trois leviers, par
# ordre d'efficacite : reduire MAX_SAMPLES_RANDOM_FOREST (chaque arbre voit une fraction
# du train), plafonner MAX_DEPTH_RANDOM_FOREST, et garder les grilles courtes.
# ============================================================
GRILLE_N_ESTIMATORS_RANDOM_FOREST = [300]
# Nb d'arbres de la foret. Contrairement a LightGBM, ce n'est PAS un budget dont on peut
# sortir par arret anticipe : les arbres sont independants et tous construits. Plus il y
# en a, plus la prediction est stable -- sans jamais sur-apprendre davantage (Breiman
# 2001), la variance de la moyenne diminue, c'est tout. Au-dela de ~300-500 le gain devient
# marginal alors que le temps de calcul, lui, reste proportionnel.

GRILLE_MAX_DEPTH_RANDOM_FOREST = [3, 5, 8]
# Profondeur maximale de chaque arbre (None = arbres complets). C'est LE parametre de
# regularisation d'une foret sur des rendements mensuels : le signal est si faible que des
# arbres complets memorisent surtout du bruit. Des valeurs faibles (3-10) sont la norme
# dans la litterature asset pricing.

# ------------------------------------------------------------
# max_features : recherche EN DEUX TEMPS (propre au Random Forest)
#
# Fraction des predicteurs tiree au hasard a CHAQUE noeud -- le "random" de random forest,
# celui qui decorrele les arbres entre eux. 1.0 revient a du bagging pur. Sur des
# predicteurs tres correles entre eux (les caracteristiques GKX le sont), une fraction
# basse aide en general. Accepte aussi 'sqrt' ou 'log2'.
#
# ⚠️ CE PARAMETRE N'EST PAS CHERCHE EN PRODUIT CARTESIEN avec les trois autres, contrairement
# a tous les autres hyperparametres du projet. La recherche se fait en DEUX ETAPES, dans
# CHAQUE fenetre :
#
#   Etape A -- max_features FIGE a MAX_FEATURES_ETAPE_A_RANDOM_FOREST, on parcourt le
#              produit cartesien n_estimators x max_depth x min_samples_leaf et on retient
#              le triplet qui maximise le R2_oos de validation DE CETTE FENETRE.
#   Etape B -- ce triplet est a son tour FIGE, et on ne fait plus varier que max_features,
#              sur les valeurs de GRILLE_MAX_FEATURES_RANDOM_FOREST ci-dessous (la valeur
#              de l'etape A est automatiquement ignoree ici : elle a deja ete evaluee).
#
# Le modele retenu de la fenetre est le meilleur R2_oos de validation sur A ∪ B : si aucune
# valeur de l'etape B ne bat celle de l'etape A, le gagnant de A est conserve tel quel.
#
# ℹ️ Pourquoi : c'est une descente par coordonnees, et elle transforme un cout MULTIPLICATIF
# en cout ADDITIF. Avec 3 triplets et 2 valeurs supplementaires de max_features :
#     produit cartesien -> 3 x 3 = 9 forets par fenetre
#     deux etapes       -> 3 + 2 = 5 forets par fenetre
# Chaque valeur ajoutee ci-dessous ne coute donc QU'UNE seule foret de plus par fenetre.
# En contrepartie, l'optimum trouve est conditionnel au triplet de l'etape A : ce n'est pas
# necessairement l'optimum global de la grille complete. C'est le compromis assume ici.
#
# ⚠️ Ce protocole entre dans la cle d'unicite de l'experience (voir specs.py,
# RandomForest.params_specifiques) : une recherche en deux temps et une recherche en produit
# cartesien sur les MEMES grilles restent deux experiences distinctes au journal.
# ------------------------------------------------------------
MAX_FEATURES_ETAPE_A_RANDOM_FOREST = 0.33
# Valeur UNIQUE (un scalaire, pas une liste) utilisee pendant toute l'etape A.
# 0.33 = la regle p/3 de Breiman (2001), defaut historique en regression -- d'ou ce choix
# comme point de depart. Avec ~30 predicteurs, cela fait ~10 variables tirees par noeud.

GRILLE_MAX_FEATURES_RANDOM_FOREST = [0.33]
# Valeurs testees a l'etape B, UNE PAR UNE, avec le triplet gagnant de l'etape A.
# ⚠️ Ne PAS y remettre MAX_FEATURES_ETAPE_A_RANDOM_FOREST : elle serait ignoree de toute
# facon (deja evaluee a l'etape A), mais autant que la liste dise ce qu'elle fait.
# Reperes, avec ~30 predicteurs :
#     0.15 -> ~4 variables par noeud  (forte decorrelation des arbres)
#     0.18 -> ~5 variables            (≈ regle sqrt(p), defaut sklearn en classification)
#     0.33 -> ~10 variables           (regle p/3 de Breiman -- l'etape A)
#     0.66 -> ~20 variables           (randomisation faible)
#     1.00 -> 30 variables            (plus aucune randomisation : BAGGING PUR)
# ℹ️ Garder 1.00 dans la liste a un interet propre : c'est le temoin qui chiffre ce que la
# randomisation rapporte reellement. Si 1.00 gagne systematiquement, ta foret est en fait du
# bagging, et cela merite une phrase dans le memoire.

GRILLE_MIN_SAMPLES_LEAF_RANDOM_FOREST = [2000]
# Nb minimum d'observations par feuille. Second garde-fou contre le bruit, complementaire
# de max_depth : une feuille qui ne contient que quelques titres-mois ne mesure rien de
# reel. A caler sur la taille du panel (equivalent de MIN_CHILD_SAMPLES pour LightGBM).

MAX_SAMPLES_RANDOM_FOREST = None
# Fraction du train tiree (avec remise) pour construire CHAQUE arbre. None = 100% des
# lignes, comportement par defaut de scikit-learn. Mettre 0.3 divise environ par 3 le
# temps de construction de chaque arbre -- le levier le plus efficace si l'etape 07 est
# trop lente sur ton PC. ⚠️ C'est un hyperparametre du modele (il change les resultats),
# pas un simple reglage technique : il est enregistre comme tel dans le journal.

OOB_SCORE_RANDOM_FOREST = True
# Calcule le R2_oos "out-of-bag" du modele retenu de chaque fenetre : chaque observation du
# train est predite uniquement par les arbres qui ne l'ont PAS vue (elle etait hors de leur
# tirage bootstrap). C'est une estimation hors-echantillon gratuite, propre au bagging,
# affichee au notebook 07 a cote du R2 de train et de validation -- utile pour voir a quel
# point le R2 de train est optimiste. Coute quelques secondes par fenetre. Sans effet sur
# le choix des hyperparametres, qui reste fait sur la validation de la fenetre.

GARDER_CANDIDATS_RANDOM_FOREST = True
# COMMENT le modele retenu de chaque fenetre est obtenu, une fois la grille parcourue. Ce
# reglage ne change AUCUN resultat (random_state=0 est fixe : la foret ré-entrainee est
# identique, bit pour bit, a celle de la grille) -- uniquement le temps de calcul et la
# memoire consommee. Il n'entre donc pas dans la signature de l'experience (journal.py).
#
#   True  (defaut) : chaque foret de la grille est GARDEE en memoire ; la gagnante est
#                    simplement reprise dans la liste. C'est le comportement de l'etape 06
#                    (LightGBM). Aucun entrainement en trop.
#                    ⚠️ Toutes les forets de la grille coexistent en RAM pendant la fenetre.
#
#   False          : seuls les SCORES de chaque combinaison sont conserves ; la gagnante est
#                    reconstruite et ré-entrainee a la fin de la fenetre. C'est le
#                    comportement de l'etape 05 (Elastic Net). Une seule foret en memoire a
#                    la fois, au prix d'UN entrainement supplementaire par fenetre.
#
# Comment choisir : c'est une question de taille des forets, et donc de nombre de FEUILLES.
# Les feuilles se partagent les lignes du train sans recouvrement, donc leur nombre est
# plafonne a la fois par 2^max_depth et par n_train / (2 x min_samples_leaf) -- c'est le plus
# petit des deux qui s'applique. La memoire d'une foret vaut environ
#
#     n_estimators x (2 x nb_feuilles) x 64 octets
#
# (64 octets par noeud : scikit-learn stocke 8 tableaux de 8 octets par noeud). Avec un train
# de 1,2 M de lignes et 300 arbres :
#
#     max_depth=8, min_samples_leaf=1000  ->  256 feuilles  ->  ~10 Mo par foret
#     max_depth=None, min_samples_leaf=1000 -> 600 feuilles ->  ~23 Mo par foret
#     max_depth=None, min_samples_leaf=100  -> 6 000 feuilles -> ~230 Mo par foret
#     max_depth=None, min_samples_leaf=10   -> 60 000 feuilles -> ~2,3 Go par foret
#
# Multiplie par le nombre de combinaisons de ta grille pour savoir ce que True consomme.
# ⚠️ La memoire est proportionnelle a n_train / min_samples_leaf : diviser min_samples_leaf
# par 10 la multiplie par 10, et un panel plus long l'augmente aussi. Une grille exploratoire
# avec un min_samples_leaf faible et max_depth=None est le cas ou il faut passer a False.
#
# ℹ️ Tu n'as pas a surveiller ca toi-meme : avec True, le script ESTIME la memoire avant de
# lancer chaque fenetre et bascule automatiquement sur le comportement False (en te le disant
# a l'ecran) si l'estimation depasse PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO ci-dessous.

PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO = 2000
# Plafond (en Mo) au-dela duquel garder toute la grille en memoire est juge trop risque : le
# script bascule alors seul sur le ré-entrainement, plutot que de risquer un plantage memoire
# au milieu d'une fenetre apres une heure de calcul. Sans effet si
# GARDER_CANDIDATS_RANDOM_FOREST = False. Mets une valeur plus basse que la RAM libre de ta
# machine (l'estimation ne compte que les forets, pas le panel ni le reste du processus).

N_JOBS_RANDOM_FOREST = -1
# Nb de coeurs utilises pour construire les arbres en parallele (-1 = tous). Reglage
# purement technique : il change le TEMPS de calcul, jamais le resultat -- il n'entre donc
# pas dans la signature de l'experience (journal.py).


# ============================================================
# Portefeuille COMBINE (ensemble de plusieurs modeles) -- notebook 08, partie C
#
# Au lieu de choisir un seul modele, on combine les predictions de plusieurs d'entre eux en
# une prediction unique, puis on en fait un portefeuille long-short par decile exactement
# comme pour les modeles individuels (memes deciles, memes mesures de performance).
#
# L'idee vient de la litterature sur la combinaison de previsions (Bates & Granger 1969 ;
# Granger & Ramanathan 1984 ; Timmermann 2006) : deux modeles qui se trompent sur des
# choses DIFFERENTES se completent, et leur moyenne a une erreur plus faible que chacun
# pris isolement -- meme quand l'un des deux est nettement moins bon. C'est le meme
# argument que la diversification d'un portefeuille, applique aux previsions.
#
# ⚠️ La combinaison se fait sur les PREDICTIONS deja sauvegardees (outputs/predictions_*.parquet),
# jamais en ré-entrainant quoi que ce soit : le notebook 08 reste un notebook d'evaluation.
# ============================================================
MODELES_ENSEMBLE = ['LightGBM', 'Random Forest','Regression lineaire', 'Elastic Net']
# Les modeles a combiner (2 ou plus), avec EXACTEMENT les noms utilises par les scripts :
# 'Regression lineaire', 'Elastic Net', 'LightGBM', 'Random Forest'.
# ⚠️ Leurs predictions doivent etre sur le disque, donc les scripts correspondants doivent
# avoir tourne (avec les memes parametres generaux, sinon les fenetres ne coincident pas).

METHODE_PONDERATION_ENSEMBLE = 'moindres_carres'
# Comment les poids de chaque modele sont determines. Cinq methodes, de la plus simple a la
# plus adaptative :
#
#   'manuelle'          : les poids que TU fixes dans POIDS_ENSEMBLE ci-dessous. Constants
#                         sur toute la periode. A utiliser pour tester une intuition
#                         precise (ex: 70% LightGBM / 30% Random Forest).
#
#   'egale'             : 1/N pour chaque modele, constant. C'est la reference a battre :
#                         la litterature montre qu'elle est etonnamment difficile a
#                         depasser ("forecast combination puzzle", Smith & Wallis 2009),
#                         parce que les poids estimes sont eux-memes bruites.
#
#   'r2_validation'     : poids proportionnels au R2_oos de VALIDATION (pooled) de chaque
#                         modele, lu dans outputs/resultats_*.parquet. Constants aussi, mais
#                         determines par les donnees et non a la main. Un modele dont le
#                         R2_oos de validation est negatif recoit un poids nul.
#                         ⚠️ Aucune fuite : la validation n'a jamais servi de test.
#
#   'inverse_variance'  : poids proportionnels a 1 / erreur quadratique moyenne de chaque
#                         modele (Bates & Granger 1969), recalcules CHAQUE MOIS a partir des
#                         mois de test DEJA PASSES uniquement. Poids variables dans le temps :
#                         un modele qui se degrade perd automatiquement du poids.
#
#   'moindres_carres'   : les poids qui minimisent l'erreur quadratique de la combinaison,
#                         estimes par regression du rendement realise sur les predictions des
#                         modeles (Granger & Ramanathan 1984, ce qu'on appelle aujourd'hui du
#                         "stacking"). Comme au-dessus, estimes CHAQUE MOIS sur les mois de
#                         test deja passes seulement -- jamais sur le mois qu'on predit, ni
#                         sur le futur. C'est la methode la plus puissante des cinq, et la
#                         plus exposee au bruit d'estimation : d'ou l'option
#                         POIDS_ENSEMBLE_POSITIFS ci-dessous.
#
# ⚠️ Les deux dernieres methodes sont "glissantes" : les tout premiers mois de test n'ont
# pas assez d'historique pour estimer quoi que ce soit et retombent sur des poids egaux
# (voir MOIS_MINIMUM_PONDERATION_ENSEMBLE).

METHODES_ENSEMBLE_COMPAREES = ('egale', 'r2_validation',
                               'inverse_variance', 'moindres_carres')
# Les regles de ponderation rejouees cote a cote en partie C du notebook 08, a composition
# de modeles FIXE. Analyse de sensibilite : aucune n'est "choisie", et seule
# METHODE_PONDERATION_ENSEMBLE ci-dessus alimente le journal des experiences.

POIDS_ENSEMBLE = {'LightGBM': 0.6, 'Random Forest': 0.4}
# Utilise UNIQUEMENT si METHODE_PONDERATION_ENSEMBLE = 'manuelle'. Une entree par modele de
# MODELES_ENSEMBLE. Les poids sont renormalises pour sommer a 1 (tu peux donc ecrire 60/40
# ou 0.6/0.4 indifferemment). Des poids negatifs sont acceptes (position vendeuse sur les
# previsions d'un modele), mais rarement une bonne idee.

FENETRE_PONDERATION_ENSEMBLE_MOIS = 60
# Pour 'inverse_variance' et 'moindres_carres' : nb de mois de test PASSES utilises pour
# estimer les poids du mois courant. 60 = 5 ans glissants. Mettre None (ou 0) pour utiliser
# TOUT l'historique passe disponible (fenetre extensive) : poids plus stables, mais plus
# lents a reagir a un changement de regime.

MOIS_MINIMUM_PONDERATION_ENSEMBLE = 24
# Nb de mois d'historique minimum avant d'estimer des poids. En dessous, la combinaison
# utilise des poids EGAUX (repli neutre) plutot que des poids estimes sur trop peu de
# donnees. Sans effet sur les methodes a poids constants.

POIDS_ENSEMBLE_POSITIFS = True
# Pour 'moindres_carres' : contraindre les poids a etre >= 0 et a sommer a 1. Fortement
# recommande. Sans contrainte, la regression produit reguliererement des poids du type
# +2.4 / -1.4 qui collent parfaitement au passe et se comportent tres mal ensuite -- c'est
# le resultat classique de la litterature (Timmermann 2006). Mettre False donne la version
# non contrainte de Granger & Ramanathan (1984), utile surtout a titre de comparaison.


NOM_MODELE_ENSEMBLE = 'Ensemble'
# Nom sous lequel le portefeuille combine apparait dans les tableaux, les graphiques, le
# journal des experiences et l'historique des performances de portefeuille.


# ============================================================
# Portefeuilles long-short par decile (notebook 07, partie B ; consomme les
# predictions deja sauvegardees par 04/05/06 sans rien ré-entrainer)
# ============================================================
NB_DECILES = 10  # decile 1 = predictions les plus faibles, decile NB_DECILES = les plus elevees


# ============================================================
# Evaluation factorielle : FF5 + momentum (notebook 08 partie D ; voir facteurs.py)
#
# Deux questions DISTINCTES, et il faut les deux dans le memoire :
#   1. ALPHA      -- le long-short rapporte-t-il ce que les facteurs n'expliquent pas ?
#                    Regression r_LS,t = alpha + b' F_t + e_t.
#   2. COMPARAISON -- le long-short fait-il MIEUX qu'un portefeuille construit avec les
#                    facteurs eux-memes ? Sharpe contre Sharpe, meme tableau.
# Un alpha positif significatif ET un Sharpe inferieur a celui du momentum seul est un
# resultat coherent, pas une contradiction : le premier parle de rendement ORTHOGONAL, le
# second de rendement TOTAL.
#
# ⚠️ Ces parametres ne changent aucune prediction, donc aucune cle d'experience : ils ne
# sont volontairement PAS enregistres au journal, exactement comme ceux de l'etape 10.
# ============================================================
LAGS_NEWEY_WEST_ALPHA = 3
# Nb de retards Newey-West des regressions d'alpha. 3 = convention usuelle sur des
# rendements mensuels non chevauchants (piste 1 mois).
# ⚠️ Sur la piste 12 mois, les cohortes de Jegadeesh-Titman partagent 11 douziemes de leurs
# positions d'un mois sur l'autre : le notebook 11 doit passer HORIZON_PREDICTION_MOIS - 1,
# pas cette valeur. `facteurs.lags_par_defaut(horizon)` le fait automatiquement.

MODELE_FACTORIEL_REFERENCE = 'FF5+MOM'
# Le modele dont l'alpha sert de reference dans les tableaux. Les autres cles disponibles
# ('CAPM', 'FF3', 'FF5') restent calculees : le tableau emboite montre COMBIEN d'alpha
# chaque bloc de facteurs absorbe, et notamment ce que MOM enleve.

METHODE_BENCHMARK_FACTORIEL = 'equipondere'
# Comment fabriquer UN portefeuille a partir des 6 facteurs, pour la comparaison Sharpe :
#   'equipondere'      -- 1/N sur les 6 facteurs. Aucun parametre estime, donc aucun risque
#                         de look-ahead. C'est le repere honnete par defaut (DeMiguel,
#                         Garlappi & Uppal 2009).
#   'variance_inverse' -- poids inversement proportionnels a la variance des facteurs,
#                         estimee sur les FENETRE_BENCHMARK_FACTORIEL_MOIS mois PASSES.
#                         Meme logique glissante que ensemble.py.
# ⚠️ Volontairement PAS de portefeuille tangent estime sur toute la periode : il utiliserait
# les rendements futurs pour choisir ses poids, et battrait n'importe quel modele par pure
# construction. Ce serait le repere le plus flatteur et le moins defendable.

FENETRE_BENCHMARK_FACTORIEL_MOIS = 60
MOIS_MINIMUM_BENCHMARK_FACTORIEL = 24
# Pour 'variance_inverse' uniquement. Memes valeurs et meme esprit que les parametres
# homonymes de l'ensemble : 60 mois glissants, repli sur des poids egaux en dessous de 24.

NOM_BENCHMARK_FACTORIEL = 'FF5+MOM (1/N)'
# Nom sous lequel le portefeuille de facteurs apparait dans le tableau de comparaison.


# ------------------------------------------------------------
# Turnover et couts (notebook 08 partie D ; voir couts.py)
#
# DEUX couts, qui ne se confondent pas et ne se reduisent pas par les memes leviers :
#   TRANSACTION -- se paie quand on BOUGE. Base : le turnover.
#   EMPRUNT     -- se paie quand on RESTE short. Base : le notionnel vendu, chaque mois.
# La zone tampon fait baisser le premier en allongeant la detention, donc elle ALLONGE la
# duree de location de la jambe vendeuse. Un parametre unique melangerait deux effets de
# signe oppose.
# ------------------------------------------------------------
PONDERATIONS_EVALUEES = ['equipondere', 'capitalisation']
# Quelles constructions passent dans la partie D. L'equipondere est la reference du projet ;
# la ponderation par capitalisation est le test de robustesse de Gu, Kelly & Xiu (2020), et
# celui qu'attend la critique d'Avramov, Cheng & Metzker (2023).

COUT_TRANSACTION_BPS = {'equipondere': 40.0, 'capitalisation': 15.0}
# Cout UNIDIRECTIONNEL proportionnel, en points de base, PAR SCHEMA DE PONDERATION.
# ⚠️ Valeurs volontairement DIFFERENCIEES. Appliquer le meme chiffre aux deux sous-estime
# systematiquement le handicap de l'equipondere : Novy-Marx & Velikov (2016) montrent que les
# couts effectifs varient d'un ordre de grandeur entre grandes et petites capitalisations, et
# l'equipondere met autant d'argent sur le plus petit titre de l'univers que sur le plus
# gros. Les niveaux retenus sont des ordres de grandeur post-decimalisation pour un univers
# deja filtre de son decile le moins liquide ; ils ne remplacent PAS l'analyse de
# sensibilite, qui est le vrai resultat (voir GRILLE_COUTS_BPS).

COUT_EMPRUNT_ANNUEL_BPS = 30.0
# Loyer annuel de la jambe vendeuse, en points de base. 20 a 40 pour des titres liquides et
# largement detenus (`general collateral`).
# ⚠️ Un taux uniforme sous-estime le cout des titres difficiles a emprunter, et D'Avolio
# (2002) montre que ce sont precisement ceux qui peuplent les deciles d'anomalie extremes. Le
# filtre de liquidite de l'etape 03 attenue le probleme sans le supprimer : a dire en note.

TAMPON_DECILES = 0
# Largeur de la zone tampon, en deciles. 0 reproduit exactement les parties B et C.
# 1 : on entre au decile 10 mais on ne vend qu'en dessous du decile 9 (`buffering` de
# Novy-Marx & Velikov). Le SIGNAL du modele est inchange, seule la regle de sortie s'elargit.

IGNORER_PREMIER_MOIS_TURNOVER = True
# Le premier mois construit les deux jambes a partir de rien : turnover mecanique de 2.0, qui
# ne dit rien de la strategie. La litterature l'ecarte. False repond a une autre question,
# celle du cout de MISE EN PLACE, legitime mais distincte.

GRILLE_COUTS_BPS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
# Grille de l'analyse de sensibilite. Defendre un niveau de cout unique est toujours
# contestable ; montrer la reponse pour tous les niveaux plausibles ne l'est pas.

SEUIL_T_RENTABILITE = 1.96
# t-stat visee par `couts.cout_seuil_significativite`. Le cout qui ramene la t-stat a ce
# seuil est plus bas -- souvent nettement -- que celui qui annule le rendement : retrancher un
# cout proportionnel au turnover ne fait pas que baisser la moyenne, il ajoute de la variance,
# puisque le turnover varie d'un mois sur l'autre.


# ============================================================
# Analyse par segment de TAILLE (etape 10 / notebook 10)
#
# Question posee : OU le modele fonctionne-t-il ? Les memes predictions, deja sauvegardees
# par 04/05/06/07, sont simplement RE-EVALUEES separement sur les petites et les grandes
# capitalisations. AUCUN re-entrainement, AUCUNE modification des etapes 04 a 07 : elles
# continuent d'entrainer et d'evaluer sur l'univers COMPLET, exactement comme avant.
# C'est le protocole de Gu, Kelly & Xiu (2020).
#
# ⚠️ Ces parametres ne changent aucune prediction, donc aucune cle d'experience : ils ne
# sont volontairement PAS enregistres au journal. Les modifier ne cree pas une nouvelle
# experience, il suffit de relancer `python scripts/etape10_analyse_par_taille.py`.
# L'etape 03 n'a PAS a etre relancee : elle ne stocke que la capitalisation brute
# (`mvel1_brut`), et le decoupage en est derive a chaque execution de l'etape 10.
#
# Chemins de sortie : dans chemins.py (TAILLE_*), re-exposes plus bas en FICHIER_TAILLE_*.
# ============================================================
MODE_GROUPES_TAILLE = 'terciles'
# Comment l'univers est decoupe selon la capitalisation boursiere (mvel1) :
#
#   --- Decoupages RELATIFS (percentiles recalcules CHAQUE MOIS) ---
#   'mediane'      -> 2 groupes : 'Small' / 'Large'         (mediane mensuelle)
#   'terciles'     -> 3 groupes : 'Small' / 'Mid' / 'Large'
#   'quintiles'    -> 5 groupes : 'Q1 (small)' ... 'Q5 (large)'
#   'personnalise' -> les percentiles et noms de SEUILS_PERCENTILES_* ci-dessous
#
#   --- Decoupage ABSOLU (seuils en DOLLARS, fixes sur toute la periode) ---
#   'dollars'      -> les seuils et noms de SEUILS_DOLLARS_* ci-dessous
#
# ⚠️ Dans les modes RELATIFS, le decoupage est refait MOIS PAR MOIS (percentiles de la
# population de ce mois-la), et jamais fige une fois pour toutes : une entreprise doit
# pouvoir changer de groupe en grandissant. Un decoupage fige sur la taille moyenne ou
# finale d'une entreprise introduirait une information du futur (on saurait des 1990
# qu'elle deviendra une grande capitalisation).

# --- Mode 'personnalise' : percentiles sur mesure ---
SEUILS_PERCENTILES_GROUPES_TAILLE_PERSONNALISES = [0.3, 0.7]
# Liste CROISSANTE de percentiles de coupure, strictement entre 0 et 1.
# Ex: [0.3, 0.7] = 3 groupes (30% / 40% / 30%).

NOMS_GROUPES_TAILLE_PERSONNALISES = ['Small', 'Mid', 'Large']
# Noms des groupes, du plus PETIT au plus GRAND. Il en faut exactement un de plus que de
# seuils. Mettre None a la place d'un nom EXCLUT la tranche correspondante de l'analyse
# (voir le mode 'dollars' ci-dessous, ou cette possibilite sert le plus souvent).

# --- Mode 'dollars' : seuils absolus, en DOLLARS ---
SEUILS_DOLLARS_GROUPES_TAILLE = [2_000_000_000, 10_000_000_000]
NOMS_GROUPES_TAILLE_DOLLARS = ['Small', None, 'Large']
# Meme logique que les percentiles, mais les seuils sont des MONTANTS EN DOLLARS, compares
# a la capitalisation apres conversion par MULTIPLICATEUR_MVEL1_EN_DOLLARS (ci-dessous).
#
# ℹ️ Le `None` cree une ZONE TAMPON EXCLUE. L'exemple ci-dessus se lit :
#     Small = capitalisation <= 2 Md$      |  Large = capitalisation > 10 Md$
#     les titres entre 2 et 10 Md$ sont exclus de l'analyse par taille
# C'est ce qu'il faut pour comparer deux groupes NETTEMENT separes plutot que deux moities
# contigues. Pour un simple seuil unique sans zone tampon (Small <= 2 Md$, Large > 2 Md$) :
#     SEUILS_DOLLARS_GROUPES_TAILLE = [2_000_000_000]
#     NOMS_GROUPES_TAILLE_DOLLARS   = ['Small', 'Large']
#
# ⚠️⚠️ AVERTISSEMENT METHODOLOGIQUE, A LIRE AVANT D'UTILISER CE MODE.
# Un seuil en dollars FIXE sur 40 ans n'est pas comparable d'un bout a l'autre de
# l'echantillon : sous l'effet de l'inflation et de la croissance des marches, "2 milliards
# de dollars" designe une tres grande entreprise en 1980 et une entreprise moyenne en 2020.
# Le groupe 'Large' sera donc quasi VIDE en debut de periode et absorbera une part
# croissante de l'univers ensuite. Les effectifs mensuels du notebook 10 (section 2) rendent
# cette derive tres visible : REGARDE-LES avant de conclure quoi que ce soit.
# Consequences a assumer :
#   - un groupe peut compter trop peu de titres certains mois pour former NB_DECILES
#     deciles ; ces mois-la sont ALORS EXCLUS du portefeuille de ce groupe (l'etape 10
#     affiche combien, colonne n_mois_sans_portefeuille) ;
#   - la comparaison Small/Large melange un effet de TAILLE et un effet de PERIODE, puisque
#     les deux groupes ne sont pas peuples aux memes dates.
# Les modes relatifs n'ont aucun de ces defauts. Utilise 'dollars' pour ILLUSTRER un propos
# avec des montants concrets, et garde un mode relatif comme analyse principale.

MULTIPLICATEUR_MVEL1_EN_DOLLARS = 1000.0
# ⚠️ A VERIFIER SUR TES DONNEES AVANT DE CITER UN CHIFFRE DANS LE MEMOIRE, et a plus forte
# raison avant de fixer des seuils en mode 'dollars' : c'est ce facteur qui determine a quoi
# ils correspondent reellement.
# mvel1 (datashare de Gu, Kelly & Xiu) est la capitalisation telle que fournie par CRSP,
# c'est-a-dire |prc| x shrout, ou shrout est en MILLIERS d'actions : mvel1 est donc en
# MILLIERS de dollars, d'ou le 1000.0 pour convertir en dollars.
# Le notebook 10 (section 1bis) affiche la colonne brute A COTE de la colonne convertie,
# pour que tu puisses recouper avec une entreprise dont tu connais la capitalisation a une
# date donnee. Si l'echelle ne colle pas, corrige ce multiplicateur (1.0 si mvel1 est deja
# en dollars, 1e6 s'il est en millions) et relance SEULEMENT l'etape 10 : la colonne stockee
# dans le panel reste brute, la conversion n'a lieu qu'ici.

NOM_GROUPE_UNIVERS_COMPLET = 'Univers complet'
# Nom de la ligne de reference (tout l'univers, sans decoupage) affichee a cote des groupes
# dans tous les tableaux de l'etape 10 : c'est elle qui reproduit exactement les chiffres du
# notebook 08, et donc le point de comparaison de chaque groupe.

COLONNE_MVEL1_BRUT = 'mvel1_brut'
# Colonne ECRITE PAR L'ETAPE 03 (partie B, section B.2bis) dans le panel : la capitalisation
# AVANT winsorizing et rank transform (apres quoi mvel1 vit dans [-1, 1] et n'est plus
# interpretable en dollars). C'est la SEULE chose que l'etape 03 conserve pour l'etape 10.
# ⚠️ Ce n'est JAMAIS un predicteur : les etapes 04 a 07 selectionnent explicitement
# PREDICTEURS, donc elles l'ignorent.

COLONNE_GROUPE_TAILLE = 'groupe_taille'
# Colonne CALCULEE PAR L'ETAPE 10 (elle n'existe pas dans le panel : c'est une derivation de
# la capitalisation, recalculee a chaque execution). NaN = zone tampon, c'est-a-dire une
# tranche exclue de l'analyse (nom valant None ci-dessus).


# ============================================================
# HORIZON DE PREDICTION LONG (etapes 11 a 14 / notebook 11)
#
# Piste PARALLELE au pipeline principal : au lieu de predire le rendement excedentaire du
# mois suivant, on predit celui des HORIZON_PREDICTION_MOIS mois suivants.
#
# ⚠️ Elle S'AJOUTE, elle ne remplace rien. Les scripts etape04 a etape07 et les notebooks
# 04 a 10 ne sont NI modifies NI relances : ils continuent de tourner sur `excess_return`
# (horizon 1 mois), et leurs fichiers de sortie ne sont jamais ecrases (les sorties de
# l'horizon long sont suffixees, voir fichiers_horizon() plus haut).
#
# Les scripts etape11 a etape14 reutilisent les fonctions `entrainer()` des scripts 04 a 07
# telles quelles : ils basculent simplement CIBLE et l'embargo le temps de leur execution
# (horizon.activer_mode_horizon), puis ecrivent dans leurs propres fichiers.
# ============================================================
HORIZON_PREDICTION_MOIS = 12
# Nombre de mois de l'horizon de prediction. 12 = rendement composé sur l'annee suivante.
# Parametrable (6, 24...) : rien n'est code en dur, ni dans les calculs ni dans les noms de
# fichiers. La valeur 1 n'a pas de sens ici -- c'est deja ce que font les etapes 04 a 07.

# La cible composee, telle qu'elle sera nommee dans le panel et dans les fichiers :
#     cible_{i,t} = PROD(1 + R_{i,t+k}) - PROD(1 + Rf_{t+k}),  k = 1..HORIZON
# ⚠️ C'est un rendement EXCEDENTAIRE (rendements et taux sans risque capitalises
# SEPAREMENT puis soustraits), pas un rendement total : c'est le seul choix coherent avec
# la cible d'origine `excess_return` = RET - Rfree. Utiliser PROD(1+R) - 1 changerait
# d'horizon ET de definition simultanement, et les deux pistes ne seraient plus comparables.


def nom_cible_horizon(horizon=None):
    """Nom de la colonne cible pour un horizon donne (ex: 'excess_return_12m')."""
    return f"excess_return_{horizon or HORIZON_PREDICTION_MOIS}m"


TRAITEMENT_RADIATION = 'taux_sans_risque'
# Que faire quand l'historique d'un titre s'arrete AVANT la fin du panel (faillite, rachat,
# radiation) et que l'horizon deborde ? La grille du titre est alors PROLONGEE de HORIZON
# mois (voir horizon.py, section A.1), et ce parametre decide du rendement attribue a ces
# mois FANTOMES -- et rien d'autre. Les mois REELLEMENT observes avant la radiation, eux,
# sont toujours conserves tels quels dans le produit compose, dans les deux conventions :
# la chute des derniers mois de cotation n'est jamais effacee.
#
#   'taux_sans_risque' -> RET = Rf sur les mois fantomes. Liquidation au dernier rendement
#                         observe, puis placement du solde AU TAUX SANS RISQUE jusqu'a
#                         t+HORIZON. Convention de Shumway (1997), standard dans la
#                         litterature. Pour une date de formation entierement posterieure a
#                         la radiation, la cible vaut alors
#                              PROD(1+Rf) - PROD(1+Rf) = 0
#                         soit un rendement EXCEDENTAIRE exactement nul : l'investisseur
#                         gagne precisement le taux sans risque, ni plus ni moins.
#                         -> la radiation est un evenement NEUTRE.
#
#   'zero'             -> RET = 0 sur les mois fantomes. Le solde de liquidation dort en
#                         caisse sans rien rapporter. Meme date de formation :
#                              PROD(1+0) - PROD(1+Rf) = 1 - PROD(1+Rf) < 0
#                         soit MOINS le taux sans risque compose sur l'horizon (de l'ordre
#                         de -5 % sur 12 mois). L'investisseur assume le cout d'opportunite.
#                         -> la radiation devient un evenement NEGATIF.
#
# Quelle difference concretement : sous 'zero', le modele apprend plus fortement a fuir les
# titres au bord de la faillite, et le decile 1 des portefeuilles devrait se creuser. C'est
# une convention plus CONSERVATRICE que Shumway, et c'est exactement le genre de sensibilite
# qui a sa place dans un tableau de robustesse -- les deux valeurs produisent deux
# experiences distinctes au journal (voir journal.py, `traitement_radiation`).
#
# ⚠️ Changer cette valeur change le PANEL, pas seulement l'entrainement : il faut relancer
# `python scripts/construction_panel.py` PUIS les etapes 11 a 14.
#
# ⚠️ Limite commune aux deux conventions : toutes les observations dont la fenetre est
# entierement posterieure a la radiation recoivent, pour un mois donne, la MEME valeur de
# cible (0, ou -Rf compose). Cela cree un petit point de masse dans la distribution, sans
# aucune variation transversale : le modele n'a rien a y apprendre. C'est inherent a la
# methode, pas un defaut d'implementation, mais cela se mentionne.
#
# ℹ️ Une troisieme convention, 'ecarter' (supprimer purement et simplement ces
# observations), n'est VOLONTAIREMENT pas proposee : la disparition d'une entreprise est
# massivement CORRELEE a sa performance, donc l'ecarter reviendrait a supprimer precisement
# les observations ou le rendement a HORIZON mois est le plus negatif. Le modele ne serait
# entraine et evalue que sur des titres dont on sait retrospectivement qu'ils ont survecu :
# une information du futur, et un BIAIS DE SURVIE severe.

CONSTRUCTIONS_PORTEFEUILLE_HORIZON = ['mensuel', 'cohortes', 'annuel']
# Les trois constructions connues. Liste FIGEE : ne pas modifier a la main, c'est
# MODE_PORTEFEUILLE_HORIZON juste en dessous qui choisit lesquelles sont calculees.
#
#   'mensuel'   -> REBALANCEMENT MENSUEL, exactement comme le notebook 08 (horizon 1 mois) :
#                  chaque mois on classe les titres en deciles selon la prediction a HORIZON
#                  mois, on detient UN mois, et le rendement realise utilise est le
#                  rendement excedentaire MENSUEL. Serie mensuelle non chevauchante,
#                  annualisable normalement (x12 et racine(12)).
#                  ⚠️ Seul le SIGNAL DE CLASSEMENT change par rapport au notebook 08 -- la
#                  mecanique du portefeuille est identique. C'est donc la seule construction
#                  qui permette une comparaison 1 mois / HORIZON mois toutes choses egales
#                  par ailleurs : une prevision a HORIZON mois classe-t-elle mieux les
#                  titres qu'une prevision a 1 mois, pour une strategie rebalancee tous les
#                  mois ? C'est la construction PRINCIPALE de cette piste.
#   'cohortes'  -> portefeuilles chevauchants de Jegadeesh-Titman : chaque mois on forme une
#                  cohorte detenue HORIZON mois, et le rendement du mois t est la moyenne
#                  des HORIZON cohortes actives, calcule sur les rendements MENSUELS.
#                  Serie mensuelle, annualisable normalement. Reference en momentum, et la
#                  seule des trois qui respecte l'horizon de detention de la cible.
#   'annuel'    -> rebalancement une fois par an : simple et sans chevauchement, mais
#                  HORIZON fois moins d'observations. Controle de robustesse.
#
# ⚠️ Ce qu'il ne faut SURTOUT pas faire : moyenner LA CIBLE LONGUE par decile et par mois
# puis annualiser en x12 et racine(12). Ces "rendements mensuels" seraient en realite des
# rendements sur HORIZON mois qui SE CHEVAUCHENT (deux mois consecutifs partagent HORIZON-1
# mois de rendement) : l'annualisation serait fausse et la volatilite massivement
# sous-estimee. Les trois constructions ci-dessus evitent ce piege, chacune a sa facon --
# 'mensuel' et 'cohortes' en consommant le rendement MENSUEL realise, 'annuel' en espacant
# les observations de HORIZON mois.

MODE_PORTEFEUILLE_HORIZON = 'tous'
# Lesquelles calculer au notebook 11 : 'tous' (defaut, les trois cote a cote), ou le nom
# d'une seule d'entre elles.

CONSTRUCTION_PRINCIPALE_HORIZON = 'mensuel'
# Celle qui sert de reference partout ailleurs dans le notebook 11 : richesse cumulee,
# constructions alternatives, et surtout comparaison avec l'horizon 1 mois (section 4).
# Les deux autres restent affichees a cote d'elle dans le tableau de la section 3, mais ne
# pilotent aucune autre cellule.
# ⚠️ Doit figurer dans les constructions calculees (voir MODE_PORTEFEUILLE_HORIZON).


def constructions_portefeuille_horizon(mode=None):
    """Liste des constructions a calculer, d'apres MODE_PORTEFEUILLE_HORIZON.

    Verifie au passage que la construction PRINCIPALE en fait partie : sans elle, la moitie
    du notebook 11 n'aurait plus de reference a afficher.
    """
    mode = mode or MODE_PORTEFEUILLE_HORIZON
    if mode == 'tous':
        demandees = list(CONSTRUCTIONS_PORTEFEUILLE_HORIZON)
    elif mode in CONSTRUCTIONS_PORTEFEUILLE_HORIZON:
        demandees = [mode]
    else:
        raise ValueError(
            f"MODE_PORTEFEUILLE_HORIZON = {mode!r} inconnu. Valeurs acceptees : 'tous', "
            + ", ".join(repr(c) for c in CONSTRUCTIONS_PORTEFEUILLE_HORIZON) + "."
        )
    if CONSTRUCTION_PRINCIPALE_HORIZON not in demandees:
        raise ValueError(
            f"CONSTRUCTION_PRINCIPALE_HORIZON = {CONSTRUCTION_PRINCIPALE_HORIZON!r} n'est pas "
            f"calculee avec MODE_PORTEFEUILLE_HORIZON = {mode!r}. Mets le mode a 'tous', ou "
            f"fais des deux parametres la meme construction."
        )
    return demandees


# ============================================================
# CONSTRUCTIONS ALTERNATIVES DE PORTEFEUILLE (notebooks 08 et 11)
#
# La construction de REFERENCE du projet ne change pas : long-short EQUIPONDERE, decile le
# plus haut moins decile le plus bas, recalcule mois par mois. C'est elle qui alimente le
# tableau de performance, le journal des experiences et le notebook 09 -- rien de ce qui
# suit ne la remplace.
#
# Les variantes ci-dessous sont calculees EN PLUS, cote a cote, dans une section dediee, et
# repondent a une seule question : de combien le ratio de Sharpe bouge-t-il quand on change
# la facon de construire le portefeuille, a predictions strictement identiques ? Aucune ne
# demande de ré-entrainer quoi que ce soit : elles rejouent les predictions deja sur disque.
# ============================================================
PCT_LONG_ONLY = 0.20
# Fraction des titres achetes par le portefeuille LONG ONLY : chaque mois, on garde les
# PCT_LONG_ONLY x 100 % des titres les mieux notes par le modele, et on les detient sans
# aucune vente a decouvert. 0.20 = les 20 % du haut (soit les deux deciles superieurs).
#
# ℹ️ Pourquoi cette variante compte pour le memoire : la plupart des investisseurs
# institutionnels ne peuvent PAS vendre a decouvert. Un long-short brillant dont toute la
# performance vient de la jambe courte n'est alors pas implementable. ⚠️ Attention a la
# lecture : ce portefeuille contient l'exposition au MARCHE (beta ≈ 1), que le long-short
# neutralise en grande partie. Son Sharpe doit donc etre compare a celui du marche, pas a
# celui du long-short.
# Valeurs raisonnables : 0.05 (tres concentre) a 0.30. En dessous de 0.02, le portefeuille
# devient trop concentre pour que la moyenne mensuelle veuille encore dire grand-chose.

PONDERATION_PAR_CAPITALISATION = True
# Calcule EN PLUS, dans la section des constructions alternatives, les memes portefeuilles
# ponderes par la CAPITALISATION BOURSIERE (colonne COLONNE_MVEL1_BRUT du panel, connue en t
# donc sans information du futur) au lieu d'etre equiponderes.
#
# ⚠️ C'est LE test de robustesse attendu par un jury, et celui de Gu, Kelly & Xiu (2020).
# Un portefeuille equipondere met autant d'argent sur une micro-cap illiquide que sur Apple ;
# or c'est precisement chez les petites capitalisations que les modeles d'apprentissage
# trouvent l'essentiel de leur signal. La ponderation par capitalisation reflete ou l'argent
# peut REELLEMENT etre investi, et le Sharpe s'effondre generalement au passage. Si c'est le
# cas chez toi, ce n'est pas un echec : c'est un resultat, et il se rapproche du notebook 10
# (analyse par taille).
# Mettre False pour sauter ce calcul (il exige de relire la capitalisation dans le panel).

# ============================================================
# COMPATIBILITE -- anciens noms de chemins (FICHIER_*)
#
# Ces alias sont GENERES a partir de chemins.py, jamais ecrits a la main : impossible qu'un
# nom de fichier diverge entre l'ancien et le nouveau code. Ils existent uniquement pour que
# les notebooks 01 a 11 (qui ecrivent `config.FICHIER_PREDICTIONS_LIGHTGBM`) tournent sans
# modification.
#
# ℹ️ Dans du code NEUF, prefere la forme fonctionnelle, qui ne demande rien a ajouter ici
# quand tu ajoutes un modele :
#     chemins.predictions('lightgbm')            au lieu de FICHIER_PREDICTIONS_LIGHTGBM
#     chemins.resultats('lightgbm', horizon=12)  au lieu de fichiers_horizon(...)['resultats']
# ============================================================

FICHIER_CARACTERISTIQUES_BRUT = chemins.CARACTERISTIQUES_BRUT
FICHIER_RETURNS_BRUT = chemins.RETURNS_BRUT
FICHIER_MACRO_BRUT = chemins.MACRO_BRUT

FICHIER_CARACTERISTIQUES_CLEAN = chemins.CARACTERISTIQUES_CLEAN
FICHIER_RETURNS_CLEAN = chemins.RETURNS_CLEAN
FICHIER_MACRO_CLEAN = chemins.MACRO_CLEAN
FICHIER_CARACTERISTIQUES_RETENUES = chemins.CARACTERISTIQUES_RETENUES

FICHIER_PANEL_FINAL = chemins.PANEL_FINAL
FICHIER_PANEL_MODELISATION = chemins.PANEL_MODELISATION

# Un bloc de 5 alias par modele, engendre par boucle : aucune ligne a ecrire pour un 5e.
for _cle, _SUFFIXE in [
    ('regression_lineaire', 'REGRESSION_LINEAIRE'),
    ('elastic_net', 'ELASTIC_NET'),
    ('lightgbm', 'LIGHTGBM'),
    ('random_forest', 'RANDOM_FOREST'),
]:
    globals()[f'FICHIER_MODELE_{_SUFFIXE}'] = chemins.modele(_cle)
    globals()[f'FICHIER_PREDICTIONS_{_SUFFIXE}'] = chemins.predictions(_cle)
    globals()[f'FICHIER_RESULTATS_{_SUFFIXE}'] = chemins.resultats(_cle)
    globals()[f'FICHIER_RESULTATS_{_SUFFIXE}_PAR_FENETRE'] = chemins.resultats(_cle, par_fenetre=True)
del _cle, _SUFFIXE

FICHIER_SIGNIFICATIVITE_REGRESSION_LINEAIRE = chemins.importance('regression_lineaire')
FICHIER_IMPORTANCE_ELASTIC_NET = chemins.importance('elastic_net')
FICHIER_IMPORTANCE_LIGHTGBM = chemins.importance('lightgbm')
FICHIER_IMPORTANCE_RANDOM_FOREST = chemins.importance('random_forest')

FICHIER_JOURNAL_EXPERIENCES = chemins.JOURNAL_EXPERIENCES
FICHIER_RENDEMENTS_PORTEFEUILLES = chemins.RENDEMENTS_PORTEFEUILLES
FICHIER_PERFORMANCE_PORTEFEUILLES = chemins.PERFORMANCE_PORTEFEUILLES
FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES = chemins.HISTORIQUE_PERFORMANCE_PORTEFEUILLES
FICHIER_PERFORMANCE_CONSTRUCTIONS = chemins.PERFORMANCE_CONSTRUCTIONS
FICHIER_PREDICTIONS_ENSEMBLE = chemins.PREDICTIONS_ENSEMBLE
FICHIER_POIDS_ENSEMBLE = chemins.POIDS_ENSEMBLE
FICHIER_POIDS_ENSEMBLE_METHODES = chemins.POIDS_ENSEMBLE_METHODES
FICHIER_PERFORMANCE_ENSEMBLES_METHODES = chemins.PERFORMANCE_ENSEMBLES_METHODES
FICHIER_ENSEMBLES_POIDS_PNG = chemins.ENSEMBLES_POIDS_PNG
FICHIER_ENSEMBLES_CUMULATIF_PNG = chemins.ENSEMBLES_CUMULATIF_PNG

FICHIER_COMPARAISON_PARQUET = chemins.COMPARAISON_PARQUET
FICHIER_COMPARAISON_PNG = chemins.COMPARAISON_PNG
FICHIER_EVOLUTION_R2_PNG = chemins.EVOLUTION_R2_PNG
FICHIER_CUMULATIF_PNG = chemins.CUMULATIF_PNG
FICHIER_DECILES_PNG = chemins.DECILES_PNG
FICHIER_CONSTRUCTIONS_PNG = chemins.CONSTRUCTIONS_PNG

FICHIER_TAILLE_DESCRIPTIF = chemins.TAILLE_DESCRIPTIF
FICHIER_TAILLE_BORNES_MENSUELLES = chemins.TAILLE_BORNES_MENSUELLES
FICHIER_TAILLE_PERFORMANCE = chemins.TAILLE_PERFORMANCE
FICHIER_TAILLE_DECILES = chemins.TAILLE_DECILES
FICHIER_TAILLE_RENDEMENTS_LS = chemins.TAILLE_RENDEMENTS_LS
FICHIER_TAILLE_IC_MENSUEL = chemins.TAILLE_IC_MENSUEL
FICHIER_TAILLE_R2_PNG = chemins.TAILLE_R2_PNG
FICHIER_TAILLE_DECILES_PNG = chemins.TAILLE_DECILES_PNG
FICHIER_TAILLE_CUMULATIF_PNG = chemins.TAILLE_CUMULATIF_PNG
FICHIER_TAILLE_SEUILS_PNG = chemins.TAILLE_SEUILS_PNG
FICHIER_BENCHMARKS = chemins.BENCHMARKS
FICHIER_BENCHMARKS_PNG = chemins.BENCHMARKS_PNG

FICHIER_FACTEURS_FF5_BRUT = chemins.FACTEURS_FF5_BRUT
FICHIER_FACTEURS_MOM_BRUT = chemins.FACTEURS_MOM_BRUT
FICHIER_FACTEURS_CLEAN = chemins.FACTEURS_CLEAN
FICHIER_ALPHAS_PORTEFEUILLES = chemins.ALPHAS_PORTEFEUILLES
FICHIER_COMPARAISON_FACTEURS = chemins.COMPARAISON_FACTEURS
FICHIER_PERFORMANCE_FACTEURS = chemins.PERFORMANCE_FACTEURS
FICHIER_RENDEMENTS_BENCHMARK_FACTORIEL = chemins.RENDEMENTS_BENCHMARK_FACTORIEL
FICHIER_ALPHAS_PNG = chemins.ALPHAS_PNG
FICHIER_FACTEURS_CUMULATIF_PNG = chemins.FACTEURS_CUMULATIF_PNG
FICHIER_TURNOVER_PORTEFEUILLES = chemins.TURNOVER_PORTEFEUILLES
FICHIER_COUTS_PORTEFEUILLES = chemins.COUTS_PORTEFEUILLES
FICHIER_SEUILS_RENTABILITE = chemins.SEUILS_RENTABILITE
FICHIER_SENSIBILITE_COUTS = chemins.SENSIBILITE_COUTS
FICHIER_SENSIBILITE_COUTS_PNG = chemins.SENSIBILITE_COUTS_PNG

def fichiers_horizon(cle_modele, horizon=None):
    """Chemins de sortie d'un modele pour l'horizon long. Conserve pour le notebook 11.

    Equivaut a `chemins.fichiers_modele(cle_modele, horizon)` -- meme dict, memes chemins.
    """
    return chemins.fichiers_modele(cle_modele, horizon or HORIZON_PREDICTION_MOIS)
