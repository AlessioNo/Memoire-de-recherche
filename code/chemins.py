"""
Chemins de tous les fichiers du projet -- la seule chose que ce module contient.

Pourquoi ce fichier existe
--------------------------
config.py contenait jusqu'ici une cinquantaine de constantes FICHIER_* ecrites une par une
(FICHIER_PREDICTIONS_LIGHTGBM, FICHIER_RESULTATS_LIGHTGBM_PAR_FENETRE...). Ajouter un
cinquieme modele demandait d'en ecrire quatre de plus, et l'horizon long avait sa propre
fonction a cote. Tout cela suit en realite UNE regle unique :

    outputs/<role>_<cle_modele><suffixe_horizon>.parquet

Ce module l'ecrit une fois. `config.py` importe ce qui suit et re-expose les anciens noms
FICHIER_* tels quels, pour que les notebooks 01 a 11 continuent de fonctionner sans la
moindre modification.

⚠️ Les noms de fichiers produits sont EXACTEMENT ceux d'avant le refactor, y compris les
petites incoherences historiques (extension .joblib a 1 mois, .pkl et prefixe "modele_" a
12 mois) : les notebooks les relisent, et changer un nom aurait casse leurs lectures. Voir
`modele()` ci-dessous.

Aucune dependance : ce module n'importe ni config ni rien d'autre du projet, ce qui evite
tout import circulaire (config -> chemins, jamais l'inverse).
"""

from pathlib import Path

# ------------------------------------------------------------
# Dossiers, tous derives de l'emplacement de ce fichier (racine du projet) : le code
# fonctionne quel que soit le repertoire de travail courant depuis lequel on le lance.
# ------------------------------------------------------------
RACINE = Path(__file__).resolve().parent

DATA_RAW = RACINE / "data" / "raw"
DATA_INTERIM = RACINE / "data" / "interim"
DATA_PROCESSED = RACINE / "data" / "processed"
MODELES_DIR = RACINE / "modeles"
OUTPUTS_DIR = RACINE / "outputs"
RAPPORTS_DIR = OUTPUTS_DIR / "rapports"

DOSSIERS_A_CREER = [DATA_INTERIM, DATA_PROCESSED, MODELES_DIR, OUTPUTS_DIR]


def assurer_dossiers():
    """Cree les dossiers de sortie s'ils n'existent pas (premier lancement, ou clone du
    depot : data/interim, data/processed, modeles/ et outputs/ ne contiennent aucun
    fichier versionne). Sans effet s'ils existent deja."""
    for dossier in DOSSIERS_A_CREER:
        dossier.mkdir(parents=True, exist_ok=True)
    RAPPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Fichiers bruts et intermediaires (etapes 02 et 03)
# ------------------------------------------------------------
CARACTERISTIQUES_BRUT = DATA_RAW / "datashare.parquet"
RETURNS_BRUT = DATA_RAW / "StockReturn.parquet"
MACRO_BRUT = DATA_RAW / "MacroData.parquet"

# Facteurs de risque de la Data Library de Ken French. Deux CSV a dezipper a la main dans
# data/raw/, comme les 3 fichiers ci-dessus (dossier non versionne).
#   F-F_Research_Data_5_Factors_2x3_CSV.zip  -> MKT-RF, SMB, HML, RMW, CMA, RF
#   F-F_Momentum_Factor_CSV.zip              -> MOM
# ⚠️ Ils ne servent JAMAIS a predire : ce sont des series d'EVALUATION (etape 08).
FACTEURS_FF5_BRUT = DATA_RAW / "F-F_Research_Data_5_Factors_2x3.csv"
FACTEURS_MOM_BRUT = DATA_RAW / "F-F_Momentum_Factor.csv"

CARACTERISTIQUES_CLEAN = DATA_INTERIM / "characteristics_clean.parquet"
RETURNS_CLEAN = DATA_INTERIM / "returns_clean.parquet"
MACRO_CLEAN = DATA_INTERIM / "macro_clean.parquet"
FACTEURS_CLEAN = DATA_INTERIM / "facteurs_clean.parquet"
CARACTERISTIQUES_RETENUES = DATA_INTERIM / "caracteristiques_retenues.json"

PANEL_FINAL = DATA_PROCESSED / "panel_final.parquet"
PANEL_MODELISATION = DATA_PROCESSED / "panel_pret_modelisation.parquet"


# ------------------------------------------------------------
# Sorties d'UN modele : les 4 fonctions qui remplacent 24 constantes
#
# `cle` vaut 'regression_lineaire', 'elastic_net', 'lightgbm' ou 'random_forest'
# (voir entrainement/specs.py : c'est la cle canonique, utilisee partout dans le projet).
# `horizon` vaut 1 (piste principale) ou HORIZON_PREDICTION_MOIS (piste longue).
# ------------------------------------------------------------

def _suffixe(horizon):
    """'' a 1 mois, '_h12' a 12 mois. C'est ce suffixe qui garantit que les deux pistes
    cohabitent sur disque sans jamais s'ecraser."""
    return "" if horizon == 1 else f"_h{horizon}"


def modele(cle, horizon=1):
    """Modele entraine de la DERNIERE fenetre (joblib).

    ⚠️ Le nom differe entre les deux pistes pour des raisons purement historiques
    (`lightgbm.joblib` a 1 mois, `modele_lightgbm_h12.pkl` a 12 mois). C'est reproduit a
    l'identique pour ne pas invalider des fichiers deja produits ; les deux sont ecrits
    par `joblib.dump`, seule l'extension du nom change."""
    if horizon == 1:
        return MODELES_DIR / f"{cle}.joblib"
    return MODELES_DIR / f"modele_{cle}_h{horizon}.pkl"


def predictions(cle, horizon=1):
    """Predictions hors-echantillon, toutes fenetres mises bout a bout."""
    return OUTPUTS_DIR / f"predictions_{cle}{_suffixe(horizon)}.parquet"


def resultats(cle, horizon=1, par_fenetre=False):
    """Resume pooled (une ligne) ou detail par fenetre (une ligne par fenetre)."""
    suffixe_role = "_par_fenetre" if par_fenetre else ""
    return OUTPUTS_DIR / f"resultats_{cle}{suffixe_role}{_suffixe(horizon)}.parquet"


def importance(cle):
    """Tableau d'importance / significativite / stabilite des variables.

    ⚠️ La regression lineaire fait exception : son tableau s'appelle
    `significativite_*` (Fama-MacBeth), pas `importance_*` -- il ne contient pas une
    importance mais des t-stats. Nom conserve tel quel."""
    if cle == 'regression_lineaire':
        return OUTPUTS_DIR / "significativite_regression_lineaire.parquet"
    return OUTPUTS_DIR / f"importance_{cle}.parquet"


def fichiers_modele(cle, horizon=1):
    """Les 4 chemins d'un modele d'un coup, sous forme de dict -- c'est ce que consomme
    `entrainement/boucle.py`, et ce que le notebook 11 recoit via `config.fichiers_horizon`."""
    return {
        'modele': modele(cle, horizon),
        'predictions': predictions(cle, horizon),
        'resultats': resultats(cle, horizon),
        'resultats_fenetre': resultats(cle, horizon, par_fenetre=True),
        'importance': importance(cle),
    }


# ------------------------------------------------------------
# Sorties transverses (journal, portefeuilles, figures, analyse par taille)
# ------------------------------------------------------------

def sortie(nom, extension="parquet"):
    """Un fichier quelconque de outputs/, nomme explicitement."""
    return OUTPUTS_DIR / f"{nom}.{extension}"


JOURNAL_EXPERIENCES = sortie("journal_experiences")

RENDEMENTS_PORTEFEUILLES = sortie("rendements_portefeuilles_deciles")
PERFORMANCE_PORTEFEUILLES = sortie("performance_portefeuilles")
HISTORIQUE_PERFORMANCE_PORTEFEUILLES = sortie("historique_performance_portefeuilles")

# Constructions ALTERNATIVES de portefeuille (notebook 08 section B.9, notebook 11 section
# 3bis) : long-short vs long only, equipondere vs pondere par la capitalisation. Elles ne
# remplacent JAMAIS performance_portefeuilles.parquet ci-dessus, qui reste la reference du
# projet et la seule a alimenter le journal des experiences.
PERFORMANCE_CONSTRUCTIONS = sortie("performance_constructions")

PREDICTIONS_ENSEMBLE = sortie("predictions_ensemble")
POIDS_ENSEMBLE = sortie("poids_ensemble_par_mois")
# Comparaison des METHODES de ponderation (notebook 08 partie C). Fichiers SEPARES de
# performance_portefeuilles.parquet : seule la methode de reference
# (config.METHODE_PONDERATION_ENSEMBLE) alimente le journal des experiences.
POIDS_ENSEMBLE_METHODES = sortie("poids_ensemble_par_methode")
PERFORMANCE_ENSEMBLES_METHODES = sortie("performance_ensembles_methodes")
ENSEMBLES_POIDS_PNG = sortie("ensembles_poids_par_methode", "png")
ENSEMBLES_CUMULATIF_PNG = sortie("ensembles_richesse_cumulee", "png")


COMPARAISON_PARQUET = sortie("comparaison_modeles")
COMPARAISON_PNG = sortie("comparaison_modeles", "png")
EVOLUTION_R2_PNG = sortie("evolution_r2_oos_par_fenetre", "png")
CUMULATIF_PNG = sortie("portefeuilles_richesse_cumulee", "png")
DECILES_PNG = sortie("portefeuilles_rendement_par_decile", "png")
CONSTRUCTIONS_PNG = sortie("portefeuilles_constructions_alternatives", "png")

TAILLE_DESCRIPTIF = sortie("analyse_taille_descriptif_groupes")
TAILLE_BORNES_MENSUELLES = sortie("analyse_taille_bornes_mensuelles")
TAILLE_PERFORMANCE = sortie("analyse_taille_performance")
TAILLE_DECILES = sortie("analyse_taille_rendement_par_decile")
TAILLE_RENDEMENTS_LS = sortie("analyse_taille_rendements_long_short")
TAILLE_IC_MENSUEL = sortie("analyse_taille_ic_mensuel")

TAILLE_R2_PNG = sortie("analyse_taille_r2_et_ic", "png")
TAILLE_DECILES_PNG = sortie("analyse_taille_rendement_par_decile", "png")
TAILLE_CUMULATIF_PNG = sortie("analyse_taille_richesse_cumulee", "png")
TAILLE_SEUILS_PNG = sortie("analyse_taille_evolution_capitalisations", "png")

BENCHMARKS = sortie("benchmarks_portefeuilles")
BENCHMARKS_PNG = sortie("portefeuilles_vs_benchmarks", "png")

# Evaluation factorielle (notebook 08 partie D). Fichiers SEPARES de
# performance_portefeuilles.parquet et de l'historique : ils ne dependent d'aucune
# prediction nouvelle et n'alimentent donc PAS le journal des experiences -- ajouter des
# colonnes d'alpha a journal.COLONNES_PORTEFEUILLE rendrait l'historique deja ecrit
# inconcatenable avec les lignes futures.
ALPHAS_PORTEFEUILLES = sortie("alphas_portefeuilles")
COMPARAISON_FACTEURS = sortie("comparaison_modeles_vs_facteurs")
PERFORMANCE_FACTEURS = sortie("performance_facteurs")
RENDEMENTS_BENCHMARK_FACTORIEL = sortie("rendements_benchmark_factoriel")

TURNOVER_PORTEFEUILLES = sortie("turnover_portefeuilles")
COUTS_PORTEFEUILLES = sortie("performance_brute_vs_nette")
SEUILS_RENTABILITE = sortie("couts_seuils_rentabilite")
SENSIBILITE_COUTS = sortie("sensibilite_aux_couts")

ALPHAS_PNG = sortie("portefeuilles_alphas_factoriels", "png")
FACTEURS_CUMULATIF_PNG = sortie("portefeuilles_vs_facteurs_richesse_cumulee", "png")
SENSIBILITE_COUTS_PNG = sortie("portefeuilles_sensibilite_couts", "png")
