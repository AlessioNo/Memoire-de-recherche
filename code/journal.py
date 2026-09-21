"""
Journal des experiences (parametres + resultats), consomme par le notebook 08.

Pourquoi un fichier a part (et pas directement dans config.py) ? Meme logique que
fenetres.py : config.py ne contient que des VALEURS (chemins, parametres) ; celui-ci
contient de la LOGIQUE (des fonctions) reutilisee par les notebooks 04, 05, 06 (pour
ecrire) et 08 (pour lire) -- les regrouper ici evite de copier-coller les memes fonctions
dans 4 notebooks differents.

Le probleme que ce fichier resout : les scripts 04/05/06 ECRASENT leurs fichiers de
resultats (outputs/resultats_*.parquet) a chaque execution -- un seul jeu de resultats a
la fois, celui du dernier lancement (c'est voulu : le notebook 07 doit toujours pouvoir
lire "le" dernier modele entraine, sans ambiguite). Mais ca veut dire qu'on ne peut PAS
comparer plusieurs lancements entre eux (ex: expanding vs rolling, ou deux grilles
d'hyperparametres differentes pour l'Elastic Net) une fois qu'on est passe au suivant --
le premier resultat est perdu.

Ce module ajoute donc un second fichier, outputs/journal_experiences.parquet, qui n'est
JAMAIS ecrase : chaque appel a `enregistrer_experience` (fait par 04/05/06, section 7)
ajoute une ligne, SAUF si une experience strictement identique y figure deja (voir
`enregistrer_experience`), auquel cas rien n'est ajoute -- relancer 2 fois le meme modele
avec exactement les memes parametres ne cree donc jamais de doublon.

Deux familles de parametres sont enregistrees pour chaque experience (voir aussi
config.py, section "Parametres GENERAUX" / "Parametres SPECIFIQUES") :
- Les parametres GENERAUX (`params_generaux_actuels`) : predicteurs, mode de fenetres,
  seuils de filtrage... Ils sont les MEMES pour les 3 modeles, lus directement dans
  config.py au moment de l'appel (donc TOUJOURS ceux actuellement definis dans
  config.py -- si tu changes config.py entre deux notebooks sans relancer le premier, le
  journal reflete ce que CHAQUE notebook a reellement utilise au moment de son execution,
  pas un etat global unique).
- Les parametres SPECIFIQUES (passes explicitement par le notebook appelant, ex: la
  grille d'alpha pour l'Elastic Net) : propres a un seul modele, donc pas dans config.py
  sous une forme commune.

Le notebook 08 regroupe ensuite les lignes du journal par (modele, parametres
specifiques) : un groupe = un tableau, avec une ligne par combinaison de parametres
generaux testee pour ce groupe -- voir `tableaux_par_modele`.

============================================================
Mesures de portefeuille (Sharpe, Sortino...) dans les tableaux du notebook 08
============================================================
Le notebook 07 calcule, a CHAQUE execution, les mesures de performance du portefeuille
long-short des 3 modeles a partir des PREDICTIONS ALORS SUR DISQUE -- et les tague avec
leur `cle_experience` (voir `cle_experience_actuelle` plus bas), la MEME cle que celle deja
ecrite dans `outputs/resultats_*.parquet` par 04/05/06 (section 7 de chaque notebook), et
donc la meme que celle de la ligne correspondante dans `outputs/journal_experiences.parquet`.

Comme pour le journal des experiences, DEUX fichiers de sortie existent pour ces mesures
(voir config.py) :
- `outputs/performance_portefeuilles.parquet` : ECRASE a chaque lancement de 07, ne reflete
  que le DERNIER lancement de chaque modele (utilise par 07 lui-meme, ex: pour son graphique
  de richesse cumulee).
- `outputs/historique_performance_portefeuilles.parquet` : JAMAIS ecrase (comme le journal
  des experiences) -- chaque execution de 07 y AJOUTE les mesures des experiences pas
  encore vues, dedupliquees par `cle_experience` (voir `enregistrer_performances_portefeuilles`),
  sans jamais retoucher aux lignes deja presentes. Relancer 07 plusieurs fois sur les MEMES
  predictions n'ajoute donc rien de plus -- mais un changement de modele/parametres, suivi
  d'un nouveau lancement de 07, ajoute une ligne SANS supprimer celle du lancement precedent.

`tableaux_par_modele` (plus bas) fusionne le journal avec CET HISTORIQUE (et non le simple
snapshot) sur `cle_experience` : chaque tableau du notebook 08 affiche donc, EN PLUS du
`R²_oos` (qui reste le critere de tri), les mesures de portefeuille de TOUTE experience pour
laquelle le notebook 07 a deja ete execute au moins une fois pendant que ses predictions
etaient sur disque -- `NaN` uniquement pour les experiences qui n'ont ENCORE JAMAIS ete
evaluees par 07 (ex: 05 vient d'etre relance avec de nouveaux parametres, mais 07 n'a pas
encore tourne depuis). Voir aussi README.md.
"""

import hashlib
import json

import numpy as np
import pandas as pd

import config
import fenetres
import rapports


# ============================================================
# Parametres GENERAUX : lus directement dans config.py
# ============================================================

CIBLE_ORIGINE = 'excess_return'
# Cible du pipeline principal (horizon 1 mois). Sert de reference : tant que config.CIBLE
# vaut cette valeur, la signature d'une experience est EXACTEMENT celle d'avant l'ajout de
# l'horizon long.


def params_generaux_actuels():
    """Snapshot des parametres GENERAUX actuellement definis dans config.py -- ceux qui,
    s'ils changent, changent le resultat des 3 modeles (04, 05, 06) de la meme facon.
    Voir config.py, section "Parametres GENERAUX", pour le detail de chacun."""
    params = {
        'predicteurs': list(config.PREDICTEURS),
        'type_fenetre': config.TYPE_FENETRE,
        'annee_debut_entrainement': config.ANNEE_DEBUT_ENTRAINEMENT,
        'annees_train_initial': config.ANNEES_TRAIN_INITIAL,
        'annees_validation': config.ANNEES_VALIDATION,
        'annees_test_par_fenetre': config.ANNEES_TEST_PAR_FENETRE,
        'seuil_percentile_taille': config.SEUIL_PERCENTILE_TAILLE,
        'seuil_percentile_liquidite': config.SEUIL_PERCENTILE_LIQUIDITE,
    }

    # Reduction progressive de la validation (config.py) : ces 3 parametres n'entrent dans
    # la signature de l'experience que lorsque la reduction est ACTIVE. Sans ce test, la
    # simple existence de l'option changerait la cle de TOUTES les experiences deja au
    # journal (lancees avant son ajout) : relancer un modele a l'identique creerait un
    # doublon, et ses mesures de portefeuille (notebook 07) ne se rattacheraient plus a
    # rien. Reduction = 0 donne donc exactement les memes fenetres, et la meme cle, qu'avant.
    if config.REDUCTION_VALIDATION_PAR_FENETRE:
        params['reduction_validation_par_fenetre'] = config.REDUCTION_VALIDATION_PAR_FENETRE
        params['fenetre_debut_reduction_validation'] = config.FENETRE_DEBUT_REDUCTION_VALIDATION
        params['annees_validation_minimum'] = config.ANNEES_VALIDATION_MINIMUM

    # Horizon de prediction long (scripts etape11 a etape14) : meme principe que ci-dessus.
    # Ces 3 cles n'entrent dans la signature que lorsque la cible n'est PLUS celle d'origine,
    # c'est-a-dire uniquement pendant l'execution d'un script 11 a 14 (qui bascule
    # config.CIBLE via horizon.activer_mode_horizon). Les experiences a 1 mois deja au
    # journal gardent donc exactement la cle qu'elles avaient, et les deux pistes cohabitent
    # sans jamais se confondre : le notebook 09 les distingue tout seul.
    if config.CIBLE != CIBLE_ORIGINE:
        params['cible'] = config.CIBLE
        params['horizon_mois'] = config.HORIZON_PREDICTION_MOIS
        params['traitement_radiation'] = config.TRAITEMENT_RADIATION
        params['mois_embargo'] = fenetres.MOIS_EMBARGO_PAR_DEFAUT

    return params


# ============================================================
# Cle d'unicite d'une experience (deduplication) et description lisible
# ============================================================

def cle_experience_actuelle(modele, params_specifiques):
    """Calcule la cle d'unicite qu'aurait une experience `modele` avec les parametres
    GENERAUX ACTUELS de config.py et les parametres SPECIFIQUES `params_specifiques`,
    SANS RIEN ENREGISTRER dans le journal -- exactement la meme cle que celle que calculera
    `enregistrer_experience` plus loin dans le notebook (memes arguments = meme hash, voir
    `_cle_experience`).

    A quoi ca sert : les scripts scripts/etape04, etape05 et etape06 appellent cette fonction pour
    taguer leur `resultats_finaux` (nouvelle colonne 'cle_experience') AVANT meme d'appeler
    `enregistrer_experience` (section 8) -- cette colonne se retrouve donc dans
    `outputs/resultats_*.parquet`, que lit ensuite le notebook 07. Celui-ci la reutilise a
    son tour pour taguer ses mesures de portefeuille avec la meme cle (voir
    `enregistrer_performances_portefeuilles`), ce qui permet a `tableaux_par_modele` (plus
    bas) de relier le Sharpe/Sortino/... de CHAQUE experience deja evaluee par 07 a LA
    BONNE ligne de son tableau au notebook 08."""
    return _cle_experience(modele, params_generaux_actuels(), params_specifiques)


def _cle_experience(modele, params_generaux, params_specifiques):
    """Hash court et stable identifiant une experience de facon unique : deux appels
    avec le meme modele et EXACTEMENT les memes parametres (generaux + specifiques)
    donnent toujours la meme cle, quel que soit l'ordre des cles des dicts (`sort_keys`)."""
    signature = json.dumps(
        {
            'modele': modele,
            'generaux': rapports.nettoyer_pour_json(params_generaux),
            'specifiques': rapports.nettoyer_pour_json(params_specifiques),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]


def _label_params_specifiques(params_specifiques):
    """Repr courte et lisible des parametres specifiques, utilisee comme titre de
    tableau/groupe au notebook 08 (ex: 'grille_alpha=[1e-07..1e-01] (10 val.),
    grille_l1_ratio=[0.1..0.9] (5 val.), max_iter=1000'). Les listes de plus de 4
    valeurs sont resumees (min..max + compte) pour rester lisibles en titre."""
    params_specifiques = rapports.nettoyer_pour_json(params_specifiques)
    if not params_specifiques:
        return "(aucun hyperparametre)"
    morceaux = []
    for cle, valeur in params_specifiques.items():
        if isinstance(valeur, list) and len(valeur) > 4:
            morceaux.append(f"{cle}=[{valeur[0]:g}..{valeur[-1]:g}] ({len(valeur)} val.)")
        elif isinstance(valeur, list):
            morceaux.append(f"{cle}={valeur}")
        else:
            morceaux.append(f"{cle}={valeur}")
    return ", ".join(morceaux)


# ============================================================
# Ecriture (appelee par les scripts scripts/etape04, etape05, etape06)
# ============================================================

def enregistrer_experience(modele, params_specifiques, resultats, duree_entrainement_secondes):
    """Ajoute une ligne au journal des experiences (outputs/journal_experiences.parquet),
    sauf si une experience strictement identique (meme modele, memes parametres GENERAUX
    actuellement dans config.py, memes parametres SPECIFIQUES ci-dessous) y figure deja
    -- dans ce cas rien n'est ajoute et la fonction le signale.

    Parametres
    ----------
    modele : str
        Nom du modele, ex: "Regression lineaire", "Elastic Net", "LightGBM" -- garde le
        meme nom d'un script a l'autre pour que le notebook 08 les reconnaisse.
    params_specifiques : dict
        Hyperparametres propres a CE modele uniquement (dict vide {} pour la regression
        lineaire, qui n'en a aucun). Ex pour l'Elastic Net :
        {'grille_alpha': config.GRILLE_ALPHA_ELASTIC_NET,
         'grille_l1_ratio': config.GRILLE_L1_RATIO_ELASTIC_NET,
         'max_iter': config.MAX_ITER_ELASTIC_NET}
    resultats : dict
        Doit contenir au moins 'r2_oos_train', 'r2_oos_validation', 'r2_oos_test' et
        'n_fenetres' (voir scripts/etape04, etape05, etape06).
    duree_entrainement_secondes : float
        Temps total d'entrainement, TOUTES fenetres confondues (recherche
        d'hyperparametres incluse) -- mesure avec time.perf_counter() autour de la
        boucle d'entrainement, voir scripts/etape04, etape05, etape06.

    Retourne
    --------
    True si la ligne a ete ajoutee, False si l'experience existait deja (aucun doublon
    cree).
    """
    params_generaux = params_generaux_actuels()
    cle = _cle_experience(modele, params_generaux, params_specifiques)

    config.assurer_dossiers()
    journal = charger_journal()

    if len(journal) > 0 and cle in journal['cle_experience'].values:
        print(f"Experience deja presente dans le journal (cle {cle}) -- rien ajoute "
              "(meme modele, memes parametres generaux et specifiques qu'un lancement precedent).")
        return False

    params_generaux_json = rapports.nettoyer_pour_json(params_generaux)
    ligne = {
        'cle_experience': cle,
        'horodatage': pd.Timestamp.now().isoformat(timespec='seconds'),
        'modele': modele,
        'duree_entrainement_secondes': float(duree_entrainement_secondes),
        'params_specifiques_json': json.dumps(rapports.nettoyer_pour_json(params_specifiques), sort_keys=True),
        'params_specifiques_label': _label_params_specifiques(params_specifiques),
        'gen_n_predicteurs': len(params_generaux_json['predicteurs']),
        'gen_predicteurs_json': json.dumps(params_generaux_json['predicteurs']),
        'gen_type_fenetre': params_generaux_json['type_fenetre'],
        'gen_annee_debut_entrainement': params_generaux_json['annee_debut_entrainement'],
        'gen_annees_train_initial': params_generaux_json['annees_train_initial'],
        'gen_annees_validation': params_generaux_json['annees_validation'],
        'gen_annees_test_par_fenetre': params_generaux_json['annees_test_par_fenetre'],
        # NaN (et non 0) quand la reduction est inactive : ces deux parametres n'ont alors
        # aucun effet, et deux lancements qui ne different que par eux sont bien la MEME
        # experience (meme cle) -- afficher une valeur ferait croire le contraire.
        'gen_reduction_validation_par_fenetre': params_generaux_json.get('reduction_validation_par_fenetre', 0),
        'gen_fenetre_debut_reduction_validation': params_generaux_json.get('fenetre_debut_reduction_validation', np.nan),
        'gen_annees_validation_minimum': params_generaux_json.get('annees_validation_minimum', np.nan),
        'gen_seuil_percentile_taille': params_generaux_json['seuil_percentile_taille'],
        'gen_seuil_percentile_liquidite': params_generaux_json['seuil_percentile_liquidite'],
        'res_n_fenetres': int(resultats['n_fenetres']),
        'res_r2_oos_train': float(resultats['r2_oos_train']),
        'res_r2_oos_validation': float(resultats['r2_oos_validation']),
        'res_r2_oos_test': float(resultats['r2_oos_test']),
    }

    journal = pd.concat([journal, pd.DataFrame([ligne])], ignore_index=True)
    journal.to_parquet(config.FICHIER_JOURNAL_EXPERIENCES, index=False)
    print(f"Experience ajoutee au journal (cle {cle}) : {modele} | {_label_params_specifiques(params_specifiques)}")
    return True


# ============================================================
# Lecture (appelee par le notebook 08)
# ============================================================

_COLONNES_JOURNAL = [
    'cle_experience', 'horodatage', 'modele', 'duree_entrainement_secondes',
    'params_specifiques_json', 'params_specifiques_label',
    'gen_n_predicteurs', 'gen_predicteurs_json', 'gen_type_fenetre',
    'gen_annee_debut_entrainement', 'gen_annees_train_initial', 'gen_annees_validation', 'gen_annees_test_par_fenetre',
    'gen_reduction_validation_par_fenetre', 'gen_fenetre_debut_reduction_validation',
    'gen_annees_validation_minimum',
    'gen_seuil_percentile_taille', 'gen_seuil_percentile_liquidite',
    'res_n_fenetres', 'res_r2_oos_train', 'res_r2_oos_validation', 'res_r2_oos_test',
]


def charger_journal():
    """Charge outputs/journal_experiences.parquet, ou renvoie un DataFrame vide (memes
    colonnes) s'il n'existe pas encore -- cas du tout premier lancement du projet, avant
    d'avoir execute 04, 05 ou 06 au moins une fois."""
    if config.FICHIER_JOURNAL_EXPERIENCES.exists():
        journal = pd.read_parquet(config.FICHIER_JOURNAL_EXPERIENCES)
        # Un journal ecrit AVANT l'ajout d'un parametre general (ex: la reduction
        # progressive de la validation) n'en a pas la colonne : on la cree vide plutot
        # que de laisser le notebook 08 planter sur une colonne manquante.
        for colonne in _COLONNES_JOURNAL:
            if colonne not in journal.columns:
                journal[colonne] = 0 if colonne == 'gen_reduction_validation_par_fenetre' else np.nan
        return journal[_COLONNES_JOURNAL]
    return pd.DataFrame(columns=_COLONNES_JOURNAL)


# Mesures de performance des portefeuilles long-short (ecrites par le notebook 07, partie
# B.6 -- voir le bandeau en tete de fichier) que `tableaux_par_modele` va essayer de
# rattacher a chaque ligne du journal, via 'cle_experience'.
COLONNES_PORTEFEUILLE = [
    'sharpe_ratio', 'sortino_ratio', 'rendement_annualise', 'volatilite_annualisee',
    'max_drawdown', 't_stat', 'p_value', 'pct_mois_positifs', 'skewness', 'kurtosis', 'n_mois',
]


def charger_performance_portefeuilles():
    """Charge outputs/performance_portefeuilles.parquet (ecrit par le notebook 07, partie
    B.6 -- une ligne par modele, mesures de performance de son portefeuille long-short
    construit a partir des PREDICTIONS ACTUELLEMENT SUR DISQUE, plus la colonne
    'cle_experience' correspondante -- voir `cle_experience_actuelle`), ou un DataFrame
    vide (memes colonnes) si ce fichier n'existe pas encore (notebook 07 jamais execute).

    ⚠️ Ce fichier est ECRASE a chaque lancement de 07 : il ne reflete que le DERNIER
    lancement de chaque modele. Pour l'historique COMPLET (utilise par `tableaux_par_modele`
    / le notebook 08), voir `charger_historique_performance_portefeuilles` ci-dessous."""
    if config.FICHIER_PERFORMANCE_PORTEFEUILLES.exists():
        return pd.read_parquet(config.FICHIER_PERFORMANCE_PORTEFEUILLES)
    return pd.DataFrame(columns=['cle_experience'] + COLONNES_PORTEFEUILLE)


# ============================================================
# Historique CUMULATIF des mesures de portefeuille (jamais ecrase, comme le journal des
# experiences) -- ecriture appelee par le notebook 07, lecture par `tableaux_par_modele`
# (notebook 08). Voir le bandeau en tete de fichier pour le detail du "pourquoi".
# ============================================================

_COLONNES_HISTORIQUE_PORTEFEUILLE = ['cle_experience', 'modele', 'horodatage'] + COLONNES_PORTEFEUILLE


def charger_historique_performance_portefeuilles():
    """Charge outputs/historique_performance_portefeuilles.parquet -- l'historique CUMULATIF
    (jamais ecrase) des mesures de portefeuille long-short calculees par le notebook 07 au
    fil de ses executions successives, une ligne par (modele, cle_experience) deja evaluee
    au moins une fois -- voir `enregistrer_performances_portefeuilles`. Renvoie un
    DataFrame vide (memes colonnes) si ce fichier n'existe pas encore (notebook 07 jamais
    execute depuis l'ajout de cet historique)."""
    if config.FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES.exists():
        return pd.read_parquet(config.FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES)
    return pd.DataFrame(columns=_COLONNES_HISTORIQUE_PORTEFEUILLE)


def enregistrer_performances_portefeuilles(performance):
    """Ajoute a l'historique CUMULATIF (outputs/historique_performance_portefeuilles.parquet,
    jamais ecrase) les lignes de `performance` (le DataFrame calcule par le notebook 07,
    partie B.6 -- index ou colonne 'modele', colonnes COLONNES_PORTEFEUILLE + 'cle_experience')
    dont la `cle_experience` n'y figure PAS DEJA -- meme logique de deduplication que
    `enregistrer_experience` pour le journal des experiences : relancer 07 plusieurs fois
    sur les MEMES predictions (meme cle_experience) n'ajoute jamais de doublon.

    A quoi ca sert : contrairement a outputs/resultats_*.parquet (04/05/06) et
    outputs/performance_portefeuilles.parquet (07), qui ne gardent que le DERNIER lancement
    de chaque modele (voir README.md), cet historique s'accumule au fil des executions de
    07, au meme titre que outputs/journal_experiences.parquet -- c'est lui, et non le simple
    snapshot, que lit par defaut `tableaux_par_modele` pour enrichir les tableaux du
    notebook 08 : une experience garde donc ses mesures de portefeuille meme apres avoir ete
    supplantee par un lancement plus recent du meme modele.

    Parametres
    ----------
    performance : DataFrame indexe par nom de modele (ou avec une colonne 'modele'),
        contenant au moins 'cle_experience' et les colonnes de COLONNES_PORTEFEUILLE.

    Retourne la liste des `cle_experience` effectivement ajoutees (liste vide si toutes y
    figuraient deja -- rien n'est alors ecrit sur disque)."""
    config.assurer_dossiers()
    historique = charger_historique_performance_portefeuilles()
    deja_presentes = set(historique['cle_experience']) if len(historique) > 0 else set()

    a_ajouter = performance.reset_index(names='modele') if 'modele' not in performance.columns else performance.copy()
    a_ajouter = a_ajouter[~a_ajouter['cle_experience'].isin(deja_presentes)].copy()

    if len(a_ajouter) == 0:
        print("Historique des performances de portefeuille : rien de nouveau (toutes les "
              "experiences de ce lancement y figurent deja).")
        return []

    a_ajouter['horodatage'] = pd.Timestamp.now().isoformat(timespec='seconds')
    for c in _COLONNES_HISTORIQUE_PORTEFEUILLE:
        if c not in a_ajouter.columns:
            a_ajouter[c] = np.nan
    a_ajouter = a_ajouter[_COLONNES_HISTORIQUE_PORTEFEUILLE]

    historique = pd.concat([historique, a_ajouter], ignore_index=True)
    historique.to_parquet(config.FICHIER_HISTORIQUE_PERFORMANCE_PORTEFEUILLES, index=False)

    for _, ligne in a_ajouter.iterrows():
        print(f"Performance de portefeuille ajoutee a l'historique (cle {ligne['cle_experience']}) : "
              f"{ligne['modele']} (Sharpe = {ligne['sharpe_ratio']:.3f})")
    return list(a_ajouter['cle_experience'])


# Renommage des colonnes gen_*/res_* vers des noms courts et lisibles, pour l'affichage
# des tableaux au notebook 08 (les noms gen_/res_ restent utiles pour trier/filtrer par
# code, mais sont un peu lourds a lire tels quels dans un tableau).
_NOMS_AFFICHAGE = {
    'gen_n_predicteurs': 'n_predicteurs',
    'gen_type_fenetre': 'type_fenetre',
    'gen_annee_debut_entrainement': 'annee_debut_entrainement',
    'gen_annees_train_initial': 'annees_train_initial',
    'gen_annees_validation': 'annees_validation',
    'gen_annees_test_par_fenetre': 'annees_test_par_fenetre',
    'gen_reduction_validation_par_fenetre': 'reduction_validation',
    'gen_fenetre_debut_reduction_validation': 'debut_reduction_validation',
    'gen_annees_validation_minimum': 'validation_minimum',
    'gen_seuil_percentile_taille': 'seuil_percentile_taille',
    'gen_seuil_percentile_liquidite': 'seuil_percentile_liquidite',
    'res_n_fenetres': 'n_fenetres',
    'res_r2_oos_train': 'r2_oos_train',
    'res_r2_oos_validation': 'r2_oos_validation',
    'res_r2_oos_test': 'r2_oos_test',
    'duree_entrainement_secondes': 'duree_entrainement_(s)',
    'horodatage': 'date_experience',
}

COLONNES_PARAMS_GENERAUX = [
    'gen_n_predicteurs', 'gen_type_fenetre', 'gen_annee_debut_entrainement',
    'gen_annees_train_initial',
    'gen_annees_validation', 'gen_annees_test_par_fenetre',
    'gen_reduction_validation_par_fenetre', 'gen_fenetre_debut_reduction_validation',
    'gen_annees_validation_minimum',
    'gen_seuil_percentile_taille', 'gen_seuil_percentile_liquidite',
]
COLONNES_RESULTATS = ['res_r2_oos_train', 'res_r2_oos_validation', 'res_r2_oos_test', 'res_n_fenetres']


def tableaux_par_modele(journal, historique_performance_portefeuilles=None):
    """Regroupe le journal en un tableau par (modele, parametres specifiques) -- c'est
    la structure attendue par le notebook 08 : DEUX lancements du meme modele avec les
    memes parametres specifiques mais des parametres GENERAUX differents (ex: expanding
    vs rolling) tombent dans le MEME tableau (une ligne chacun) ; deux lancements avec
    des parametres SPECIFIQUES differents (ex: deux grilles d'alpha differentes pour
    l'Elastic Net) donnent DEUX tableaux distincts.

    Chaque tableau inclut aussi les mesures de portefeuille long-short (`sharpe_ratio`,
    `sortino_ratio`, ... voir COLONNES_PORTEFEUILLE) associees a chaque experience, en
    fusionnant sur 'cle_experience' avec l'HISTORIQUE CUMULATIF des mesures de portefeuille
    (par defaut, charge automatiquement outputs/historique_performance_portefeuilles.parquet
    via `charger_historique_performance_portefeuilles` si l'argument n'est pas fourni).
    ⚠️ Ces colonnes valent `NaN` uniquement pour les experiences qui n'ont ENCORE JAMAIS ete
    evaluees par le notebook 07 (voir le bandeau en tete de ce fichier) -- une experience
    deja evaluee au moins une fois GARDE ses mesures, meme apres avoir ete supplantee par un
    lancement plus recent du meme modele. Le TRI de chaque tableau reste base sur
    `r2_oos_test`, jamais sur ces mesures de portefeuille.

    Parametres
    ----------
    journal : DataFrame, tel que renvoye par `charger_journal()`.
    historique_performance_portefeuilles : DataFrame optionnel, tel que renvoye par
        `charger_historique_performance_portefeuilles()` -- passe-le explicitement si tu
        l'as deja charge par ailleurs (evite une relecture disque) ; None (par defaut) le
        recharge.

    Retourne une liste de dicts : {'modele', 'params_specifiques_label', 'tableau'}, un
    par groupe, triee par modele puis par le meilleur r2_oos_test du groupe (decroissant).
    """
    groupes = []
    if len(journal) == 0:
        return groupes

    if historique_performance_portefeuilles is None:
        historique_performance_portefeuilles = charger_historique_performance_portefeuilles()
    colonnes_perf_dispo = [c for c in COLONNES_PORTEFEUILLE if c in historique_performance_portefeuilles.columns]
    if len(historique_performance_portefeuilles) > 0 and 'cle_experience' in historique_performance_portefeuilles.columns:
        # si jamais une meme cle_experience apparaissait plusieurs fois (ne devrait pas
        # arriver, enregistrer_performances_portefeuilles deduplique a l'ecriture), on ne
        # garde que la plus recente pour rester sur UNE ligne par experience au merge
        perf_a_fusionner = (
            historique_performance_portefeuilles
            .sort_values('horodatage')
            .drop_duplicates('cle_experience', keep='last')
            [['cle_experience'] + colonnes_perf_dispo]
            .copy()
        )
    else:
        perf_a_fusionner = pd.DataFrame(columns=['cle_experience'] + COLONNES_PORTEFEUILLE)

    for (modele, label), sous_df in journal.groupby(['modele', 'params_specifiques_label'], sort=False):
        colonnes_a_garder = (
            ['date_experience'] + [c for c in COLONNES_PARAMS_GENERAUX if c in sous_df.columns] +
            ['duree_entrainement_secondes'] + COLONNES_RESULTATS + ['cle_experience']
        )
        tableau = sous_df.rename(columns={'horodatage': 'date_experience'})[colonnes_a_garder].copy()
        tableau = tableau.rename(columns=_NOMS_AFFICHAGE)

        # fusion des mesures de portefeuille (NaN seulement si cette experience n'a JAMAIS
        # ete evaluee par le notebook 07)
        tableau = tableau.merge(perf_a_fusionner, on='cle_experience', how='left')
        for c in COLONNES_PORTEFEUILLE:
            if c not in tableau.columns:
                tableau[c] = np.nan

        tableau = tableau.sort_values('r2_oos_test', ascending=False).reset_index(drop=True)
        groupes.append({
            'modele': modele,
            'params_specifiques_label': label,
            'tableau': tableau,
            'meilleur_r2_oos_test': tableau['r2_oos_test'].max(),
        })

    groupes.sort(key=lambda g: (g['modele'], -g['meilleur_r2_oos_test']))
    return groupes
