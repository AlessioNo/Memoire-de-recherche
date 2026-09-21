"""
Combinaison des predictions de plusieurs modeles en une prediction unique -- la logique du
portefeuille COMBINE du notebook 08 (partie C).

Pourquoi un fichier a part (et pas directement dans le notebook 08) ? Meme raison que
fenetres.py, journal.py et rapports.py : config.py contient des VALEURS, les notebooks
AFFICHENT, et la LOGIQUE vit dans un module a la racine -- testable, reutilisable, et
modifiable a un seul endroit. Les parametres, eux, sont tous dans config.py, section
"Portefeuille COMBINE".

============================================================
L'idee
============================================================
Deux modeles qui se trompent sur des choses DIFFERENTES se completent : la moyenne de leurs
previsions a une erreur quadratique plus faible que chacune prise isolement, meme quand l'un
des deux est nettement moins bon. C'est le resultat fondateur de Bates & Granger (1969), et
c'est exactement l'argument de la diversification d'un portefeuille, applique aux previsions
plutot qu'aux actifs.

Formellement, pour deux previsions d'erreurs de variances s1^2 et s2^2 et de correlation r,
la combinaison de poids w et (1-w) a une variance d'erreur
    w^2 s1^2 + (1-w)^2 s2^2 + 2 w (1-w) r s1 s2
qui est, pour un w bien choisi, INFERIEURE a min(s1^2, s2^2) des que r est assez faible.
Moins les deux modeles sont correles, plus le gain est grand -- d'ou l'interet de combiner
des familles differentes (une foret et un boosting se trompent differemment ; deux Elastic
Net avec des alphas voisins, non).

============================================================
Les cinq methodes de ponderation (config.METHODE_PONDERATION_ENSEMBLE)
============================================================
Deux familles :

A) POIDS CONSTANTS sur toute la periode
   - 'manuelle'       : les poids fixes dans config.POIDS_ENSEMBLE.
   - 'egale'          : 1/N. C'est la reference a battre -- la litterature montre qu'elle
                        est etonnamment difficile a depasser ("forecast combination puzzle",
                        Smith & Wallis 2009), parce que des poids estimes sont eux-memes
                        bruites, et que ce bruit d'estimation coute souvent plus que ce que
                        rapporte l'optimisation.
   - 'r2_validation'  : proportionnels au R2_oos de VALIDATION (pooled) de chaque modele.
                        Determines par les donnees, sans jamais toucher au test.

B) POIDS VARIABLES, ré-estimes CHAQUE MOIS sur les mois de test DEJA PASSES
   - 'inverse_variance' : w proportionnel a 1 / erreur quadratique moyenne (Bates & Granger
                          1969). Simple, robuste : chaque modele est juge sur sa seule
                          precision, sans regarder les correlations entre modeles.
   - 'moindres_carres'  : les poids qui minimisent l'erreur quadratique de la COMBINAISON,
                          c.-a-d. la regression du rendement realise sur les predictions des
                          modeles (Granger & Ramanathan 1984 ; ce qu'on appelle aujourd'hui
                          du "stacking"). Tient compte des correlations entre modeles -- et
                          c'est aussi la plus exposee au bruit d'estimation, d'ou l'option
                          config.POIDS_ENSEMBLE_POSITIFS.

⚠️ FUITE DE DONNEES -- le point critique de tout ce module. Les poids appliques au mois t
ne sont JAMAIS estimes en utilisant le mois t lui-meme, ni aucun mois posterieur :
uniquement les mois strictement anterieurs (au plus config.FENETRE_PONDERATION_ENSEMBLE_MOIS
d'entre eux). Un portefeuille combine construit avec les poids optimaux calcules sur toute
la periode de test afficherait un Sharpe flatteur et parfaitement irrealisable -- c'est
l'erreur classique de ce genre d'exercice. Les tout premiers mois, faute d'historique
suffisant (config.MOIS_MINIMUM_PONDERATION_ENSEMBLE), retombent sur des poids egaux.

============================================================
Utilisation (notebook 08, partie C)
============================================================
    import ensemble

    poids, infos = ensemble.calculer_poids(
        predictions, colonnes_par_modele, config.CIBLE,
        methode=config.METHODE_PONDERATION_ENSEMBLE, ...)
    predictions['pred_ensemble'] = ensemble.appliquer(predictions, colonnes_par_modele, poids)
"""

import numpy as np
import pandas as pd
import fenetres
import portefeuilles
import config

METHODES = ('manuelle', 'egale', 'r2_validation', 'inverse_variance', 'moindres_carres')

# Les methodes de la famille B (poids ré-estimes chaque mois sur le passe).
METHODES_GLISSANTES = ('inverse_variance', 'moindres_carres')


# ============================================================
# Verification de la configuration (appelee en tete de la partie C du notebook 08)
# ============================================================

def verifier_configuration(modeles_disponibles):
    """Verifie config.MODELES_ENSEMBLE / METHODE_PONDERATION_ENSEMBLE / POIDS_ENSEMBLE et
    leve une erreur explicite (avec la correction a faire dans config.py) si quelque chose
    ne colle pas -- plutot qu'un KeyError obscur trois cellules plus loin.

    `modeles_disponibles` : les modeles dont les predictions sont effectivement sur le
    disque (voir la partie B du notebook 08).
    """
    modeles = list(config.MODELES_ENSEMBLE)

    if len(modeles) < 2:
        raise ValueError(
            f"config.MODELES_ENSEMBLE ne contient que {len(modeles)} modele(s) : "
            "il en faut au moins 2 pour construire un portefeuille combine."
        )

    inconnus = [m for m in modeles if m not in modeles_disponibles]
    if inconnus:
        raise ValueError(
            f"Modeles introuvables dans config.MODELES_ENSEMBLE : {inconnus}.\n"
            f"-> Modeles disponibles (predictions sur le disque) : {sorted(modeles_disponibles)}\n"
            "   Verifie l'orthographe exacte dans config.py, ou lance d'abord le script du "
            "modele manquant (scripts/etape04 a etape07)."
        )

    if config.METHODE_PONDERATION_ENSEMBLE not in METHODES:
        raise ValueError(
            f"config.METHODE_PONDERATION_ENSEMBLE = {config.METHODE_PONDERATION_ENSEMBLE!r} "
            f"inconnue. Valeurs acceptees : {list(METHODES)}."
        )

    if config.METHODE_PONDERATION_ENSEMBLE in METHODES_GLISSANTES:
        fenetre = config.FENETRE_PONDERATION_ENSEMBLE_MOIS
        minimum = config.MOIS_MINIMUM_PONDERATION_ENSEMBLE
        if fenetre and fenetre < minimum:
            # Piege silencieux : la fenetre ne peut alors JAMAIS contenir assez de mois pour
            # declencher une estimation, et la combinaison resterait a poids egaux pour
            # toujours -- sans que rien ne le signale, puisque ce repli est prevu.
            raise ValueError(
                f"config.FENETRE_PONDERATION_ENSEMBLE_MOIS ({fenetre}) est plus petite que "
                f"config.MOIS_MINIMUM_PONDERATION_ENSEMBLE ({minimum}) : la fenetre "
                "d'estimation n'atteindra jamais l'historique minimum exige, et les poids "
                "resteraient egaux sur TOUTE la periode.\n"
                "-> Augmente la fenetre, ou baisse le minimum."
            )

    if config.METHODE_PONDERATION_ENSEMBLE == 'manuelle':
        manquants = [m for m in modeles if m not in config.POIDS_ENSEMBLE]
        if manquants:
            raise ValueError(
                f"config.METHODE_PONDERATION_ENSEMBLE = 'manuelle' mais config.POIDS_ENSEMBLE "
                f"n'a pas de poids pour : {manquants}. Ajoute une entree par modele de "
                "MODELES_ENSEMBLE."
            )
        total = sum(config.POIDS_ENSEMBLE[m] for m in modeles)
        if abs(total) < 1e-12:
            raise ValueError(
                "config.POIDS_ENSEMBLE : la somme des poids des modeles retenus est nulle, "
                "impossible de renormaliser."
            )

    return modeles


# ============================================================
# Poids CONSTANTS (famille A)
# ============================================================

def poids_constants(modeles, methode, poids_manuels=None, scores_validation=None):
    """Renvoie une Series de poids (index = modeles), sommant a 1.

    scores_validation : dict {modele: r2_oos_validation}, requis pour 'r2_validation' --
    c'est le R2_oos de validation POOLED de chaque modele, lu dans outputs/resultats_*.parquet.
    """
    if methode == 'egale':
        poids = pd.Series(1.0, index=modeles)

    elif methode == 'manuelle':
        poids = pd.Series({m: float(poids_manuels[m]) for m in modeles})

    elif methode == 'r2_validation':
        # Un modele dont le R2_oos de validation est negatif fait pire que de predire zero :
        # il ne merite aucun poids. Si TOUS sont negatifs, on retombe sur des poids egaux
        # (mieux vaut une combinaison neutre qu'un choix arbitraire).
        bruts = pd.Series({m: max(float(scores_validation[m]), 0.0) for m in modeles})
        poids = bruts if bruts.sum() > 0 else pd.Series(1.0, index=modeles)

    else:
        raise ValueError(f"poids_constants n'accepte pas la methode {methode!r} "
                         f"(methodes a poids constants : 'egale', 'manuelle', 'r2_validation').")

    return poids / poids.sum()


# ============================================================
# Poids GLISSANTS (famille B)
# ============================================================

def _resumes_mensuels(predictions, colonnes, cible):
    """Pre-calcule, pour CHAQUE mois, les quantites suffisantes a l'estimation des poids :

        n   : nombre d'observations du mois
        yy  : somme des y^2
        XX  : matrice (k x k) des produits croises des predictions
        Xy  : vecteur (k) des produits prediction x rendement realise

    Tout ce dont les deux methodes glissantes ont besoin s'en deduit par simple ADDITION sur
    les mois d'une fenetre -- c'est ce qui rend l'estimation mois par mois quasi instantanee
    meme sur des centaines de milliers de lignes : on ne retouche jamais aux donnees
    individuelles, seulement a des matrices k x k (k = nb de modeles, typiquement 2 a 4).
    """
    mois_tries = sorted(predictions['annee_mois'].unique())
    k = len(colonnes)

    n = np.zeros(len(mois_tries))
    yy = np.zeros(len(mois_tries))
    XX = np.zeros((len(mois_tries), k, k))
    Xy = np.zeros((len(mois_tries), k))

    for i, (_, groupe) in enumerate(predictions.groupby('annee_mois', sort=True)):
        X = groupe[colonnes].to_numpy(dtype=float)
        y = groupe[cible].to_numpy(dtype=float)
        n[i] = len(groupe)
        yy[i] = y @ y
        XX[i] = X.T @ X
        Xy[i] = X.T @ y

    return mois_tries, n, yy, XX, Xy


def _poids_inverse_variance(XX_cumul, Xy_cumul, yy_cumul, n_cumul, k):
    """Bates & Granger (1969) : w proportionnel a 1 / erreur quadratique moyenne.

    L'EQM de chaque modele se lit directement dans les sommes cumulees :
        somme (y - p_m)^2 = yy - 2 * Xy[m] + XX[m, m]
    """
    eqm = np.array([(yy_cumul - 2 * Xy_cumul[m] + XX_cumul[m, m]) / n_cumul for m in range(k)])
    if not np.all(np.isfinite(eqm)) or np.any(eqm <= 0):
        return None
    poids = 1.0 / eqm
    return poids / poids.sum()


def _poids_moindres_carres(XX_cumul, Xy_cumul, k, positifs):
    """Granger & Ramanathan (1984) / stacking : les poids qui minimisent
    somme (y - somme_m w_m p_m)^2 sur la fenetre d'estimation.

    Sans contrainte, c'est la solution des equations normales XX w = Xy (sans constante :
    on predit un rendement excedentaire, et le R2_oos du projet compare a une prevision de
    ZERO -- ajouter une constante reviendrait a changer de reference).

    Avec `positifs` (recommande, config.POIDS_ENSEMBLE_POSITIFS) : meme critere, mais sous
    les contraintes w >= 0 et somme(w) = 1. Sans elles, la regression sort regulierement des
    poids du type +2.4 / -1.4, qui collent parfaitement au passe et se comportent tres mal
    ensuite : c'est le resultat classique de la litterature sur la combinaison de previsions
    (Timmermann 2006), et la raison pour laquelle la moyenne simple est si dure a battre.
    """
    if not positifs:
        try:
            poids = np.linalg.solve(XX_cumul, Xy_cumul)
        except np.linalg.LinAlgError:
            return None
        return poids if np.all(np.isfinite(poids)) else None

    # Probleme quadratique de tres petite taille (k = nb de modeles) : SLSQP le resout
    # exactement et instantanement.
    from scipy.optimize import minimize

    def objectif(w):
        return w @ XX_cumul @ w - 2 * w @ Xy_cumul

    def gradient(w):
        return 2 * (XX_cumul @ w - Xy_cumul)

    depart = np.full(k, 1.0 / k)
    resultat = minimize(
        objectif, depart, jac=gradient, method='SLSQP',
        bounds=[(0.0, 1.0)] * k,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                      'jac': lambda w: np.ones(k)}],
        options={'maxiter': 200, 'ftol': 1e-12},
    )
    if not resultat.success or not np.all(np.isfinite(resultat.x)):
        return None
    poids = np.clip(resultat.x, 0.0, None)
    return poids / poids.sum() if poids.sum() > 0 else None


def poids_glissants(predictions, colonnes, cible, methode,
                    fenetre_mois=None, mois_minimum=24, positifs=True):
    """Poids ré-estimes pour CHAQUE mois, sur les mois strictement ANTERIEURS uniquement.

    Parametres
    ----------
    fenetre_mois : nb de mois passes utilises (None ou 0 = tout l'historique passe).
    mois_minimum : en dessous de ce nombre de mois d'historique, poids EGAUX (repli neutre).
    positifs : contrainte w >= 0 et somme = 1, pour 'moindres_carres' uniquement.

    Retourne (DataFrame des poids indexe par annee_mois, dict de diagnostics).
    """
    mois_tries, n, yy, XX, Xy = _resumes_mensuels(predictions, colonnes, cible)
    k = len(colonnes)

    # Sommes cumulees : la fenetre [debut, i-1] s'obtient par difference de deux cumuls,
    # sans jamais reparcourir les lignes.
    cum_n = np.concatenate([[0.0], np.cumsum(n)])
    cum_yy = np.concatenate([[0.0], np.cumsum(yy)])
    cum_XX = np.concatenate([np.zeros((1, k, k)), np.cumsum(XX, axis=0)])
    cum_Xy = np.concatenate([np.zeros((1, k)), np.cumsum(Xy, axis=0)])

    poids_egaux = np.full(k, 1.0 / k)
    lignes = []
    n_replis = 0
    n_mois_egaux_debut = 0

    for i in range(len(mois_tries)):
        debut = 0 if not fenetre_mois else max(0, i - int(fenetre_mois))
        n_mois_estimation = i - debut

        if n_mois_estimation < mois_minimum:
            lignes.append(poids_egaux)
            n_mois_egaux_debut += 1
            continue

        XX_cumul = cum_XX[i] - cum_XX[debut]
        Xy_cumul = cum_Xy[i] - cum_Xy[debut]
        yy_cumul = cum_yy[i] - cum_yy[debut]
        n_cumul = cum_n[i] - cum_n[debut]

        if methode == 'inverse_variance':
            poids = _poids_inverse_variance(XX_cumul, Xy_cumul, yy_cumul, n_cumul, k)
        elif methode == 'moindres_carres':
            poids = _poids_moindres_carres(XX_cumul, Xy_cumul, k, positifs)
        else:
            raise ValueError(f"poids_glissants n'accepte pas la methode {methode!r} "
                             f"(methodes glissantes : {list(METHODES_GLISSANTES)}).")

        if poids is None:   # systeme mal conditionne / optimisation en echec : repli neutre
            poids = poids_egaux
            n_replis += 1
        lignes.append(poids)

    tableau = pd.DataFrame(lignes, index=pd.Index(mois_tries, name='annee_mois'),
                           columns=colonnes)
    diagnostics = {
        'n_mois': len(mois_tries),
        'n_mois_poids_egaux_faute_historique': n_mois_egaux_debut,
        'n_mois_repli_technique': n_replis,
    }
    return tableau, diagnostics


# ============================================================
# Point d'entree unique (utilise par le notebook 08)
# ============================================================

def calculer_poids(predictions, colonnes_par_modele, cible, methode,
                   poids_manuels=None, scores_validation=None,
                   fenetre_mois=None, mois_minimum=24, positifs=True):
    """Renvoie (DataFrame des poids indexe par annee_mois, dict de diagnostics).

    Interface UNIQUE pour les cinq methodes : meme avec des poids constants, le tableau
    renvoye a une ligne par mois (la meme, repetee). Le notebook n'a donc qu'un seul chemin
    de code a gerer, et le graphique d'evolution des poids fonctionne dans tous les cas.

    `colonnes_par_modele` : dict ordonne {nom du modele: nom de la colonne de prediction}.
    """
    modeles = list(colonnes_par_modele)
    colonnes = [colonnes_par_modele[m] for m in modeles]
    mois_tries = sorted(predictions['annee_mois'].unique())

    if methode in METHODES_GLISSANTES:
        tableau, diagnostics = poids_glissants(
            predictions, colonnes, cible, methode,
            fenetre_mois=fenetre_mois, mois_minimum=mois_minimum, positifs=positifs)
        tableau.columns = modeles
        diagnostics['constants'] = False
        return tableau, diagnostics

    poids = poids_constants(modeles, methode, poids_manuels=poids_manuels,
                            scores_validation=scores_validation)
    tableau = pd.DataFrame(
        np.tile(poids.values, (len(mois_tries), 1)),
        index=pd.Index(mois_tries, name='annee_mois'), columns=modeles)
    diagnostics = {
        'n_mois': len(mois_tries),
        'n_mois_poids_egaux_faute_historique': 0,
        'n_mois_repli_technique': 0,
        'constants': True,
    }
    return tableau, diagnostics


def appliquer(predictions, colonnes_par_modele, poids_par_mois):
    """Prediction combinee de chaque ligne : somme des predictions des modeles, ponderee par
    les poids DU MOIS de cette ligne.

    Retourne une Series alignee sur l'index de `predictions`.
    """
    modeles = list(colonnes_par_modele)
    poids_alignes = poids_par_mois.reindex(predictions['annee_mois'].values)
    if poids_alignes.isna().any().any():
        raise ValueError("Certains mois de `predictions` n'ont pas de poids : "
                         "poids_par_mois ne couvre pas toute la periode.")

    valeurs = np.zeros(len(predictions))
    for modele in modeles:
        valeurs += (predictions[colonnes_par_modele[modele]].to_numpy(dtype=float)
                    * poids_alignes[modele].to_numpy(dtype=float))
    return pd.Series(valeurs, index=predictions.index)


# ============================================================
# Identite de l'experience "Ensemble" (journal / historique des portefeuilles)
# ============================================================

def params_specifiques(modeles, cles_composantes):
    """Parametres SPECIFIQUES du portefeuille combine, au sens de journal.py.

    ⚠️ `cles_composantes` (les cle_experience des modeles combines, lues dans
    outputs/resultats_*.parquet) EN FONT PARTIE : sans elles, combiner un LightGBM entraine
    avec une grille A ou une grille B donnerait la meme cle d'experience, alors que ce sont
    deux portefeuilles combines differents. Avec elles, chaque combinaison de modeles-sources
    a bien sa propre ligne au journal et son propre historique de performance.
    """
    params = {
        'modeles': list(modeles),
        'methode_ponderation': config.METHODE_PONDERATION_ENSEMBLE,
        'cles_composantes': [str(c) for c in cles_composantes],
    }
    if config.METHODE_PONDERATION_ENSEMBLE == 'manuelle':
        params['poids_manuels'] = {m: float(config.POIDS_ENSEMBLE[m]) for m in modeles}
    if config.METHODE_PONDERATION_ENSEMBLE in METHODES_GLISSANTES:
        params['fenetre_ponderation_mois'] = config.FENETRE_PONDERATION_ENSEMBLE_MOIS
        params['mois_minimum_ponderation'] = config.MOIS_MINIMUM_PONDERATION_ENSEMBLE
    if config.METHODE_PONDERATION_ENSEMBLE == 'moindres_carres':
        params['poids_positifs'] = bool(config.POIDS_ENSEMBLE_POSITIFS)
    return params


def description(methode=None):
    """Une phrase decrivant la methode de ponderation, pour l'en-tete de la partie C."""
    methode = methode or config.METHODE_PONDERATION_ENSEMBLE
    textes = {
        'manuelle': "poids fixes a la main dans config.POIDS_ENSEMBLE, constants sur toute la periode",
        'egale': "poids egaux (1/N), constants sur toute la periode",
        'r2_validation': "poids proportionnels au R2_oos de validation (pooled) de chaque modele, constants",
        'inverse_variance': ("poids proportionnels a 1/EQM (Bates & Granger 1969), ré-estimes chaque "
                             "mois sur les mois de test deja passes"),
        'moindres_carres': ("poids de moindres carres (Granger & Ramanathan 1984 / stacking), "
                            "ré-estimes chaque mois sur les mois de test deja passes"),
    }
    return textes[methode]


# ============================================================
# Comparaison des methodes de ponderation (notebook 08, partie C)
# ============================================================

METHODES_COMPAREES_DEFAUT = ('egale', 'r2_validation',
                             'inverse_variance', 'moindres_carres')
# 'manuelle' est volontairement exclue : elle demande config.POIDS_ENSEMBLE et n'est pas une
# regle mais un choix arbitraire -- elle n'a pas sa place dans une analyse de sensibilite.


def comparer_methodes(predictions, colonnes_par_modele, cible, nb_deciles=10,
                      methodes=None, scores_validation=None, poids_manuels=None,
                      fenetre_mois=None, mois_minimum=24, positifs=True):
    """Rejoue LA MEME combinaison de modeles sous plusieurs regles de ponderation.

    Composition fixe (colonnes_par_modele), memes predictions, memes mois : tout ecart
    entre deux methodes vient donc UNIQUEMENT de la regle de ponderation. C'est le meme
    protocole que la section B.9 pour les constructions de portefeuille.

    Retourne un dict :
      'methodes', 'modeles'        : listes, dans l'ordre d'affichage
      'poids'                      : {methode: DataFrame mois x modeles}
      'poids_moyens'               : DataFrame modeles x methodes (poids moyen sur la periode)
      'diagnostics'                : DataFrame methodes x diagnostics de calculer_poids
      'predictions_combinees'      : DataFrame aligne sur predictions.index, 1 colonne/methode
      'rendements'                 : DataFrame mois x methodes (long-short mensuel)
      'performance'                : DataFrame methodes x mesures (calculer_metriques)
      'r2_oos'                     : Series methode -> R2_oos test
      'r2_oos_composants'          : Series modele  -> R2_oos test, memes lignes
    """
    methodes = list(methodes) if methodes is not None else list(METHODES_COMPAREES_DEFAUT)
    modeles = list(colonnes_par_modele)

    # Copie de travail minimale : assigner_deciles_par_mois a besoin de 'annee_mois'.
    travail = predictions[['annee_mois', cible]].copy()

    poids, diagnostics, rendements, performance, r2 = {}, {}, {}, {}, {}
    predictions_combinees = pd.DataFrame(index=predictions.index)

    for methode in methodes:
        tableau_poids, diag = calculer_poids(
            predictions, colonnes_par_modele, cible, methode,
            poids_manuels=poids_manuels, scores_validation=scores_validation,
            fenetre_mois=fenetre_mois, mois_minimum=mois_minimum, positifs=positifs)

        combinee = appliquer(predictions, colonnes_par_modele, tableau_poids)

        # MEME construction que la partie B : deciles recalcules mois par mois, D10 - D1,
        # equipondere. Sans quoi les Sharpe ne seraient pas comparables aux autres tableaux.
        travail['pred_combinee'] = combinee.to_numpy()
        travail['decile_combinee'] = portefeuilles.assigner_deciles_par_mois(
            travail, 'pred_combinee', n=nb_deciles)
        tableau_deciles = portefeuilles.rendements_par_decile(
            travail, 'decile_combinee', cible)

        poids[methode] = tableau_poids
        diagnostics[methode] = diag
        predictions_combinees[methode] = combinee
        rendements[methode] = portefeuilles.rendement_long_short(
            tableau_deciles, nb_deciles=nb_deciles)
        performance[methode] = portefeuilles.calculer_metriques(rendements[methode])
        r2[methode] = fenetres.r2_oos(predictions[cible], combinee)

    return {
        'methodes': methodes,
        'modeles': modeles,
        'poids': poids,
        'poids_moyens': pd.DataFrame({m: poids[m].mean() for m in methodes}),
        'diagnostics': pd.DataFrame(diagnostics).T,
        'predictions_combinees': predictions_combinees,
        'rendements': pd.DataFrame(rendements),
        'performance': pd.DataFrame(performance).T,
        'r2_oos': pd.Series(r2),
        'r2_oos_composants': pd.Series(
            {m: fenetres.r2_oos(predictions[cible], predictions[colonnes_par_modele[m]])
             for m in modeles}),
    }