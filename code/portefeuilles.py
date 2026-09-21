"""
Portefeuilles par decile, portefeuille long-short et mesures de performance.

Pourquoi ce fichier existe
--------------------------
Cette logique vivait jusqu'ici DANS LES CELLULES du notebook 08 (`assigner_decile`,
la construction des rendements par decile, `calculer_metriques`). Elle etait donc
impossible a reutiliser ailleurs sans la copier-coller -- ce qui aurait garanti, a la
premiere correction, deux versions divergentes du meme calcul.

Meme esprit que fenetres.py / journal.py / ensemble.py : config.py ne contient que des
VALEURS, les modules de la racine contiennent la LOGIQUE partagee. Ici, elle est partagee
entre :
  - le notebook 08  : portefeuilles long-short sur l'univers COMPLET, un par modele
  - scripts/etape10 : exactement les memes portefeuilles, mais construits SEPAREMENT a
                      l'interieur de chaque segment de taille (small caps / large caps)

⚠️ Point de methode important pour l'etape 10 : quand on evalue un sous-univers, les
deciles doivent etre RECALCULES A L'INTERIEUR de ce sous-univers, mois par mois. Reutiliser
les deciles calcules sur l'univers complet n'aurait aucun sens : le decile 10 de l'univers
complet peut ne contenir presque aucune small cap, et le "portefeuille small cap" serait
alors vide ou minuscule certains mois. Les fonctions ci-dessous prennent donc toutes le
sous-ensemble deja filtre en entree.

Toutes les fonctions supposent des rendements MENSUELS non chevauchants (rebalancement
mensuel), ce qui est le cas avec la cible actuelle du projet (`excess_return`, horizon
1 mois).
"""

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Deciles
# ============================================================

def assigner_decile(serie, n=10):
    """Classe une serie en n groupes (deciles), du plus faible (1) au plus eleve (n).

    ℹ️ On passe par `rank(method='first')` avant `qcut` : ca garantit exactement n groupes
    de taille quasi identique, meme en presence de predictions strictement egales (rares
    mais possibles) -- un `qcut` direct sur les predictions brutes peut echouer ou produire
    moins de n groupes en presence d'egalites.

    ⚠️ Si la serie compte moins de n valeurs, les deciles n'ont aucun sens (on ne peut pas
    faire 10 groupes avec 6 titres) : on renvoie NaN plutot que de laisser `qcut` lever une
    exception ou fabriquer des groupes d'un seul titre. Le mois concerne est alors
    simplement absent du portefeuille. Le cas se produit surtout avec un decoupage en
    seuils de DOLLARS fixes, ou un groupe peut etre quasi vide en debut ou en fin de
    periode (voir config.MODE_GROUPES_TAILLE = 'dollars'). L'etape 10 compte et affiche ces
    mois ecartes.
    """
    valides = serie.notna().sum()
    if valides < n:
        return pd.Series(np.nan, index=serie.index)
    return pd.qcut(serie.rank(method='first'), n, labels=False) + 1


def assigner_deciles_par_mois(donnees, colonne_prediction, n=10, colonne_mois='annee_mois'):
    """Renvoie une Series de deciles, recalcules MOIS PAR MOIS sur `donnees`.

    ⚠️ Pourquoi mois par mois, et pas sur toute la periode d'un coup : un decile doit
    representer un portefeuille qu'on pourrait REELLEMENT construire a un instant donne,
    avec les entreprises disponibles CE MOIS-LA. Classer sur l'ensemble de la periode
    melangerait des entreprises de mois differents dans un meme "portefeuille".
    """
    return (
        donnees.groupby(colonne_mois, observed=True)[colonne_prediction]
        .transform(lambda x: assigner_decile(x, n))
    )


def rendements_par_decile(donnees, colonne_decile, colonne_cible, colonne_mois='annee_mois',
                          colonne_poids=None):
    """Rendement mensuel de chaque decile : moyenne du rendement reellement realise (pas
    predit) des entreprises classees dans ce decile ce mois-la.

    `colonne_poids` :
      - None (defaut) : moyenne EQUIPONDEREE. C'est la construction de reference du projet,
        et le comportement historique de cette fonction -- strictement inchange.
      - un nom de colonne (typiquement `config.COLONNE_MVEL1_BRUT`, la capitalisation
        boursiere) : moyenne PONDEREE par cette colonne. Les lignes de poids manquant, nul
        ou negatif sont ecartees du calcul de ce decile-mois.

    ⚠️ Le poids doit etre connu EN t (date de formation), jamais plus tard : `mvel1_brut`
    est la capitalisation du mois t, contemporaine de la prediction, donc sans information
    du futur. Ponderer par une capitalisation de fin de periode de detention serait une
    fuite pure et simple.

    Retourne un DataFrame : index = annee_mois (trie), colonnes = 1..n_deciles.
    """
    if colonne_poids is None:
        return (
            donnees.groupby([colonne_mois, colonne_decile], observed=True)[colonne_cible]
            .mean()
            .unstack(colonne_decile)
            .sort_index()
        )

    utiles = donnees[[colonne_mois, colonne_decile, colonne_cible, colonne_poids]].copy()
    utiles = utiles.dropna(subset=[colonne_cible, colonne_poids])
    utiles = utiles[utiles[colonne_poids] > 0]
    if utiles.empty:
        raise ValueError(
            f"Aucune ligne exploitable pour ponderer par {colonne_poids!r} : colonne "
            "entierement manquante, nulle ou negative. Verifie qu'elle a bien ete fusionnee "
            "depuis le panel (config.COLONNE_MVEL1_BRUT)."
        )

    # Moyenne ponderee = somme(poids x rendement) / somme(poids), decile par decile et mois
    # par mois. On passe par deux sommes plutot que par `apply` : c'est le meme calcul, mais
    # vectorise -- sur un panel de plusieurs millions de lignes, la difference est nette.
    utiles['_produit'] = utiles[colonne_cible] * utiles[colonne_poids]
    agrege = utiles.groupby([colonne_mois, colonne_decile], observed=True)[
        ['_produit', colonne_poids]].sum()
    return (agrege['_produit'] / agrege[colonne_poids]).unstack(colonne_decile).sort_index()


def rendement_long_short(tableau_deciles, nb_deciles=None):
    """Rendement mensuel du portefeuille long-short : achat du decile le plus eleve
    (prediction la plus forte), vente a decouvert du decile le plus faible.

    C'est la construction standard d'un "portefeuille de spread" : si le modele a un vrai
    pouvoir predictif, ce portefeuille doit degager un rendement positif et
    statistiquement significatif, sans qu'on ait besoin de connaitre le signe ou l'ampleur
    du rendement du marche.
    """
    if nb_deciles is None:
        nb_deciles = int(max(tableau_deciles.columns))
    return tableau_deciles[nb_deciles] - tableau_deciles[1]


# ============================================================
# Mesures de performance
# ============================================================

def calculer_metriques(rendements, nb_periodes_par_an=12):
    """Mesures de performance financiere standard, a partir d'une serie de rendements
    mensuels DEJA nets du taux sans risque (`excess_return`, voir etape 03 partie A) --
    d'ou l'absence de soustraction supplementaire de Rfree dans le ratio de Sharpe.

    Retourne un dict (les cles correspondent exactement a journal.COLONNES_PORTEFEUILLE).
    """
    r = pd.Series(rendements).dropna().values
    n = len(r)
    if n < 2:
        vide = {cle: np.nan for cle in [
            'rendement_annualise', 'volatilite_annualisee', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'pct_mois_positifs', 't_stat', 'p_value', 'skewness', 'kurtosis',
        ]}
        vide['n_mois'] = n
        return vide

    moyenne = r.mean()
    ecart_type = r.std(ddof=1)

    rendement_annualise = moyenne * nb_periodes_par_an
    volatilite_annualisee = ecart_type * np.sqrt(nb_periodes_par_an)
    sharpe_ratio = rendement_annualise / volatilite_annualisee if volatilite_annualisee > 0 else np.nan

    # Sortino : ne penalise que la volatilite a la baisse (rendements negatifs)
    downside = np.minimum(r, 0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(nb_periodes_par_an)
    sortino_ratio = rendement_annualise / downside_dev if downside_dev > 0 else np.nan

    # Drawdown maximum, a partir de la courbe de richesse cumulee (base 1)
    richesse = np.cumprod(1 + r)
    sommet_cumule = np.maximum.accumulate(richesse)
    drawdown = richesse / sommet_cumule - 1
    max_drawdown = drawdown.min()

    pct_mois_positifs = (r > 0).mean()

    # t-stat / p-value du rendement moyen (test bilateral, H0 : rendement moyen nul)
    t_stat = moyenne / (ecart_type / np.sqrt(n)) if ecart_type > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

    return {
        'rendement_annualise': rendement_annualise,
        'volatilite_annualisee': volatilite_annualisee,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'pct_mois_positifs': pct_mois_positifs,
        't_stat': t_stat,
        'p_value': p_value,
        'skewness': stats.skew(r),
        'kurtosis': stats.kurtosis(r),  # kurtosis "excess" (0 = gaussien)
        'n_mois': n,
    }


# ============================================================
# Rank-IC (correlation de rang prediction / rendement realise)
# ============================================================

def rank_ic_par_mois(donnees, colonne_prediction, colonne_cible,
                     colonne_mois='annee_mois', n_minimum=10):
    """Information Coefficient de rang (Spearman) mois par mois : correlation entre le
    CLASSEMENT predit et le CLASSEMENT realise des titres, a l'interieur de chaque mois.

    ⚠️ Pourquoi cette mesure est indispensable des qu'on compare deux sous-univers (small
    caps vs large caps) : le R2_oos a pour denominateur la somme des rendements au carre
    (voir fenetres.r2_oos), qui depend directement de la VARIANCE des rendements du
    sous-univers. Les small caps etant nettement plus volatiles, leur R2_oos est
    mecaniquement tire vers le bas sans que le modele y soit pour quelque chose. Le rank-IC,
    lui, ne depend que de l'ORDRE des titres : il est invariant a l'echelle des rendements,
    donc directement comparable d'un groupe a l'autre.

    Renvoie une Series indexee par annee_mois. Les mois comptant moins de `n_minimum`
    titres renvoient NaN (une correlation sur 3 observations ne mesure rien).
    """
    def _ic(groupe):
        if len(groupe) < n_minimum:
            return np.nan
        rang_pred = groupe[colonne_prediction].rank()
        rang_reel = groupe[colonne_cible].rank()
        if rang_pred.std(ddof=0) == 0 or rang_reel.std(ddof=0) == 0:
            return np.nan
        return rang_pred.corr(rang_reel)

    return (
        donnees.groupby(colonne_mois, observed=True)[[colonne_prediction, colonne_cible]]
        .apply(_ic)
        .sort_index()
    )


def resumer_rank_ic(ic_mensuel, nb_periodes_par_an=12):
    """Resume d'une serie mensuelle de rank-IC.

    - `ic_moyen`      : le rank-IC moyen. Ordre de grandeur usuel en asset pricing : 0.01 a
                        0.05 pour un signal mensuel utile.
    - `ic_ir`         : "Information Ratio" du signal = ic_moyen / ecart-type, annualise.
                        Mesure la REGULARITE du signal, pas seulement son intensite.
    - `ic_t_stat`     : test de nullite du rank-IC moyen (H0 : le classement predit n'a
                        aucun lien avec le classement realise).
    - `pct_mois_ic_positif` : proportion de mois ou le signal a le bon signe.
    """
    ic = pd.Series(ic_mensuel).dropna()
    n = len(ic)
    if n < 2:
        return {'ic_moyen': np.nan, 'ic_ecart_type': np.nan, 'ic_ir': np.nan,
                'ic_t_stat': np.nan, 'pct_mois_ic_positif': np.nan, 'n_mois_ic': n}

    moyenne = ic.mean()
    ecart_type = ic.std(ddof=1)
    return {
        'ic_moyen': moyenne,
        'ic_ecart_type': ecart_type,
        'ic_ir': (moyenne / ecart_type * np.sqrt(nb_periodes_par_an)) if ecart_type > 0 else np.nan,
        'ic_t_stat': (moyenne / (ecart_type / np.sqrt(n))) if ecart_type > 0 else np.nan,
        'pct_mois_ic_positif': float((ic > 0).mean()),
        'n_mois_ic': n,
    }


# ============================================================
# Enchainement complet (utilise par scripts/etape10)
# ============================================================

def evaluer_sous_univers(donnees, colonne_prediction, colonne_cible,
                         nb_deciles=10, colonne_mois='annee_mois'):
    """Enchaine tout le calcul d'evaluation sur UN sous-univers deja filtre : deciles
    recalcules mois par mois, rendements par decile, portefeuille long-short, mesures de
    performance et rank-IC.

    Retourne un dict :
        'metriques'            : dict (mesures de performance + resume du rank-IC)
        'rendements_deciles'   : DataFrame (index = annee_mois, colonnes = 1..nb_deciles)
        'rendement_long_short' : Series mensuelle
        'ic_mensuel'           : Series mensuelle
    """
    donnees = donnees.copy()
    donnees['_decile'] = assigner_deciles_par_mois(
        donnees, colonne_prediction, n=nb_deciles, colonne_mois=colonne_mois)

    tableau = rendements_par_decile(donnees, '_decile', colonne_cible, colonne_mois=colonne_mois)
    ls = rendement_long_short(tableau, nb_deciles=nb_deciles)
    ic = rank_ic_par_mois(donnees, colonne_prediction, colonne_cible, colonne_mois=colonne_mois)

    metriques = calculer_metriques(ls)
    metriques.update(resumer_rank_ic(ic))
    metriques['spread_decile_haut_bas'] = float(
        tableau[nb_deciles].mean() - tableau[1].mean())

    return {
        'metriques': metriques,
        'rendements_deciles': tableau,
        'rendement_long_short': ls,
        'ic_mensuel': ic,
    }


# ============================================================
# Portefeuilles a HORIZON long (scripts etape11 a etape14 / notebook 11)
#
# ⚠️ POURQUOI LES FONCTIONS CI-DESSUS NE CONVIENNENT PAS a une cible de H mois.
# `rendements_par_decile` moyenne la cible par decile et par mois, et `calculer_metriques`
# annualise ensuite en x12 et racine(12). Avec une cible longue, ces "rendements mensuels"
# sont en realite des rendements sur H mois qui SE CHEVAUCHENT : deux observations
# consecutives partagent H-1 mois de rendement. L'annualisation est alors fausse, et la
# volatilite massivement sous-estimee (le chevauchement lisse artificiellement la serie).
#
# Trois constructions correctes, toutes implementees ici :
#   - `portefeuille_mensuel`   : rebalancement MENSUEL -- deciles refaits chaque mois sur la
#                                prediction longue, detention 1 mois, rendement MENSUEL
#                                realise. Construction PRINCIPALE de la piste.
#   - `portefeuille_cohortes`  : Jegadeesh-Titman, chevauchant mais reposant sur les
#                                rendements MENSUELS -> serie mensuelle annualisable
#   - `portefeuille_annuel`    : rebalancement une fois par an, sans chevauchement
# Les deux premieres consomment le rendement MENSUEL realise (`excess_return`), jamais la
# cible longue : celle-ci ne sert qu'a CLASSER les titres au moment de la formation. La
# troisieme est la seule a consommer directement la cible longue, ce qu'elle peut se
# permettre puisqu'elle espace ses observations de `horizon` mois.
# ============================================================

def portefeuille_mensuel(donnees, colonne_prediction, colonne_rendement_mensuel,
                         nb_deciles=10, colonne_mois='annee_mois', colonne_poids=None):
    """Rebalancement MENSUEL : exactement la construction du notebook 08, avec pour seule
    difference le SIGNAL utilise pour classer les titres.

    Chaque mois t : on classe les titres en deciles selon la prediction a `horizon` mois, on
    detient UN mois, et le rendement realise pris en compte est le rendement excedentaire
    MENSUEL. La serie obtenue est mensuelle et NON CHEVAUCHANTE : elle s'annualise
    normalement (x12 et racine(12)), sans aucune des precautions qu'exige la cible longue.

    ⚠️ Ce n'est PAS incoherent d'utiliser une prevision a 12 mois pour une detention d'un
    mois : la prediction sert uniquement de SIGNAL DE CLASSEMENT. On fait l'hypothese qu'un
    titre dont on attend un bon rendement sur l'annee tend a mieux se comporter des le mois
    suivant -- c'est exactement l'hypothese que la comparaison avec l'horizon 1 mois teste.

    ⚠️ En contrepartie, cette construction IGNORE l'horizon de detention implicite de la
    cible : elle liquide tout au bout d'un mois alors que le modele a ete entraine a prevoir
    12 mois. Le turnover est donc maximal, et `portefeuille_cohortes` reste la construction
    qui respecte l'horizon. Les deux se lisent ensemble.

    L'interet, et c'est ce qui en fait la construction principale : la mecanique du
    portefeuille etant identique a celle du notebook 08, la comparaison entre les deux
    horizons devient une comparaison TOUTES CHOSES EGALES PAR AILLEURS -- seul le signal
    change, pas la facon de le transformer en portefeuille.

    `colonne_poids` : voir `rendements_par_decile` (None = equipondere).

    Retourne un dict : 'rendements_deciles' (DataFrame mois x decile),
    'rendement_long_short' (Series mensuelle), 'n_mois' (int).
    """
    colonnes = [colonne_mois, colonne_prediction, colonne_rendement_mensuel]
    if colonne_poids is not None:
        colonnes.append(colonne_poids)
    donnees = donnees[colonnes].copy()
    donnees[colonne_mois] = donnees[colonne_mois].astype(str)

    donnees['_decile'] = assigner_deciles_par_mois(
        donnees, colonne_prediction, n=nb_deciles, colonne_mois=colonne_mois)

    tableau = rendements_par_decile(donnees, '_decile', colonne_rendement_mensuel,
                                    colonne_mois=colonne_mois, colonne_poids=colonne_poids)
    return {
        'rendements_deciles': tableau,
        'rendement_long_short': rendement_long_short(tableau, nb_deciles=nb_deciles),
        'n_mois': int(len(tableau)),
    }

def portefeuille_cohortes(donnees, colonne_prediction, colonne_rendement_mensuel,
                          horizon, nb_deciles=10, colonne_mois='annee_mois',
                          colonne_titre='permno', colonne_poids=None):
    """Portefeuilles chevauchants de Jegadeesh-Titman (1993).

    Principe : chaque mois t, on forme une COHORTE en classant les titres selon la
    prediction a H mois, et on la detient H mois. Le rendement du portefeuille au mois t
    est la MOYENNE des rendements mensuels des H cohortes encore actives ce mois-la.

    ⚠️ Le rendement utilise est le rendement MENSUEL realise, jamais la cible longue :
    c'est ce qui permet d'obtenir une serie mensuelle non chevauchante en sortie, donc
    annualisable normalement (x12 et racine(12)), et directement comparable a celle de
    l'horizon 1 mois.

    C'est la convention de reference de la litterature momentum, et elle correspond a une
    strategie reellement implementable : on investit 1/H du capital chaque mois.

    `colonne_poids` : voir `rendements_par_decile`. ⚠️ Le poids retenu est celui du MOIS DE
    FORMATION de la cohorte, pas celui du mois de detention : c'est la capitalisation connue
    au moment ou l'argent a ete investi, et elle reste figee pendant toute la detention --
    exactement ce que fait un portefeuille qu'on ne rebalance pas.

    Retourne un dict : 'rendements_deciles' (DataFrame mois x decile),
    'rendement_long_short' (Series mensuelle), 'n_cohortes_actives' (Series mensuelle).
    """
    colonnes = [colonne_titre, colonne_mois, colonne_prediction, colonne_rendement_mensuel]
    if colonne_poids is not None:
        colonnes.append(colonne_poids)
    donnees = donnees[colonnes].copy()
    donnees[colonne_mois] = donnees[colonne_mois].astype(str)

    # Deciles de formation : recalcules a chaque date de formation, sur les titres
    # disponibles ce mois-la.
    donnees['_decile'] = assigner_deciles_par_mois(
        donnees, colonne_prediction, n=nb_deciles, colonne_mois=colonne_mois)
    colonnes_formation = [colonne_titre, colonne_mois, '_decile']
    if colonne_poids is not None:
        colonnes_formation.append(colonne_poids)
    formation = donnees.dropna(subset=['_decile'])[colonnes_formation].copy()
    formation = formation.rename(columns={colonne_mois: 'mois_formation'})

    # Rendements mensuels realises, indexes par (titre, mois).
    rendements = donnees[[colonne_titre, colonne_mois, colonne_rendement_mensuel]].copy()
    rendements = rendements.rename(columns={colonne_mois: 'mois_detention'})

    # Chaque cohorte formee en `mois_formation` est detenue de +1 a +H mois : on duplique
    # donc chaque ligne de formation H fois, une par mois de detention.
    periodes_formation = pd.PeriodIndex(
        pd.to_datetime(formation['mois_formation'], format='%Y%m'), freq='M')
    morceaux = []
    for k in range(1, horizon + 1):
        morceau = formation.copy()
        morceau['mois_detention'] = (periodes_formation + k).astype(str).str.replace(
            '-', '', regex=False)
        morceau['mois_dans_cohorte'] = k
        morceaux.append(morceau)
    positions = pd.concat(morceaux, ignore_index=True)

    positions = positions.merge(rendements, on=[colonne_titre, 'mois_detention'], how='inner')

    # Rendement d'un decile au mois t : moyenne sur toutes les cohortes actives et tous
    # leurs titres -- equiponderee titre par titre comme le notebook 08, ou ponderee par
    # `colonne_poids` si elle est fournie.
    tableau = rendements_par_decile(
        positions, '_decile', colonne_rendement_mensuel,
        colonne_mois='mois_detention', colonne_poids=colonne_poids)
    tableau.index.name = colonne_mois

    n_cohortes = (
        positions.groupby('mois_detention', observed=True)['mois_formation']
        .nunique()
        .sort_index()
    )

    return {
        'rendements_deciles': tableau,
        'rendement_long_short': rendement_long_short(tableau, nb_deciles=nb_deciles),
        'n_cohortes_actives': n_cohortes,
    }


def portefeuille_annuel(donnees, colonne_prediction, colonne_cible,
                        horizon, nb_deciles=10, colonne_mois='annee_mois',
                        mois_formation=1, colonne_poids=None):
    """Rebalancement une fois par an : aucune observation ne se chevauche.

    On ne forme un portefeuille qu'aux dates espacees de `horizon` mois (par defaut a
    partir du mois calendaire `mois_formation`), et on utilise directement la cible longue
    comme rendement de detention.

    ⚠️ Beaucoup plus simple et statistiquement irreprochable (aucun chevauchement, donc
    aucune autocorrelation induite), mais `horizon` fois moins d'observations : sur 20 ans
    de test, 20 rendements au lieu de 240. Les t-stats s'en ressentent. A utiliser comme
    controle de robustesse des portefeuilles a cohortes, pas comme resultat principal.

    ⚠️ L'annualisation se fait ici avec nb_periodes_par_an = 12 / horizon (une periode par
    an quand horizon = 12), a passer a `calculer_metriques`.
    """
    donnees = donnees.copy()
    donnees[colonne_mois] = donnees[colonne_mois].astype(str)
    mois_calendaire = donnees[colonne_mois].str[4:6].astype(int)

    # Une date de formation tous les `horizon` mois, calee sur `mois_formation`.
    pas = horizon if horizon <= 12 else 12
    dates_retenues = ((mois_calendaire - mois_formation) % pas == 0)
    selection = donnees[dates_retenues].copy()
    if selection.empty:
        raise ValueError(
            f"Aucune date de formation trouvee (mois_formation={mois_formation}, "
            f"horizon={horizon}).")

    selection['_decile'] = assigner_deciles_par_mois(
        selection, colonne_prediction, n=nb_deciles, colonne_mois=colonne_mois)
    tableau = rendements_par_decile(selection, '_decile', colonne_cible,
                                    colonne_mois=colonne_mois, colonne_poids=colonne_poids)

    return {
        'rendements_deciles': tableau,
        'rendement_long_short': rendement_long_short(tableau, nb_deciles=nb_deciles),
        'n_formations': int(len(tableau)),
    }


# ============================================================
# CONSTRUCTIONS ALTERNATIVES (notebooks 08 et 11)
#
# ⚠️ Rien ici ne remplace la construction de reference du projet (long-short EQUIPONDERE par
# decile). Ces fonctions repondent a une question annexe mais importante pour le memoire :
# de combien le ratio de Sharpe bouge-t-il quand on change la MANIERE de transformer les
# predictions en portefeuille, a predictions strictement identiques ?
#
# Aucune ne ré-entraine quoi que ce soit : elles rejouent les predictions deja sur disque.
# Deux leviers, combinables :
#
#   PONDERATION  equipondere (defaut du projet)  vs  ponderee par la capitalisation
#   PERIMETRE    long-short (decile haut - decile bas)  vs  long only (les X % du haut)
# ============================================================

def portefeuille_long_only(donnees, colonne_prediction, colonne_cible, pct=0.20,
                           colonne_mois='annee_mois', colonne_poids=None):
    """Portefeuille LONG ONLY : chaque mois, on achete les `pct` x 100 % des titres les
    mieux notes par le modele, et on ne vend rien a decouvert.

    ⚠️ Trois choses a savoir avant de comparer son Sharpe a celui du long-short :

    1. Ce portefeuille porte l'exposition au MARCHE (beta proche de 1), que le long-short
       neutralise en grande partie. Son Sharpe doit donc etre lu contre celui du marche, pas
       contre celui du long-short : un long only qui bat le long-short ne prouve pas que le
       modele est meilleur, seulement que le marche est monte.
    2. Les rendements du projet etant deja EXCEDENTAIRES (nets du taux sans risque, voir
       etape 03 partie A), aucune soustraction supplementaire n'est necessaire.
    3. Il repond a une contrainte reelle : beaucoup d'investisseurs institutionnels ne
       peuvent pas vendre a decouvert. Si toute la performance du long-short vient de la
       jambe courte, la strategie n'est pas implementable pour eux -- et ce tableau le dit.

    `pct` : fraction des titres achetes (0.20 = les 20 % du haut, soit les deux deciles
    superieurs quand NB_DECILES = 10).
    `colonne_poids` : voir `rendements_par_decile` (None = equipondere).

    ⚠️ Un mois comptant moins de 1/pct titres ne peut pas fournir de selection : il est
    simplement absent de la serie renvoyee. Sur l'univers complet du projet le cas ne se
    presente pas, mais il peut survenir sur un sous-univers etroit (notebook 10).

    Retourne une Series mensuelle de rendements, indexee par `colonne_mois`.
    """
    if not 0 < pct <= 1:
        raise ValueError(f"pct={pct} : la fraction achetee doit etre dans ]0, 1].")

    colonnes = [colonne_mois, colonne_prediction, colonne_cible]
    if colonne_poids is not None:
        colonnes.append(colonne_poids)
    donnees = donnees[colonnes].copy()
    donnees[colonne_mois] = donnees[colonne_mois].astype(str)
    donnees = donnees.dropna(subset=[colonne_prediction, colonne_cible])

    # Rang en PERCENTILE decroissant, recalcule mois par mois : le meilleur titre du mois a
    # le rang 1/n, le pire le rang 1. On garde donc tout ce qui est <= pct.
    # `method='first'` departage les predictions strictement egales (rares, mais possibles)
    # de facon deterministe, comme `assigner_decile` le fait pour les deciles.
    rang = donnees.groupby(colonne_mois, observed=True)[colonne_prediction].rank(
        pct=True, method='first', ascending=False)
    selection = donnees[rang <= pct].copy()
    if selection.empty:
        raise ValueError(
            f"Aucun titre selectionne avec pct={pct} : chaque mois compte moins de "
            f"{int(np.ceil(1 / pct))} titres. Augmente pct, ou elargis l'univers.")

    selection['_groupe'] = 1   # un seul "decile", qui contient toute la selection
    tableau = rendements_par_decile(selection, '_groupe', colonne_cible,
                                    colonne_mois=colonne_mois, colonne_poids=colonne_poids)
    serie = tableau[1]
    serie.name = f"long_only_top_{int(round(pct * 100))}pct"
    return serie


def comparer_constructions(donnees, colonne_prediction, colonne_cible, nb_deciles=10,
                           colonne_poids=None, pct_long_only=0.20,
                           colonne_mois='annee_mois', nb_periodes_par_an=12):
    """Rejoue LES MEMES predictions sous plusieurs constructions de portefeuille, et renvoie
    leurs rendements et leurs mesures de performance cote a cote.

    Constructions calculees :
      - 'Long-short equipondere'        : la REFERENCE du projet (notebook 08)
      - 'Long only top X%  equipondere'
      - 'Long-short pondere capi'       ] uniquement si `colonne_poids` est fournie
      - 'Long only top X%  pondere capi']

    ⚠️ Toutes partent des MEMES predictions et de la MEME cible : tout ecart de Sharpe entre
    deux lignes vient donc uniquement de la construction, jamais du modele. C'est
    precisement ce qui rend le tableau lisible.

    ⚠️ Aucun cout de transaction n'est deduit nulle part, et les quatre constructions n'ont
    pas du tout le meme turnover (le long only en a nettement moins que le long-short, qui
    doit refaire ses deux jambes chaque mois). Le classement ci-dessous est donc a lire comme
    un classement BRUT : c'est une limite a mentionner explicitement dans le memoire.

    Retourne un dict :
        'rendements'  : DataFrame (index = annee_mois, colonnes = constructions)
        'performance' : DataFrame (index = constructions, colonnes = mesures)
    """
    series = {}

    def _long_short(poids):
        table = donnees[[colonne_mois, colonne_prediction, colonne_cible]
                        + ([poids] if poids else [])].copy()
        table[colonne_mois] = table[colonne_mois].astype(str)
        table['_decile'] = assigner_deciles_par_mois(
            table, colonne_prediction, n=nb_deciles, colonne_mois=colonne_mois)
        tableau = rendements_par_decile(table, '_decile', colonne_cible,
                                        colonne_mois=colonne_mois, colonne_poids=poids)
        return rendement_long_short(tableau, nb_deciles=nb_deciles)

    etiquette_long = f"Long only top {int(round(pct_long_only * 100))}%"

    series['Long-short equipondere'] = _long_short(None)
    series[f"{etiquette_long} equipondere"] = portefeuille_long_only(
        donnees, colonne_prediction, colonne_cible, pct=pct_long_only,
        colonne_mois=colonne_mois, colonne_poids=None)

    if colonne_poids is not None:
        series['Long-short pondere capi'] = _long_short(colonne_poids)
        series[f"{etiquette_long} pondere capi"] = portefeuille_long_only(
            donnees, colonne_prediction, colonne_cible, pct=pct_long_only,
            colonne_mois=colonne_mois, colonne_poids=colonne_poids)

    rendements = pd.DataFrame(series).sort_index()
    performance = pd.DataFrame({
        nom: calculer_metriques(serie, nb_periodes_par_an=nb_periodes_par_an)
        for nom, serie in series.items()
    }).T

    return {'rendements': rendements, 'performance': performance}


