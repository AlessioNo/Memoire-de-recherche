"""
Horizon de prediction long : cible composee sur H mois, et execution des modeles dessus.

Pourquoi ce fichier existe
--------------------------
Le pipeline d'origine predit le rendement excedentaire du MOIS SUIVANT. Cette piste-ci
predit le rendement excedentaire des H MOIS SUIVANTS (H = 12 par defaut) :

    cible_{i,t} = PROD_{k=1..H} (1 + R_{i,t+k})  -  PROD_{k=1..H} (1 + Rf_{t+k})

⚠️ C'est bien un rendement EXCEDENTAIRE, pas un rendement total : on capitalise les
rendements et le taux sans risque SEPAREMENT, puis on soustrait. C'est le seul choix
coherent avec la cible d'origine (`excess_return` = RET - Rfree) ; utiliser
PROD(1+R) - 1 changerait d'horizon ET de definition en meme temps, et les deux pistes ne
seraient plus comparables.

⚠️ Cette piste s'AJOUTE au pipeline existant, elle ne le remplace pas. Les sorties de
l'horizon long portent toutes le suffixe `_h{HORIZON}` (voir chemins.py) : elles
n'ecrasent JAMAIS celles de la piste a 1 mois, et les notebooks 04 a 10 continuent
d'afficher exactement ce qu'ils affichaient.

Organisation du module
----------------------
  Section A -- Construction de la cible composee (grille mensuelle, trous, radiations),
               appelee par scripts/construction_panel.py
  Section B -- Contexte d'horizon : bascule PORTEE de la cible et de l'embargo

⚠️ Ce module ne contient plus de boucle d'entrainement. Entrainer un modele sur l'horizon
long se fait par `entrainement/boucle.py` avec `horizon=12` : c'est exactement le meme code
qu'a 1 mois, ce qui garantit qu'une correction apportee a l'un profite a l'autre.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd

import config
import fenetres


# ============================================================
# Section A -- Construction de la cible composee
# ============================================================

# Rendement plancher. R = -1 exactement (perte totale) donnerait log(0) = -inf et
# contaminerait toute la fenetre. On le remplace par une valeur infinitesimalement
# superieure : le produit compose vaut alors ~0 au lieu de 0 pile, ce qui est
# economiquement identique (l'investisseur a tout perdu) et numeriquement propre.
RENDEMENT_PLANCHER = -0.999999

# Conventions de liquidation acceptees par config.TRAITEMENT_RADIATION. Voir
# `_appliquer_traitement_radiation` plus bas pour ce que chacune fait exactement.
TRAITEMENTS_RADIATION = ('taux_sans_risque', 'zero')


def _en_periode(serie_annee_mois):
    """'AAAAMM' (str ou int) -> pd.Period mensuel. Sert a construire une grille de dates
    complete, seule facon de reperer les mois REELLEMENT manquants."""
    return pd.PeriodIndex(pd.to_datetime(serie_annee_mois.astype(str), format='%Y%m'), freq='M')


def _somme_glissante_future(serie, horizon):
    """Somme de `serie` sur la fenetre FUTURE t+1 .. t+horizon, alignee sur t.

    ⚠️ Deux precautions indispensables ici :

    1. On travaille en log(1+R) plutot qu'en produit : plus stable numeriquement, et
       surtout les NaN se propagent naturellement a toute la fenetre qui les contient --
       c'est exactement le comportement voulu (un mois manquant invalide les H dates de
       prevision dont l'horizon l'enjambe, pas seulement une).

    2. `min_periods=horizon` : sans lui, pandas accepterait une fenetre incomplete en
       sommant ce qu'il trouve, ce qui reviendrait a supposer un rendement nul pour les
       mois absents. La serie DOIT deja etre posee sur une grille mensuelle complete
       (voir `_grille_mensuelle_complete`), sinon `rolling` compterait des LIGNES et non
       des MOIS, et enjamberait silencieusement les trous.
    """
    inversee = serie.iloc[::-1]
    somme_courante = inversee.rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    return somme_courante.shift(-1)


def _grille_mensuelle_complete(returns, horizon, mois_fin_panel):
    """Repose chaque permno sur une grille mensuelle continue, et prolonge les titres
    RADIES de `horizon` mois supplementaires.

    Trois situations, traitees differemment (c'est tout l'enjeu de la cible longue) :

    1. TROU AU MILIEU (des donnees reprennent apres) : le mois devient NaN et le reste.
       La cible sera NaN pour les H dates de prevision dont l'horizon l'enjambe, et ces
       lignes seront ecartees. Justification : un trou au milieu de CRSP traduit presque
       toujours un probleme de donnee (code non numerique, cotation suspendue), un
       evenement largement INDEPENDANT de la performance future -- les ecarter retire des
       observations sans deformer l'echantillon.

    2. RADIATION (l'historique s'arrete avant la fin du panel et ne reprend pas) : on
       PROLONGE la grille de `horizon` mois. Le rendement attribue a ces mois fantomes est
       decide par `config.TRAITEMENT_RADIATION` (voir `_appliquer_traitement_radiation`) :
       le taux sans risque (convention Shumway 1997) ou zero. Dans les deux cas,
       l'investisseur subit la perte REELLE jusqu'a la radiation, puis recupere le solde.
       ⚠️ Ecarter ces titres a la place introduirait un biais de survie SEVERE : la
       disparition d'une entreprise est massivement CORRELEE a sa performance (faillite),
       donc on supprimerait precisement les 12 observations ou le rendement a 12 mois est
       le plus negatif. Le modele ne serait alors entraine et evalue que sur des titres
       dont on sait retrospectivement qu'ils ont survecu -- une information du futur.

    3. CENSURE DE FIN D'ECHANTILLON (le titre vit encore au dernier mois du panel) :
       aucune prolongation. Les H dernieres dates de prevision auront une cible NaN et
       seront ecartees. Contrairement a la radiation, cette absence frappe TOUTES les
       entreprises aux memes dates, quelle que soit leur performance : l'ecarter ne
       selectionne rien.
    """
    lignes = []
    for permno, groupe in returns.groupby('permno', sort=False):
        groupe = groupe.sort_values('periode')
        premier, dernier = groupe['periode'].iloc[0], groupe['periode'].iloc[-1]

        radie = dernier < mois_fin_panel
        # ⚠️ La prolongation ne depasse JAMAIS le dernier mois du panel. Deux raisons :
        # le taux sans risque n'existe pas au-dela, et surtout la censure de fin
        # d'echantillon doit frapper TOUS les titres aux memes dates (voir plus bas).
        fin_grille = min(dernier + horizon, mois_fin_panel) if radie else dernier

        grille = pd.period_range(premier, fin_grille, freq='M')
        reindexe = groupe.set_index('periode').reindex(grille)
        reindexe['permno'] = permno
        reindexe['periode'] = grille
        reindexe['est_prolongation'] = grille > dernier
        reindexe['statut_titre'] = 'radie' if radie else 'vivant_fin_panel'
        lignes.append(reindexe)

    return pd.concat(lignes, ignore_index=True)


def _appliquer_traitement_radiation(grille, horizon, rap=None):
    """Remplit le rendement des mois PROLONGES d'un titre radie, selon
    `config.TRAITEMENT_RADIATION`.

    ⚠️ Cette fonction ne touche QUE les mois fantomes (`est_prolongation`), c'est-a-dire les
    mois posterieurs a la derniere cotation d'un titre radie, ajoutes par
    `_grille_mensuelle_complete` pour que l'horizon puisse etre compose jusqu'au bout. Les
    mois REELLEMENT observes avant la radiation ne sont jamais modifies : la chute des
    derniers mois de cotation est conservee telle quelle par les deux conventions.

    Ce que chaque convention donne pour une date de prevision ENTIEREMENT posterieure a la
    radiation (les `horizon` mois de la fenetre sont tous des mois fantomes) :

        'taux_sans_risque'  RET = Rf  ->  PROD(1+Rf) - PROD(1+Rf) = 0
                            L'investisseur recupere le produit de la liquidation et le place
                            au taux sans risque : il gagne exactement le taux sans risque,
                            donc ZERO en excedentaire. Convention Shumway (1997), NEUTRE.

        'zero'              RET = 0   ->  1 - PROD(1+Rf) = -[PROD(1+Rf) - 1]  < 0
                            Le solde dort en caisse sans rien rapporter : l'investisseur
                            supporte le cout d'opportunite du taux sans risque, soit environ
                            -5 % sur 12 mois. Convention CONSERVATRICE.

    Pour une date de prevision qui ENJAMBE la radiation (une partie des mois est reelle,
    l'autre fantome), l'ecart entre les deux conventions ne porte que sur les mois fantomes,
    et il est d'autant plus grand que la radiation intervient tot dans la fenetre.
    """
    traitement = config.TRAITEMENT_RADIATION
    if traitement not in TRAITEMENTS_RADIATION:
        raise ValueError(
            f"config.TRAITEMENT_RADIATION = {traitement!r} inconnu. Valeurs acceptees : "
            + ", ".join(repr(t) for t in TRAITEMENTS_RADIATION) + ".\n"
            "  (La convention 'ecarter' n'est volontairement pas proposee : elle introduit "
            "un biais de survie severe -- voir le commentaire de config.py.)"
        )

    prolonges = grille['est_prolongation']
    n_prolonges = int(prolonges.sum())

    if traitement == 'taux_sans_risque':
        grille.loc[prolonges, 'RET'] = grille.loc[prolonges, 'Rfree']
        libelle = ("liquidation puis placement au TAUX SANS RISQUE (Shumway 1997) -- "
                   "radiation neutre, rendement excedentaire nul apres la radiation")
    else:  # 'zero'
        grille.loc[prolonges, 'RET'] = 0.0
        libelle = ("liquidation puis solde en caisse a RENDEMENT NUL -- radiation "
                   "penalisante, rendement excedentaire negatif du montant du taux sans "
                   "risque compose apres la radiation")

    print(f"\nTraitement des radiations : {traitement!r}")
    print(f"  -> {libelle}")
    print(f"  -> {n_prolonges} mois-titres prolonges concernes")

    if rap is not None:
        rap.valeur('traitement_radiation', traitement)
        rap.valeur('horizon_n_mois_prolonges', n_prolonges)

    return grille


def construire_cible_horizon(returns, macro, horizon=None, rap=None):
    """Calcule la cible composee sur `horizon` mois, titre par titre.

    Parametres
    ----------
    returns : DataFrame [permno, annee_mois, RET]
        ⚠️ DOIT etre `returns_clean` (etape 02), c'est-a-dire l'historique de rendements
        COMPLET, AVANT la fusion avec les caracteristiques et AVANT les filtres taille /
        liquidite de l'etape 03 partie B. Composer apres ces filtres reviendrait a
        enjamber silencieusement les mois que tes propres filtres ont retires.
    macro : DataFrame [annee_mois, Rfree]
    horizon : nb de mois (defaut : config.HORIZON_PREDICTION_MOIS)
    rap : Rapport optionnel, pour y consigner le diagnostic.

    Retourne
    --------
    DataFrame [permno, annee_mois, <config.CIBLE_HORIZON>, n_mois_horizon_observes,
               statut_titre] -- a rattacher au panel par (permno, annee_mois).
    """
    horizon = horizon or config.HORIZON_PREDICTION_MOIS
    nom_cible = config.nom_cible_horizon(horizon)

    returns = returns[['permno', 'annee_mois', 'RET']].copy()
    returns['periode'] = _en_periode(returns['annee_mois'])
    returns = returns.drop(columns='annee_mois')

    taux = macro[['annee_mois', 'Rfree']].copy()
    taux['periode'] = _en_periode(taux['annee_mois'])
    taux = taux.drop(columns='annee_mois').drop_duplicates('periode')

    # ⚠️ La cible est un rendement EXCEDENTAIRE : sans taux sans risque, elle est
    # incalculable. Or l'historique de rendements de CRSP remonte typiquement bien plus loin
    # que le fichier macro (Goyal-Welch). On restreint donc la grille a la periode
    # REELLEMENT couverte par les deux sources.
    # ℹ️ Ces lignes seraient de toute facon supprimees plus tard par l'etape 03 (section A.5,
    # `dropna` sur les colonnes macro) : on ne perd donc aucune observation qui aurait
    # survecu au pipeline. On le fait simplement plus tot, parce que la composition sur H
    # mois a besoin d'une grille homogene.
    debut_macro, fin_macro = taux['periode'].min(), taux['periode'].max()
    avant_restriction = len(returns)
    dans_couverture = (returns['periode'] >= debut_macro) & (returns['periode'] <= fin_macro)
    n_avant_debut = int((returns['periode'] < debut_macro).sum())
    n_apres_fin = int((returns['periode'] > fin_macro).sum())
    returns = returns[dans_couverture]
    n_hors_macro = avant_restriction - len(returns)
    if n_hors_macro:
        print(f"Couverture du fichier macro : {debut_macro} a {fin_macro}")
        print(f"  rendements CONSERVES (dans cette periode) : {len(returns)} lignes")
        print(f"  rendements ECARTES (hors de cette periode) : {n_hors_macro} lignes "
              f"({n_hors_macro / avant_restriction * 100:.1f} %)"
              f" -- dont {n_avant_debut} anterieurs a {debut_macro}"
              f" et {n_apres_fin} posterieurs a {fin_macro}")
        print("  ℹ️ Ces lignes seraient de toute facon supprimees par l'etape 03 (section "
              "A.5, dropna sur les colonnes macro) : aucune observation exploitable n'est "
              "perdue ici.")
    if returns.empty:
        raise ValueError(
            f"Aucun rendement dans la periode couverte par le fichier macro "
            f"({debut_macro} a {fin_macro}). Verifie config.FICHIER_MACRO_CLEAN."
        )

    mois_fin_panel = returns['periode'].max()
    n_permno_depart = returns['permno'].nunique()

    # --- A.1 Grille mensuelle complete (+ prolongation des titres radies) ---
    grille = _grille_mensuelle_complete(returns, horizon, mois_fin_panel)
    grille = grille.merge(taux, on='periode', how='left')

    if grille['Rfree'].isna().any():
        mois_sans_taux = sorted(grille.loc[grille['Rfree'].isna(), 'periode'].unique())
        raise ValueError(
            f"{len(mois_sans_taux)} mois de la grille n'ont pas de taux sans risque, alors "
            f"que la grille est deja restreinte a la couverture macro ({debut_macro} a "
            f"{fin_macro}). Il y a donc un TROU A L'INTERIEUR du fichier macro.\n"
            f"  Premiers mois concernes : {[str(m) for m in mois_sans_taux[:12]]}\n"
            "  Verifie la colonne Rfree de data/interim/macro_clean.parquet (etape 02) : "
            "elle doit etre renseignee pour chaque mois, sans interruption."
        )

    # --- A.2 Convention de liquidation : que rapportent les mois PROLONGES ? ---
    grille = _appliquer_traitement_radiation(grille, horizon, rap)

    n_perte_totale = int((grille['RET'] <= -1).sum())
    grille['RET'] = grille['RET'].clip(lower=RENDEMENT_PLANCHER)

    # --- A.3 Composition sur la fenetre future t+1 .. t+horizon ---
    grille = grille.sort_values(['permno', 'periode'])

    log_actif = np.log1p(grille['RET'])
    log_sans_risque = np.log1p(grille['Rfree'])

    somme_actif = (
        pd.Series(log_actif.values, index=grille.index)
        .groupby(grille['permno'].values, sort=False)
        .transform(lambda s: _somme_glissante_future(s, horizon))
    )
    somme_sans_risque = (
        pd.Series(log_sans_risque.values, index=grille.index)
        .groupby(grille['permno'].values, sort=False)
        .transform(lambda s: _somme_glissante_future(s, horizon))
    )

    grille[nom_cible] = np.expm1(somme_actif) - np.expm1(somme_sans_risque)

    # ⚠️ CENSURE DE FIN D'ECHANTILLON, APPLIQUEE UNIFORMEMENT.
    # Toute date de prevision dont la fenetre depasse le dernier mois du panel est ecartee,
    # que le titre soit vivant ou radie. Ne l'appliquer qu'aux titres vivants creerait un
    # biais de composition severe -- et exactement inverse au biais de survie : sur les
    # 12 derniers mois du panel, l'echantillon ne contiendrait plus QUE des entreprises
    # radiees, puisque ce sont les seules dont on connaitrait le sort a 12 mois.
    fenetre_hors_panel = grille['periode'] + horizon > mois_fin_panel
    grille.loc[fenetre_hors_panel, nom_cible] = np.nan

    # Nombre de mois REELLEMENT observes dans l'horizon (hors prolongation au taux sans
    # risque) : sert au diagnostic et permet, si besoin, d'exclure a posteriori les cibles
    # trop largement reconstituees.
    observe = (~grille['est_prolongation']).astype(float)
    grille['n_mois_horizon_observes'] = (
        pd.Series(observe.values, index=grille.index)
        .groupby(grille['permno'].values, sort=False)
        .transform(lambda s: _somme_glissante_future(s, horizon))
    )

    # --- A.4 Diagnostic ---
    _diagnostiquer(grille, nom_cible, horizon, n_permno_depart, n_perte_totale, rap)

    resultat = grille.loc[
        grille[nom_cible].notna() & ~grille['est_prolongation'],
        ['permno', 'periode', nom_cible, 'n_mois_horizon_observes', 'statut_titre'],
    ].copy()
    resultat['annee_mois'] = resultat['periode'].astype(str).str.replace('-', '', regex=False)
    return resultat.drop(columns='periode').reset_index(drop=True)


def _diagnostiquer(grille, nom_cible, horizon, n_permno_depart, n_perte_totale, rap):
    """Chiffre ce que l'horizon long coute en observations, cas par cas.

    ⚠️ A LIRE AVANT DE CONCLURE QUOI QUE CE SOIT. Un seul mois manquant invalide `horizon`
    dates de prevision, pas une : le cout reel se mesure en titres-mois perdus, jamais en
    nombre de trous.
    """
    reelles = grille[~grille['est_prolongation']]
    n_lignes = len(reelles)
    n_cible_valide = int(reelles[nom_cible].notna().sum())

    trous_milieu = int(reelles['RET'].isna().sum())
    permno_avec_trou = int(reelles.loc[reelles['RET'].isna(), 'permno'].nunique())
    n_radies = int(grille.loc[grille['statut_titre'] == 'radie', 'permno'].nunique())

    # ⚠️ La cause d'une cible manquante ne se lit PAS dans le statut du titre : un titre bien
    # vivant a la fin du panel peut parfaitement avoir un trou au milieu de son historique.
    # Le seul critere fiable est la position de la fenetre : si elle depasse le dernier mois
    # du panel, c'est une censure de fin d'echantillon ; sinon, c'est qu'un mois manque a
    # l'interieur.
    perdues = reelles[reelles[nom_cible].isna()]
    mois_fin_panel = reelles['periode'].max()
    fenetre_hors_panel = perdues['periode'] + horizon > mois_fin_panel
    perdues_censure = int(fenetre_hors_panel.sum())
    perdues_trou = len(perdues) - perdues_censure

    print(f"\n--- Diagnostic de la cible a {horizon} mois ---")
    print(f"Titres-mois observes             : {n_lignes}")
    print(f"  dont cible calculable          : {n_cible_valide} ({n_cible_valide / n_lignes * 100:.1f} %)")
    print(f"  dont ecartes (trou au milieu)  : {perdues_trou} ({perdues_trou / n_lignes * 100:.1f} %)")
    print(f"  dont ecartes (censure de fin)  : {perdues_censure} ({perdues_censure / n_lignes * 100:.1f} %)")
    print(f"Entreprises au depart            : {n_permno_depart}")
    print(f"  avec au moins un trou au milieu: {permno_avec_trou}")
    print(f"  radiees (liquidation Shumway)  : {n_radies}")
    if n_perte_totale:
        print(f"⚠️ {n_perte_totale} rendements <= -100 % ramenes a {RENDEMENT_PLANCHER}")

    if perdues_trou / n_lignes > 0.10:
        print("\n⚠️ Plus de 10 % des titres-mois sont perdus a cause de trous au MILIEU des")
        print("   historiques. A ce niveau, les ecarter finit par selectionner les titres a")
        print("   historique parfait -- donc plutot les grandes capitalisations stables.")
        print("   Envisage d'imputer les mois manquants par le rendement du marche, et")
        print("   mentionne le choix dans le memoire.")

    if rap is not None:
        rap.valeur('horizon_mois', horizon)
        rap.valeur('horizon_n_titres_mois_observes', n_lignes)
        rap.valeur('horizon_n_cible_valide', n_cible_valide)
        rap.valeur('horizon_pct_cible_valide', float(n_cible_valide / n_lignes * 100))
        rap.valeur('horizon_n_perdues_trou_milieu', perdues_trou)
        rap.valeur('horizon_n_perdues_censure_fin', perdues_censure)
        rap.valeur('horizon_n_permno_avec_trou', permno_avec_trou)
        rap.valeur('horizon_n_permno_radies', n_radies)
        rap.valeur('horizon_n_rendements_perte_totale', n_perte_totale)
        rap.table('horizon_describe_cible', reelles[nom_cible].describe().rename(nom_cible))


# ============================================================
# Section B -- Contexte d'horizon (bascule PORTEE de la cible et de l'embargo)
# ============================================================

@contextmanager
def contexte(horizon):
    """Fait travailler tout le projet sur l'horizon `horizon`, le temps d'un bloc `with`.

    Deux reglages, et deux seulement :

    1. `config.CIBLE` -> la cible composee sur `horizon` mois. Toutes les fonctions du
       projet lisent `config.CIBLE` au moment de s'executer (jamais au chargement du
       module) : les basculer suffit a faire travailler la boucle d'entrainement sur
       l'autre cible, SANS dupliquer une seule ligne de code.

    2. `fenetres.MOIS_EMBARGO_PAR_DEFAUT` -> l'embargo aux frontieres des blocs.
       ⚠️ Indispensable : avec un horizon de H mois, la cible des H derniers mois du train
       porte sur des rendements qui appartiennent a la validation, et de meme entre
       validation et test. Sans embargo, il y a fuite de donnees pure et simple.
       Le compte exact est H, pas H-1 : pour qu'une observation en t ait sa fenetre
       t+1..t+H entierement dans son bloc, il faut t + H <= fin du bloc, donc t <= fin - H ;
       les mois ecartes vont de fin-H+1 a fin, soit H mois.

    ⚠️ Pourquoi un `with` et non une fonction `activer_mode_horizon()` qui basculerait ces
    deux valeurs definitivement (ce que faisait la version precedente) : la bascule est un
    effet de bord sur des modules PARTAGES. Tant qu'elle n'est jamais annulee, tout code
    execute ensuite dans le meme processus -- un notebook, un test, un second modele
    enchaine -- travaille silencieusement sur la mauvaise cible. Le `finally` ci-dessous
    restaure systematiquement l'etat d'origine, meme si le bloc leve une exception.

    A `horizon == 1`, ce contexte ne change RIEN : c'est la cible d'origine et un embargo
    nul. Les deux pistes empruntent donc litteralement le meme chemin de code.

        with horizon.contexte(12):
            ...   # config.CIBLE == 'excess_return_12m', embargo == 12
        # ici, config.CIBLE est revenu a 'excess_return' et l'embargo a 0
    """
    if horizon < 1:
        raise ValueError(f"horizon={horizon} : l'horizon de prediction vaut au minimum 1 mois.")

    cible_avant = config.CIBLE
    embargo_avant = fenetres.MOIS_EMBARGO_PAR_DEFAUT

    if horizon == 1:
        # Piste principale : rien a basculer. On passe quand meme par ce chemin pour que
        # les deux horizons partagent exactement le meme code d'appel.
        cible = config.CIBLE
        embargo = 0
    else:
        cible = config.nom_cible_horizon(horizon)
        embargo = horizon
        print(f"Mode horizon active : cible = {cible!r}, embargo = {embargo} mois "
              "aux frontieres train/validation/test")

    config.CIBLE = cible
    fenetres.MOIS_EMBARGO_PAR_DEFAUT = embargo
    try:
        yield cible
    finally:
        config.CIBLE = cible_avant
        fenetres.MOIS_EMBARGO_PAR_DEFAUT = embargo_avant


def preparer_panel(panel, horizon, rap=None):
    """Ecarte les lignes dont la cible longue n'est pas calculable, avant tout fenetrage.

    ⚠️ Ces lignes (trou au milieu d'un historique, ou censure de fin d'echantillon) doivent
    disparaitre AVANT la construction des fenetres : sinon elles fausseraient a la fois les
    R2 et le decompte des mois disponibles.

    A `horizon == 1`, la fonction ne fait rien : la cible mensuelle est deja presente
    partout dans le panel, qui a ete construit pour elle.
    """
    if horizon == 1:
        return panel

    cible = config.CIBLE
    if cible not in panel.columns:
        raise KeyError(
            f"La colonne cible {cible!r} est absente du panel. Relance "
            "`python scripts/construction_panel.py` : c'est elle qui la calcule "
            "(partie A, section A.6bis)."
        )

    avant = len(panel)
    panel = panel[panel[cible].notna()].copy()
    print(f"Lignes sans cible a {horizon} mois ecartees : "
          f"{avant - len(panel)} ({(avant - len(panel)) / avant * 100:.1f} %)")
    if rap is not None:
        rap.valeur('n_lignes_ecartees_cible_manquante', int(avant - len(panel)))
        rap.valeur('shape_panel_avec_cible', list(panel.shape))
    return panel
