"""
Facteurs de risque Fama-French 5 + momentum : nettoyage, chargement, et alphas.

Pourquoi ce fichier existe
--------------------------
Le projet mesure jusqu'ici la performance des portefeuilles long-short en NIVEAU (rendement
annualise, Sharpe, t-stat du rendement moyen) et la compare a deux reperes long only
(univers equipondere, univers pondere capi -- analyse_par_taille.py). Ces reperes disent si
le portefeuille rapporte ; ils ne disent pas s'il rapporte AUTRE CHOSE que ce qu'un
investisseur obtiendrait deja avec des facteurs connus.

C'est ce que ce module ajoute : la regression

    r_LS,t = alpha + b' F_t + e_t

ou F_t regroupe MKT-RF, SMB, HML, RMW, CMA (Fama & French 2015) et MOM (Carhart 1997,
serie de Ken French). L'alpha est le rendement que les facteurs n'expliquent pas ; c'est le
chiffre qu'attend un jury en finance quantitative devant un portefeuille trie sur des
caracteristiques dont plusieurs (`mvel1`, `bm`, `mom12m`, `operprof`, `agr`) sont
precisement les briques des facteurs.

Meme esprit que portefeuilles.py / ensemble.py : la logique est ici, les valeurs dans
config.py, les chemins dans chemins.py. Reutilisable tel quel par le notebook 08 (univers
complet), analyse_par_taille.py (par segment) et horizon.py (piste 12 mois).

⚠️ Deux points de methode, valables pour tout ce fichier
--------------------------------------------------------
1. Pas de soustraction de Rfree au membre de gauche. Le long-short est autofinance (achat
   du decile 10 finance par la vente du decile 1) : le taux sans risque se neutralise
   mecaniquement. Et les rendements du projet sont deja `excess_return` (etape 03 partie A),
   donc la difference des deux deciles l'a de toute facon deja elimine. Les portefeuilles
   long only du projet, eux, sont bien des rendements excedentaires : ils passent aussi
   directement au membre de gauche, sans retraitement.

2. Newey-West obligatoire. A 1 mois l'autocorrelation est faible mais non nulle ; a 12 mois
   les cohortes de Jegadeesh-Titman se chevauchent par construction, et un ecart-type OLS
   classique surestime alors franchement la significativite de l'alpha. `lags_newey_west`
   suit donc l'horizon : voir `lags_par_defaut`.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# Modeles factoriels : une seule liste a maintenir
# ============================================================

# Noms des colonnes tels que ce module les ecrit sur disque (minuscules, sans tiret : la
# convention du reste du projet, et ca evite les patsy/formules a rallonge).
MKT, SMB, HML, RMW, CMA, MOM = 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'mom'

TOUS_FACTEURS = [MKT, SMB, HML, RMW, CMA, MOM]

# Modeles emboites : la colonne "alpha" d'un tableau de these se lit de gauche a droite,
# du plus simple au plus complet. Le lecteur voit ainsi COMBIEN d'alpha chaque bloc de
# facteurs absorbe -- et notamment ce que MOM enleve, ce qui est l'enjeu ici.
MODELES_FACTORIELS = {
    'CAPM': [MKT],
    'FF3': [MKT, SMB, HML],
    'FF5': [MKT, SMB, HML, RMW, CMA],
    'FF5+MOM': [MKT, SMB, HML, RMW, CMA, MOM],
}

MODELE_REFERENCE = 'FF5+MOM'


def lags_par_defaut(horizon=1):
    """Nombre de retards Newey-West.

    A 1 mois : 3 retards, la convention usuelle sur des rendements mensuels non
    chevauchants. A H mois avec des portefeuilles de cohortes, H-1 retards au minimum --
    c'est exactement la duree du chevauchement induit par la detention, et l'ignorer
    reviendrait a traiter comme independantes des observations qui partagent 11 douziemes
    de leurs positions.
    """
    return 3 if horizon == 1 else max(3, horizon - 1)


# ============================================================
# Nettoyage des fichiers bruts de Ken French (a appeler depuis l'etape 02)
# ============================================================

def _lire_csv_french(chemin):
    """Lit un CSV de la Data Library de Ken French et n'en garde que le bloc MENSUEL.

    Ces fichiers ne sont pas des CSV propres : quelques lignes de copyright en tete, puis
    le bloc mensuel (cle `AAAAMM`), puis une ligne vide, puis un bloc ANNUEL (cle `AAAA`)
    dont les valeurs sont sur une echelle totalement differente. Un `pd.read_csv(skiprows=3)`
    naif avale les deux blocs et fabrique des facteurs annuels deguises en mois.

    D'ou le filtre explicite : seules les lignes dont la premiere colonne fait exactement
    6 chiffres sont conservees.
    """
    lignes = []
    with open(chemin, 'r', encoding='utf-8-sig', errors='replace') as fichier:
        entete = None
        for ligne in fichier:
            morceaux = [m.strip() for m in ligne.strip().split(',')]
            if len(morceaux) < 2:
                continue
            if morceaux[0] == '' and entete is None:
                entete = [m.lower().replace('-', '').replace(' ', '') for m in morceaux[1:]]
                continue
            if morceaux[0].isdigit() and len(morceaux[0]) == 6:
                lignes.append([morceaux[0]] + [float(m) for m in morceaux[1:]])

    if entete is None or not lignes:
        raise ValueError(
            f"Format inattendu pour {chemin} : aucun bloc mensuel identifie. Verifie que le "
            "fichier vient bien de la Data Library de Ken French et qu'il n'a pas ete "
            "re-enregistre par un tableur."
        )

    donnees = pd.DataFrame(lignes, columns=['annee_mois'] + entete[:len(lignes[0]) - 1])
    return donnees


def nettoyer_facteurs(chemin_ff5, chemin_momentum, chemin_sortie=None, rap=None):
    """Fusionne les deux CSV bruts en un unique parquet mensuel, en DECIMAL.

    ⚠️ Ken French publie en POURCENTAGE (0.63 = 0,63 %). Le panel du projet, lui, est en
    decimal (`RET` de CRSP, `Rfree` de Welch-Goyal). Sans la division par 100, tous les
    betas seraient divises par 100 et l'alpha absorberait la difference : l'erreur est
    silencieuse et donne des resultats qui ont l'air presque plausibles. C'est le seul vrai
    piege de cette etape.

    La colonne `rf` de Fama-French est conservee a titre de controle uniquement -- le projet
    continue d'utiliser le `Rfree` de Welch-Goyal pour construire `excess_return`, et melanger
    les deux sources de taux sans risque n'apporterait rien.
    """
    ff5 = _lire_csv_french(chemin_ff5)
    momentum = _lire_csv_french(chemin_momentum)

    momentum = momentum.rename(columns={momentum.columns[1]: MOM})
    ff5 = ff5.rename(columns={'mktrf': MKT, 'smb': SMB, 'hml': HML,
                              'rmw': RMW, 'cma': CMA, 'rf': 'rf_french'})

    facteurs = pd.merge(ff5, momentum[['annee_mois', MOM]], on='annee_mois', how='inner')

    colonnes_valeur = [c for c in facteurs.columns if c != 'annee_mois']
    facteurs[colonnes_valeur] = facteurs[colonnes_valeur] / 100.0

    manquantes = [f for f in TOUS_FACTEURS if f not in facteurs.columns]
    if manquantes:
        raise ValueError(f"Facteurs absents apres fusion : {manquantes}. Colonnes lues : "
                         f"{list(facteurs.columns)}")

    facteurs = facteurs.sort_values('annee_mois').reset_index(drop=True)

    print(f"Facteurs FF5 + MOM : {len(facteurs)} mois, de {facteurs['annee_mois'].iloc[0]} "
          f"a {facteurs['annee_mois'].iloc[-1]}")
    print(facteurs[TOUS_FACTEURS].describe().T[['mean', 'std', 'min', 'max']])

    if rap is not None:
        rap.valeur('D_periode_facteurs', [facteurs['annee_mois'].iloc[0],
                                          facteurs['annee_mois'].iloc[-1]])
        rap.valeur('D_n_mois_facteurs', len(facteurs))
        rap.table('D_describe_facteurs', facteurs[TOUS_FACTEURS].describe())

    if chemin_sortie is not None:
        facteurs.to_parquet(chemin_sortie, index=False)
        print("Facteurs sauvegardes :", chemin_sortie)

    return facteurs


def charger_facteurs(chemin):
    """Relit le parquet produit par `nettoyer_facteurs`, indexe par `annee_mois` (str).

    L'index en chaine 'AAAAMM' est volontaire : c'est le format de `annee_mois` partout
    ailleurs dans le projet (etape 02 partie B), ce qui rend l'alignement avec les series de
    rendement immediat et sans conversion de dates.
    """
    chemin = str(chemin)
    if not pd.io.common.file_exists(chemin):
        raise FileNotFoundError(
            f"{chemin} est introuvable. Relance `python scripts/nettoyage_donnees.py` apres "
            "avoir depose les deux CSV de Ken French dans data/raw/."
        )
    facteurs = pd.read_parquet(chemin)
    facteurs['annee_mois'] = facteurs['annee_mois'].astype(str)
    return facteurs.set_index('annee_mois').sort_index()


# ============================================================
# Regression d'alpha
# ============================================================

def alpha_facteurs(rendements, facteurs, noms_facteurs=None, lags_newey_west=3,
                   nb_periodes_par_an=12):
    """Regresse une serie de rendements mensuels sur un jeu de facteurs.

    `rendements` : Series indexee par `annee_mois` (str), rendements MENSUELS deja
    excedentaires ou autofinances -- voir le bandeau en tete de fichier.
    `facteurs`   : DataFrame indexe par `annee_mois`, tel que renvoye par `charger_facteurs`.

    Retourne un dict a plat : alpha mensuel et annualise, sa t-stat Newey-West, sa p-value,
    un beta par facteur, le R2 et le nombre de mois retenus. Format a plat pour se
    concatener directement en DataFrame, comme `portefeuilles.calculer_metriques`.
    """
    noms_facteurs = noms_facteurs or MODELES_FACTORIELS[MODELE_REFERENCE]

    serie = pd.Series(rendements).dropna()
    serie.index = serie.index.astype(str)

    aligne = pd.concat([serie.rename('_r'), facteurs[noms_facteurs]], axis=1, join='inner')
    aligne = aligne.dropna()

    n = len(aligne)
    if n <= len(noms_facteurs) + 1:
        vide = {'alpha_mensuel': np.nan, 'alpha_annualise': np.nan, 'alpha_t_stat': np.nan,
                'alpha_p_value': np.nan, 'r2_facteurs': np.nan,
                'volatilite_residuelle': np.nan, 'ratio_information': np.nan, 'n_mois': n}
        vide.update({f'beta_{f}': np.nan for f in noms_facteurs})
        return vide

    y = aligne['_r'].values
    X = sm.add_constant(aligne[noms_facteurs].values, has_constant='add')

    ajustement = sm.OLS(y, X).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': int(lags_newey_west), 'use_correction': True},
    )

    # ⚠️ Portefeuille ENGENDRE par les facteurs (R2 = 1 a la precision machine pres) : c'est
    # le cas du benchmark equipondere, qui n'est qu'une combinaison lineaire fixe des six
    # series. Son alpha vaut zero par construction, et son residu vaut ~1e-18 : le rapport
    # des deux produit une t-stat purement numerique, de l'ordre de 5, qu'un lecteur presse
    # prendrait pour un resultat. On renvoie NaN plutot qu'un chiffre inventé par les
    # arrondis.
    if 1.0 - ajustement.rsquared < 1e-10:
        engendre = {'alpha_mensuel': 0.0, 'alpha_annualise': 0.0, 'alpha_t_stat': np.nan,
                    'alpha_p_value': np.nan, 'r2_facteurs': 1.0,
                    'volatilite_residuelle': 0.0, 'ratio_information': np.nan, 'n_mois': n}
        for position, nom in enumerate(noms_facteurs, start=1):
            engendre[f'beta_{nom}'] = ajustement.params[position]
            engendre[f't_{nom}'] = np.nan
        return engendre

    # Volatilite du residu : la part de risque que les facteurs ne portent pas. Rapportee a
    # l'alpha, elle donne le ratio d'information -- c'est-a-dire le "Sharpe de l'alpha", la
    # mesure qui repond a "combien de rendement orthogonal par unite de risque orthogonal".
    # C'est cette grandeur, et non l'alpha brut, qui se compare d'un modele a l'autre quand
    # les portefeuilles n'ont pas la meme volatilite.
    volatilite_residuelle = np.std(ajustement.resid, ddof=len(noms_facteurs) + 1) * np.sqrt(
        nb_periodes_par_an)
    alpha_annualise = ajustement.params[0] * nb_periodes_par_an

    resultat = {
        'alpha_mensuel': ajustement.params[0],
        'alpha_annualise': alpha_annualise,
        'alpha_t_stat': ajustement.tvalues[0],
        'alpha_p_value': ajustement.pvalues[0],
        'r2_facteurs': ajustement.rsquared,
        'volatilite_residuelle': volatilite_residuelle,
        'ratio_information': (alpha_annualise / volatilite_residuelle
                              if volatilite_residuelle > 0 else np.nan),
        'n_mois': n,
    }
    for position, nom in enumerate(noms_facteurs, start=1):
        resultat[f'beta_{nom}'] = ajustement.params[position]
        resultat[f't_{nom}'] = ajustement.tvalues[position]
    return resultat


def alphas_emboites(rendements, facteurs, modeles=None, lags_newey_west=3):
    """Le meme portefeuille passe successivement dans CAPM, FF3, FF5, FF5+MOM.

    Retourne un DataFrame : une ligne par modele factoriel, colonnes = sortie de
    `alpha_facteurs`. Les betas des facteurs absents d'un modele valent NaN, ce qui est le
    comportement voulu dans un tableau emboite.
    """
    modeles = modeles or MODELES_FACTORIELS
    lignes = []
    for nom_modele, noms_facteurs in modeles.items():
        ligne = {'modele_factoriel': nom_modele}
        ligne.update(alpha_facteurs(rendements, facteurs, noms_facteurs, lags_newey_west))
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def alphas_par_portefeuille(rendements_par_nom, facteurs, modeles=None,
                            lags_newey_west=3, nom_colonne='modele'):
    """Applique `alphas_emboites` a plusieurs portefeuilles d'un coup.

    `rendements_par_nom` : dict {nom du portefeuille -> Series mensuelle}. Typiquement
    {'Regression lineaire': ..., 'Elastic Net': ..., 'LightGBM': ..., 'Random Forest': ...,
    'Ensemble': ...} au notebook 08.

    Retourne un DataFrame long, une ligne par (portefeuille x modele factoriel).
    """
    blocs = []
    for nom, serie in rendements_par_nom.items():
        bloc = alphas_emboites(serie, facteurs, modeles, lags_newey_west)
        bloc.insert(0, nom_colonne, nom)
        blocs.append(bloc)
    return pd.concat(blocs, ignore_index=True)


# ============================================================
# Les facteurs vus comme portefeuilles concurrents
#
# C'est l'autre moitie de la question, et elle est DISTINCTE de l'alpha. L'alpha demande
# "mon portefeuille rapporte-t-il ce que les facteurs n'expliquent pas ?". Ce qui suit
# demande "mon portefeuille fait-il MIEUX qu'un portefeuille construit avec les facteurs ?".
# Un modele peut tres bien degager un alpha positif significatif tout en ayant un Sharpe
# inferieur a celui du momentum seul : rendement orthogonal et rendement total sont deux
# grandeurs differentes, et un memoire serieux montre les deux.
# ============================================================

def performance_facteurs(facteurs, calculer_metriques, noms_facteurs=None):
    """Passe chaque facteur, un par un, dans la moulinette de mesures du projet.

    `calculer_metriques` est passe en argument (c'est `portefeuilles.calculer_metriques`)
    plutot qu'importe ici : ca evite un import croise entre deux modules de la racine.
    """
    noms_facteurs = noms_facteurs or TOUS_FACTEURS
    lignes = []
    for nom in noms_facteurs:
        ligne = {'facteur': nom}
        ligne.update(calculer_metriques(facteurs[nom]))
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def portefeuille_facteurs(facteurs, noms_facteurs=None, methode='equipondere',
                          fenetre_mois=60, mois_minimum=24):
    """Fabrique UN portefeuille a partir des six facteurs -- le concurrent de vos modeles.

    `methode` :
      - 'equipondere'      : 1/N sur les facteurs retenus. Aucun parametre estime, donc
                             aucune information future utilisee. C'est le repere par defaut,
                             et le seul totalement a l'abri de la critique (DeMiguel,
                             Garlappi & Uppal 2009 : le 1/N bat regulierement les
                             allocations estimees hors echantillon).
      - 'variance_inverse' : poids proportionnels a 1/variance, la variance de chaque
                             facteur etant estimee sur les `fenetre_mois` mois PASSES
                             uniquement. Meme mecanique glissante que ensemble.py.

    ⚠️ Il n'y a volontairement pas de portefeuille tangent estime sur l'ensemble de la
    periode. Ses poids seraient choisis en connaissant les rendements realises : il
    ecraserait n'importe quel modele par pure construction, et le comparer a des
    portefeuilles hors echantillon n'aurait aucun sens.

    ⚠️ Les six facteurs ne sont pas des rendements excedentaires de la meme nature : MKT-RF
    en est un, les cinq autres sont des spreads autofinances. Les additionner revient a
    detenir le marche a hauteur de 1/6 et cinq paris long-short a hauteur de 1/6 chacun.
    C'est defendable et courant, mais c'est un choix : le dire en note de tableau.

    Retourne une Series mensuelle indexee par `annee_mois`.
    """
    noms_facteurs = noms_facteurs or TOUS_FACTEURS
    F = facteurs[noms_facteurs].dropna().sort_index()

    if methode == 'equipondere':
        return F.mean(axis=1).rename('benchmark_facteurs')

    if methode != 'variance_inverse':
        raise ValueError(
            f"methode inconnue : {methode!r}. Attendu 'equipondere' ou 'variance_inverse'."
        )

    rendements = []
    valeurs = F.values
    for position in range(len(F)):
        debut = 0 if not fenetre_mois else max(0, position - int(fenetre_mois))
        passe = valeurs[debut:position]

        if len(passe) < mois_minimum:
            poids = np.full(valeurs.shape[1], 1.0 / valeurs.shape[1])  # repli neutre
        else:
            variances = passe.var(axis=0, ddof=1)
            variances = np.where(variances > 0, variances, np.nan)
            inverse = 1.0 / variances
            poids = inverse / np.nansum(inverse)
            poids = np.nan_to_num(poids)

        rendements.append(float(np.dot(poids, valeurs[position])))

    return pd.Series(rendements, index=F.index, name='benchmark_facteurs')


def comparer_aux_facteurs(series_modeles, facteurs, calculer_metriques,
                          noms_facteurs=None, modele_factoriel=None, lags_newey_west=3,
                          methode_benchmark='equipondere', nom_benchmark='FF5+MOM (1/N)',
                          fenetre_mois=60, mois_minimum=24, inclure_facteurs_seuls=True):
    """LE tableau du memoire : modeles et facteurs sur les memes lignes, memes colonnes.

    Une ligne par portefeuille -- vos quatre modeles, l'ensemble, le portefeuille de
    facteurs, et si `inclure_facteurs_seuls` chacun des six facteurs pris isolement. Les
    colonnes reunissent les mesures de performance du projet (Sharpe, volatilite, drawdown,
    t-stat du rendement moyen) et les mesures factorielles (alpha annualise, sa t-stat
    Newey-West, R2, ratio d'information).

    ⚠️ Toutes les series sont d'abord restreintes aux mois COMMUNS. Sans cela, le Sharpe du
    momentum serait calcule sur toute l'histoire disponible depuis 1927 et celui de vos
    modeles sur la seule periode de test : la comparaison serait truquee en faveur des
    facteurs, dont les rendements des annees 1930 a 1990 n'ont rien a voir avec 2010-2021.

    ⚠️ La ligne du benchmark factoriel et celles des facteurs seuls n'ont pas d'alpha
    renseigne quand le facteur appartient au modele factoriel : regresser un facteur sur
    lui-meme donne un alpha nul par construction, et l'afficher serait trompeur.

    Retourne un DataFrame trie : modeles d'abord, puis benchmark, puis facteurs.
    """
    noms_facteurs = noms_facteurs or MODELES_FACTORIELS[modele_factoriel or MODELE_REFERENCE]
    modele_factoriel = modele_factoriel or MODELE_REFERENCE

    F = facteurs[noms_facteurs].dropna()
    F.index = F.index.astype(str)

    # --- Periode commune : intersection des mois de TOUS les portefeuilles compares ---
    mois_communs = set(F.index)
    for serie in series_modeles.values():
        serie = pd.Series(serie).dropna()
        mois_communs &= set(serie.index.astype(str))
    mois_communs = sorted(mois_communs)

    if not mois_communs:
        raise ValueError(
            "Aucun mois commun entre les portefeuilles et les facteurs. Verifie le format "
            "de l'index : il doit etre une chaine 'AAAAMM', comme `annee_mois` partout "
            "ailleurs dans le projet."
        )

    F = F.loc[mois_communs]
    benchmark = portefeuille_facteurs(F, noms_facteurs, methode_benchmark,
                                      fenetre_mois, mois_minimum).loc[mois_communs]

    a_evaluer = [('Modele', nom, pd.Series(s).dropna().rename_axis(None))
                 for nom, s in series_modeles.items()]
    a_evaluer.append(('Benchmark', nom_benchmark, benchmark))
    if inclure_facteurs_seuls:
        a_evaluer += [('Facteur', nom, F[nom]) for nom in noms_facteurs]

    lignes = []
    for type_ligne, nom, serie in a_evaluer:
        serie = pd.Series(serie)
        serie.index = serie.index.astype(str)
        serie = serie.reindex(mois_communs).dropna()

        ligne = {'type': type_ligne, 'portefeuille': nom}
        ligne.update(calculer_metriques(serie))

        # Un facteur du modele regresse sur ce modele : alpha nul par construction, on
        # laisse la case vide plutot que d'afficher un zero qui ressemblerait a un resultat.
        if not (type_ligne == 'Facteur' and nom in noms_facteurs):
            ligne.update(alpha_facteurs(serie, F, noms_facteurs, lags_newey_west))
        lignes.append(ligne)

    tableau = pd.DataFrame(lignes)
    ordre = pd.Categorical(tableau['type'], categories=['Modele', 'Benchmark', 'Facteur'],
                           ordered=True)
    tableau = tableau.assign(_ordre=ordre).sort_values('_ordre', kind='stable')
    return tableau.drop(columns='_ordre').reset_index(drop=True)


# ============================================================
# Mise en forme pour la these
# ============================================================

def tableau_alphas_latex(alphas, nom_colonne='modele', modele_factoriel=None,
                         facteurs_affiches=None):
    """Reduit la sortie longue d'`alphas_par_portefeuille` a un tableau presentable.

    Une ligne par portefeuille, colonnes : alpha annualise en %, t-stat, betas, R2. C'est
    le format canonique des tableaux d'alpha de la litterature, et il tient dans la largeur
    d'une page en `report`.
    """
    modele_factoriel = modele_factoriel or MODELE_REFERENCE
    facteurs_affiches = facteurs_affiches or MODELES_FACTORIELS[modele_factoriel]

    filtre = alphas[alphas['modele_factoriel'] == modele_factoriel].copy()
    colonnes = ([nom_colonne, 'alpha_annualise', 'alpha_t_stat']
                + [f'beta_{f}' for f in facteurs_affiches]
                + ['r2_facteurs', 'ratio_information', 'n_mois'])
    tableau = filtre[colonnes].copy()
    tableau['alpha_annualise'] = tableau['alpha_annualise'] * 100
    return tableau.reset_index(drop=True)


def tableau_comparaison_latex(comparaison):
    """Reduit la sortie de `comparer_aux_facteurs` aux colonnes qui tiennent sur une page.

    Rendements et alpha en POURCENTAGE annualise, le reste tel quel. Les colonnes de betas
    en sont volontairement absentes : elles ont leur propre tableau
    (`tableau_alphas_latex`), et les empiler ici donnerait quinze colonnes illisibles.
    """
    colonnes = ['type', 'portefeuille', 'rendement_annualise', 'volatilite_annualisee',
                'sharpe_ratio', 'max_drawdown', 'alpha_annualise', 'alpha_t_stat',
                'ratio_information', 'n_mois']
    tableau = comparaison[[c for c in colonnes if c in comparaison.columns]].copy()
    for colonne in ['rendement_annualise', 'volatilite_annualisee', 'max_drawdown',
                    'alpha_annualise']:
        if colonne in tableau.columns:
            tableau[colonne] = tableau[colonne] * 100
    return tableau.reset_index(drop=True)
