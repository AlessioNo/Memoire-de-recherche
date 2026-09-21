"""
Fenetres d'entrainement glissantes / extensives, et utilitaires partages par
les notebooks 04 a 07.

Pourquoi un fichier a part (et pas directement dans config.py) ? config.py ne
contient que des VALEURS (chemins, parametres) ; celui-ci contient de la
LOGIQUE (des fonctions) reutilisee par 4 notebooks differents (04 a 07) --
les regrouper ici evite de copier-coller les memes fonctions dans chacun, et
garantit qu'un futur changement (ex: une correction dans le calcul du R2_oos)
se repercute automatiquement partout, comme config.py le fait deja pour les
parametres.

Rappel du principe (voir aussi config.py, section "Fenetres d'entrainement") :
au lieu d'un seul decoupage train/validation/test fixe pour tout le projet, le
modele est ré-entrainé plusieurs fois, sur une succession de fenetres qui
avancent dans le temps :

    fenetre 0 : train [1980-1997] | validation [1998-2009] | test [2010]
    fenetre 1 : train [1980-1998] | validation [1999-2010] | test [2011]   (expanding)
        ou   : train [1981-1998] | validation [1999-2010] | test [2011]   (rolling)
    fenetre 2 : train [1980-1999] | validation [2000-2011] | test [2012]   (expanding)
        ...

Option : reduction progressive de la validation
-----------------------------------------------
La validation peut RETRECIR au fil des fenetres (config.py, section "Fenetres
d'entrainement" : REDUCTION_VALIDATION_PAR_FENETRE, FENETRE_DEBUT_REDUCTION_VALIDATION,
ANNEES_VALIDATION_MINIMUM). Avec une reduction de 1 an par fenetre a partir de la
fenetre 1, un train initial de 10 ans et une validation initiale de 10 ans :

    fenetre 0 : train [1980-1989] | validation [1990-1999] (10 ans) | test [2000]
    fenetre 1 : train [1980-1991] | validation [1992-2000] ( 9 ans) | test [2001]  (expanding)
        ou   : train [1982-1991] | validation [1992-2000] ( 9 ans) | test [2001]  (rolling)
    fenetre 2 : train [1980-1993] | validation [1994-2001] ( 8 ans) | test [2002]  (expanding)
        ...

⚠️ Ce qui est FIXE dans ce protocole, c'est le test : il avance toujours de
ANNEES_TEST_PAR_FENETRE d'une fenetre a l'autre, sans trou ni chevauchement (sinon la
serie hors-echantillon mise bout a bout, et donc le R2_oos final, n'aurait plus de sens).
La validation est donc calee sur la FIN (elle s'arrete toujours juste avant le test), et
le train occupe tout ce qui reste devant elle. Consequence : l'annee liberee par la
validation n'est pas perdue, elle passe au train -- en "expanding" le train grandit de
ANNEES_TEST_PAR_FENETRE + REDUCTION_VALIDATION_PAR_FENETRE par fenetre (au lieu de
ANNEES_TEST_PAR_FENETRE seul), en "rolling" il garde sa taille fixe et glisse d'autant.

A chaque fenetre, le modele est ré-entrainé sur son train, ses hyperparametres
(s'il en a) sont choisis sur sa validation, puis on predit sur son test -- une
seule fois, jamais reutilise pour choisir quoi que ce soit. Les predictions de
test de TOUTES les fenetres sont ensuite mises bout a bout pour former une
seule serie hors-echantillon continue, exactement comme dans Gu, Kelly & Xiu
(2020) -- c'est cette serie mise bout a bout qui sert au R2_oos final et aux
portefeuilles par decile du notebook 07, pas une fenetre individuelle.
"""

import numpy as np
import pandas as pd


# ============================================================
# Embargo aux frontieres des blocs (horizon de prediction long)
# ============================================================
MOIS_EMBARGO_PAR_DEFAUT = 0
# ⚠️ 0 = comportement d'origine du projet, INCHANGE. Tant que cette valeur vaut 0,
# `preparer_fenetre` se comporte exactement comme avant, et les scripts etape04 a etape07
# ne voient aucune difference.
#
# A quoi ca sert : avec une cible qui porte sur H mois (horizon long, scripts etape11 a
# etape14), la cible des derniers mois du TRAIN porte sur des rendements qui appartiennent
# a la VALIDATION -- et de meme entre validation et test. C'est une fuite de donnees pure et
# simple : le modele s'entraine sur de l'information provenant du bloc suivant.
#
# Le remede est l'embargo (purge, Lopez de Prado) : on retire les H derniers mois de chaque
# bloc. Le compte exact est H, pas H-1 : pour qu'une observation en t ait sa fenetre
# t+1..t+H entierement contenue dans son bloc, il faut t + H <= fin du bloc, donc
# t <= fin - H ; les mois ecartes vont de fin-H+1 a fin, soit H mois.
#
# Cette valeur est positionnee par horizon.activer_mode_horizon(), appelee uniquement par
# les scripts etape11 a etape14. Elle n'est jamais modifiee ailleurs.


def appliquer_embargo(mois_bloc, mois_embargo):
    """Retire les `mois_embargo` DERNIERS mois d'un bloc (train ou validation).

    ⚠️ On ne touche jamais au TEST : ses observations ne servent a entrainer ni a
    selectionner quoi que ce soit, donc leur horizon peut deborder sur la fenetre suivante
    sans creer la moindre fuite. Les dates de test dont la cible n'est pas calculable sont
    de toute facon deja absentes du panel (voir horizon.construire_cible_horizon).
    """
    if mois_embargo <= 0:
        return list(mois_bloc)
    mois_tries = sorted(mois_bloc)
    if mois_embargo >= len(mois_tries):
        return []
    return mois_tries[:-mois_embargo]


def r2_oos(y_true, y_pred):
    """R2 hors-echantillon a la Gu, Kelly & Xiu (2020).

    Contrairement au R2 habituel (qui compare aux erreurs d'une prediction
    naive egale a la MOYENNE de y), celui-ci compare aux erreurs d'une
    prediction naive de ZERO -- plus strict, et plus adapte aux rendements
    financiers (voir notebook 04, section 2, pour la discussion complete).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)


def annees_validation_fenetre(numero, annees_validation,
                              reduction_validation_par_fenetre=0,
                              fenetre_debut_reduction_validation=1,
                              annees_validation_minimum=1):
    """Nombre d'annees de validation de la fenetre `numero` (0, 1, 2, ...).

    Sans reduction (reduction_validation_par_fenetre=0, valeur par defaut), renvoie
    toujours `annees_validation` : comportement d'origine du projet.

    Avec reduction : la fenetre `fenetre_debut_reduction_validation` est la PREMIERE a
    etre raccourcie (d'un pas), la suivante de deux pas, etc., jusqu'au plancher
    `annees_validation_minimum`. Exemple (validation=10, reduction=1, debut=1) :
    fenetre 0 -> 10 ans, fenetre 1 -> 9 ans, fenetre 2 -> 8 ans...
    """
    if reduction_validation_par_fenetre <= 0 or numero < fenetre_debut_reduction_validation:
        return annees_validation
    nb_pas = numero - fenetre_debut_reduction_validation + 1
    return max(annees_validation_minimum,
               annees_validation - reduction_validation_par_fenetre * nb_pas)


def generer_fenetres(mois_disponibles, type_fenetre, annees_train_initial,
                      annees_validation, annees_test_par_fenetre,
                      reduction_validation_par_fenetre=0,
                      fenetre_debut_reduction_validation=1,
                      annees_validation_minimum=1):
    """Construit la liste des fenetres d'entrainement/validation/test.

    Parametres
    ----------
    mois_disponibles : iterable d'annee_mois (str, format "AAAAMM"), toutes les
        periodes presentes dans le panel (pas besoin d'etre triees ni uniques).
    type_fenetre : "expanding" ou "rolling" (voir config.py, TYPE_FENETRE).
    annees_train_initial : nb d'annees d'entrainement de la toute premiere
        fenetre (et taille FIXE du train a chaque fenetre si type_fenetre="rolling").
    annees_validation : nb d'annees de validation de la PREMIERE fenetre (fenetre
        glissante, dans les 2 modes -- seul le train differe entre "expanding" et
        "rolling"). Constant d'une fenetre a l'autre, sauf si
        reduction_validation_par_fenetre > 0 ci-dessous.
    annees_test_par_fenetre : nb d'annees de test evaluees avant de ré-entrainer
        (1 = ré-entrainement annuel, comme Gu, Kelly & Xiu 2020).
    reduction_validation_par_fenetre : nb d'annees dont la validation RETRECIT a chaque
        nouvelle fenetre (0 = jamais, comportement d'origine). Voir la docstring du
        module pour le schema complet et l'effet sur le train.
    fenetre_debut_reduction_validation : numero de la PREMIERE fenetre raccourcie
        (1 = des la deuxieme fenetre ; sans effet si la reduction vaut 0).
    annees_validation_minimum : plancher, la validation ne descend jamais en dessous.

    Retourne
    --------
    Une liste de dicts (un par fenetre), avec les cles :
        'numero'     : numero de la fenetre (0, 1, 2, ...)
        'train'      : liste des annee_mois d'entrainement de cette fenetre
        'validation' : liste des annee_mois de validation de cette fenetre
        'test'       : liste des annee_mois de test de cette fenetre
        'annee_test' : annee(s) couverte(s) par le test, pour l'affichage
                       (ex: "2010" ou "2010-2011" si annees_test_par_fenetre > 1)
        'annees_validation' : nb d'annees de validation DE CETTE FENETRE (utile quand
                       la reduction est active : c'est ce qui change d'une fenetre a
                       l'autre, voir resumer_fenetres)

    Le calage se fait sur le TEST : la fenetre k teste les annees
    [premiere_annee + annees_train_initial + annees_validation + k * annees_test_par_fenetre, ...],
    la validation occupe les annees juste avant, et le train tout ce qui reste devant
    elle (depuis premiere_annee en "expanding", sur annees_train_initial annees en
    "rolling"). Les tests des fenetres successives se suivent donc exactement, sans trou
    ni chevauchement, que la validation retrecisse ou non.

    On s'arrete des que la fenetre de test suivante depasserait le dernier mois
    disponible -- la toute derniere fenetre peut donc couvrir moins d'annees de
    test que les autres si le nombre total d'annees ne tombe pas juste.
    """
    if type_fenetre not in ("expanding", "rolling"):
        raise ValueError(
            f"type_fenetre inconnu : {type_fenetre!r} (attendu 'expanding' ou 'rolling')"
        )
    if annees_validation_minimum < 1:
        raise ValueError(
            f"annees_validation_minimum={annees_validation_minimum} : il faut au moins 1 annee "
            "de validation (les hyperparametres de 05 et 06 s'y choisissent). Corrige "
            "ANNEES_VALIDATION_MINIMUM dans config.py."
        )

    mois_tries = sorted(set(mois_disponibles))
    if not mois_tries:
        raise ValueError("mois_disponibles est vide : impossible de construire des fenetres.")

    annees_disponibles = sorted(set(int(m[:4]) for m in mois_tries))
    premiere_annee = annees_disponibles[0]
    derniere_annee = annees_disponibles[-1]

    mois_par_annee = {}
    for m in mois_tries:
        mois_par_annee.setdefault(int(m[:4]), []).append(m)

    def mois_des_annees(annees):
        resultat = []
        for a in annees:
            resultat.extend(mois_par_annee.get(a, []))
        return resultat

    liste_fenetres = []
    numero = 0
    # Premiere annee de test de la fenetre 0 : juste apres le train initial et la
    # validation initiale. Elle avance ensuite de annees_test_par_fenetre par fenetre,
    # quoi qu'il arrive a la validation (voir docstring).
    debut_test = premiere_annee + annees_train_initial + annees_validation

    while debut_test <= derniere_annee:
        fin_test = min(debut_test + annees_test_par_fenetre - 1, derniere_annee)

        # La validation s'arrete toujours juste avant le test ; seule sa LONGUEUR change.
        n_annees_validation = annees_validation_fenetre(
            numero, annees_validation, reduction_validation_par_fenetre,
            fenetre_debut_reduction_validation, annees_validation_minimum)
        fin_validation = debut_test - 1
        debut_validation = fin_validation - n_annees_validation + 1

        # Le train occupe ce qui reste devant la validation.
        fin_train = debut_validation - 1
        if type_fenetre == "expanding":
            debut_train = premiere_annee                      # le debut ne bouge pas : le train grandit
        else:  # "rolling"
            debut_train = fin_train - annees_train_initial + 1  # taille fixe : le train glisse
        debut_train = max(debut_train, premiere_annee)

        mois_train = mois_des_annees(range(debut_train, fin_train + 1))
        mois_validation = mois_des_annees(range(debut_validation, fin_validation + 1))
        mois_test = mois_des_annees(range(debut_test, fin_test + 1))

        if mois_train and mois_validation and mois_test:
            liste_fenetres.append({
                'numero': numero,
                'train': mois_train,
                'validation': mois_validation,
                'test': mois_test,
                'annee_test': f"{debut_test}" if debut_test == fin_test else f"{debut_test}-{fin_test}",
                'annees_validation': n_annees_validation,
            })

        if fin_test >= derniere_annee:
            break

        numero += 1
        debut_test += annees_test_par_fenetre

    if not liste_fenetres:
        raise ValueError(
            "Aucune fenetre generee : la periode disponible "
            f"({premiere_annee}-{derniere_annee}, {len(annees_disponibles)} ans) est trop courte pour "
            f"annees_train_initial={annees_train_initial} + annees_validation={annees_validation} "
            f"+ annees_test_par_fenetre={annees_test_par_fenetre}. Reduis ces parametres dans config.py."
        )

    return liste_fenetres


def preparer_fenetre(panel, fenetre, predicteurs, macro_predicteurs, cible,
                     mois_embargo=None):
    """Isole les lignes de train/validation/test d'une fenetre, et standardise
    les predicteurs macro sur cette fenetre.

    ⚠️ Point cle (fuite de donnees) : la moyenne et l'ecart-type de chaque
    predicteur macro sont recalcules sur le TRAIN DE CETTE FENETRE UNIQUEMENT,
    jamais sur sa validation ni son test : chaque fenetre a un train different,
    donc chacune a besoin de sa PROPRE standardisation pour ne jamais laisser
    filtrer une information du futur (voir aussi notebook 03, partie B, sur le
    meme principe de standardisation train-only).

    Les caracteristiques d'entreprise, elles, n'ont besoin d'aucun retraitement
    ici : le rank transform du notebook 03 (partie B) est deja calcule mois par
    mois, sans aucune fuite -- il reste valable tel quel, quelle que soit la
    fenetre.

    `mois_embargo` : nombre de mois retires a la FIN du train et de la validation (None =
    MOIS_EMBARGO_PAR_DEFAUT, soit 0 et donc aucun effet dans le pipeline d'origine). Voir
    l'en-tete du module : ce reglage n'existe que pour l'horizon de prediction long.

    Retourne un dict avec X_train/y_train, X_validation/y_validation,
    X_test/y_test, et id_test (permno + annee_mois du test, necessaire au
    notebook 07 pour reconstruire des portefeuilles par decile mois par mois).
    """
    mois_embargo = MOIS_EMBARGO_PAR_DEFAUT if mois_embargo is None else mois_embargo

    # ⚠️ Embargo : avec une cible longue, les derniers mois du train et de la validation
    # ont un horizon qui deborde sur le bloc SUIVANT -- il faut les retirer, sinon le
    # modele s'entraine (et selectionne ses hyperparametres) sur de l'information du futur.
    # A mois_embargo = 0 (defaut, horizon 1 mois), ces trois lignes sont sans effet.
    mois_train = appliquer_embargo(fenetre['train'], mois_embargo)
    mois_validation = appliquer_embargo(fenetre['validation'], mois_embargo)
    mois_test = fenetre['test']

    if not mois_train or not mois_validation:
        raise ValueError(
            f"Fenetre {fenetre.get('numero')} : l'embargo de {mois_embargo} mois vide le "
            "train ou la validation. Augmente ANNEES_TRAIN_INITIAL / ANNEES_VALIDATION "
            "dans config.py, ou reduis HORIZON_PREDICTION_MOIS."
        )

    train = panel[panel['annee_mois'].isin(mois_train)].copy()
    validation = panel[panel['annee_mois'].isin(mois_validation)].copy()
    test = panel[panel['annee_mois'].isin(mois_test)].copy()

    moyennes_train = train[macro_predicteurs].mean()
    ecarts_types_train = train[macro_predicteurs].std()

    for jeu in (train, validation, test):
        for col in macro_predicteurs:
            ecart = ecarts_types_train[col]
            jeu[col] = (jeu[col] - moyennes_train[col]) / ecart if ecart > 0 else 0.0

    return {
        'X_train': train[predicteurs], 'y_train': train[cible],
        'X_validation': validation[predicteurs], 'y_validation': validation[cible],
        'X_test': test[predicteurs], 'y_test': test[cible],
        'id_test': test[['permno', 'annee_mois']].reset_index(drop=True),
    }


def resumer_fenetres(liste_fenetres):
    """Petit tableau recapitulatif (une ligne par fenetre) -- pratique pour un
    print() rapide en debut de notebook 04/05/06, avant de lancer les boucles
    d'entrainement (qui peuvent etre longues, surtout pour LightGBM)."""
    lignes = []
    for f in liste_fenetres:
        lignes.append({
            'fenetre': f['numero'],
            'train_debut': f['train'][0], 'train_fin': f['train'][-1], 'n_mois_train': len(f['train']),
            'n_annees_train': int(f['train'][-1][:4]) - int(f['train'][0][:4]) + 1,
            'validation_debut': f['validation'][0], 'validation_fin': f['validation'][-1],
            'n_annees_validation': f.get('annees_validation'),
            'test_debut': f['test'][0], 'test_fin': f['test'][-1],
            'annee_test': f['annee_test'],
        })
    return pd.DataFrame(lignes)


def restreindre_debut_entrainement(panel, annee_debut_entrainement, colonne='annee_mois'):
    """Ne garde que les lignes a partir de ANNEE_DEBUT_ENTRAINEMENT (config.py).

    ⚠️ A ne pas confondre avec ANNEE_DEBUT, qui filtre la base des l'etape 02 : ici, les
    donnees anterieures existent toujours dans panel_pret_modelisation.parquet, elles sont
    simplement ignorees pour construire les fenetres et entrainer les modeles. Retarder
    cette date se teste donc en relancant seulement 04/05/06, sans repasser par 02 ni 03.

    Utilise par les 3 scripts de modelisation (etape04, etape05, etape06) : la meme
    restriction pour les 3, sinon leurs R2_oos ne seraient plus comparables entre eux.
    """
    annees = panel[colonne].astype(str).str[:4].astype(int)
    avant = len(panel)
    panel = panel[annees >= annee_debut_entrainement].copy()

    if panel.empty:
        raise ValueError(
            f"Aucune ligne restante apres ANNEE_DEBUT_ENTRAINEMENT={annee_debut_entrainement} : "
            "cette annee est posterieure a la fin du panel. Corrige la valeur dans config.py."
        )

    print(f"Debut d'entrainement fixe a {annee_debut_entrainement} : {avant} -> {len(panel)} lignes "
          f"({(avant - len(panel)) / avant * 100:.2f}% ignorees)")
    return panel
