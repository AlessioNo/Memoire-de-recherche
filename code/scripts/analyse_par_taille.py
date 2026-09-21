"""
ETAPE 10 -- Analyse par segment de taille : ou le modele fonctionne-t-il ?

Ce script RE-EVALUE les predictions DEJA sauvegardees par les etapes 04 a 07
(outputs/predictions_*.parquet), separement sur chaque segment de capitalisation boursiere
(small caps / large caps, ou terciles / quintiles selon config.MODE_GROUPES_TAILLE).

⚠️ AUCUN ré-entrainement, AUCUNE modification des etapes 04 a 07 : les 4 modeles restent
entraines et evalues sur l'univers COMPLET, exactement comme avant. Seule l'EVALUATION est
decoupee ici. C'est le protocole de Gu, Kelly & Xiu (2020) : un modele unique, dont on
regarde ensuite la performance sous-univers par sous-univers.

Consequence directe : cette etape ne cree AUCUNE nouvelle experience au journal
(journal.py), et ne touche a AUCUN fichier lu par les notebooks 04 a 09. C'est une analyse
"a cote" du pipeline principal, d'ou son notebook dedie (10) plutot qu'une section ajoutee
au notebook 09.

Lancement, depuis la RACINE du projet :

    python scripts/analyse_par_taille.py

⚠️ Pre-requis :
  - `python scripts/construction_panel.py` relance APRES l'ajout de la section
    B.2bis (c'est elle qui ecrit les colonnes `mvel1_brut` et `groupe_taille` dans
    data/processed/panel_pret_modelisation.parquet) ;
  - au moins un des scripts etape04 a etape07 deja lance (leurs predictions sur disque).

Parametres, tous dans config.py, section "Analyse par segment de TAILLE" :
MODE_GROUPES_TAILLE, MULTIPLICATEUR_MVEL1_EN_DOLLARS, NB_DECILES.

Fichiers produits (chemins definis dans config.py) :
  - outputs/analyse_taille_descriptif_groupes.parquet   (capitalisations + effectifs)
  - outputs/analyse_taille_bornes_mensuelles.parquet    (evolution mois par mois)
  - outputs/analyse_taille_performance.parquet          (R2_oos, rank-IC, Sharpe... )
  - outputs/analyse_taille_rendement_par_decile.parquet
  - outputs/analyse_taille_rendements_long_short.parquet
  - outputs/analyse_taille_ic_mensuel.parquet
  - le rapport '10_taille' (outputs/rapports/) pour le notebook 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
import fenetres
import portefeuilles
import rapports


# UNE seule liste a maintenir, comme au notebook 08 : ajouter un 5e modele au projet =
# ajouter une ligne ici. Le nom doit etre EXACTEMENT celui de la constante MODELE du script
# correspondant (etape04 a etape07).
MODELES = [
    ('Regression lineaire', config.FICHIER_PREDICTIONS_REGRESSION_LINEAIRE),
    ('Elastic Net', config.FICHIER_PREDICTIONS_ELASTIC_NET),
    ('LightGBM', config.FICHIER_PREDICTIONS_LIGHTGBM),
    ('Random Forest', config.FICHIER_PREDICTIONS_RANDOM_FOREST),
]


# ============================================================
# Section 0 -- Le decoupage : lecture de config.py, puis assignation
#
# ⚠️ C'est ICI que les groupes sont definis ET calcules, et nulle part ailleurs. L'etape 03
# ne conserve que la capitalisation brute (`mvel1_brut`) ; le decoupage n'en est qu'une
# derivation, recalculee a chaque execution de cette etape. Consequence pratique : changer
# config.MODE_GROUPES_TAILLE n'oblige PAS a relancer l'etape 03, et il ne peut pas y avoir
# de desynchronisation entre un decoupage fige dans le panel et celui de config.py.
#
# config.py ne contient donc que des VALEURS (le mode, les seuils, les noms) ; la traduction
# de ces valeurs en decoupage exploitable, et sa validation, sont de la logique, et vivent
# ici -- meme principe que fenetres.py, journal.py ou portefeuilles.py.
# ============================================================

def definition_groupes_taille():
    """Traduit config.MODE_GROUPES_TAILLE en une definition exploitable du decoupage.

    Retourne
    --------
    dict avec les cles :
      'type_seuils'   : 'percentile' (seuils relatifs, recalcules chaque mois) ou
                        'dollars' (seuils absolus, fixes sur toute la periode)
      'seuils'        : liste CROISSANTE des coupures (percentiles dans [0,1], ou montants
                        en dollars)
      'noms'          : liste des len(seuils) + 1 noms, du PLUS PETIT au PLUS GRAND.
                        Un None signale une tranche EXCLUE de l'analyse (zone tampon).
      'noms_analyses' : les noms hors None, c'est-a-dire les groupes reellement analyses.
    """
    def _valider(seuils, noms, type_seuils):
        seuils = list(seuils)
        noms = list(noms)
        if len(noms) != len(seuils) + 1:
            raise ValueError(
                f"Le decoupage '{config.MODE_GROUPES_TAILLE}' a {len(seuils)} seuil(s) et "
                f"{len(noms)} nom(s) : il en faut exactement {len(seuils) + 1}."
            )
        if list(seuils) != sorted(seuils) or len(set(seuils)) != len(seuils):
            raise ValueError(f"Les seuils doivent etre STRICTEMENT croissants (recu : {seuils}).")
        if type_seuils == 'percentile' and any(s <= 0 or s >= 1 for s in seuils):
            raise ValueError(
                f"Les percentiles doivent etre strictement entre 0 et 1 (recu : {seuils}).")
        if type_seuils == 'dollars' and any(s <= 0 for s in seuils):
            raise ValueError(
                f"Les seuils en dollars doivent etre strictement positifs (recu : {seuils}).")
        noms_analyses = [n for n in noms if n is not None]
        if len(noms_analyses) < 2:
            raise ValueError(
                f"Il faut au moins 2 groupes analyses (hors None) pour une comparaison ; "
                f"recu : {noms}.")
        if len(set(noms_analyses)) != len(noms_analyses):
            raise ValueError(f"Deux groupes portent le meme nom : {noms}.")
        if config.NOM_GROUPE_UNIVERS_COMPLET in noms_analyses:
            raise ValueError(
                f"Un groupe ne peut pas s'appeler {config.NOM_GROUPE_UNIVERS_COMPLET!r} : "
                "ce nom est reserve a la ligne de reference (univers entier).")
        return {'type_seuils': type_seuils, 'seuils': seuils, 'noms': noms,
                'noms_analyses': noms_analyses}

    mode = config.MODE_GROUPES_TAILLE
    if mode == 'mediane':
        return _valider([0.5], ['Small', 'Large'], 'percentile')
    if mode == 'terciles':
        return _valider([1 / 3, 2 / 3], ['Small', 'Mid', 'Large'], 'percentile')
    if mode == 'quintiles':
        return _valider([0.2, 0.4, 0.6, 0.8],
                        ['Q1 (small)', 'Q2', 'Q3', 'Q4', 'Q5 (large)'], 'percentile')
    if mode == 'personnalise':
        return _valider(config.SEUILS_PERCENTILES_GROUPES_TAILLE_PERSONNALISES,
                        config.NOMS_GROUPES_TAILLE_PERSONNALISES, 'percentile')
    if mode == 'dollars':
        return _valider(config.SEUILS_DOLLARS_GROUPES_TAILLE,
                        config.NOMS_GROUPES_TAILLE_DOLLARS, 'dollars')
    raise ValueError(
        f"config.MODE_GROUPES_TAILLE inconnu : {mode!r} (attendu 'mediane', 'terciles', "
        "'quintiles', 'personnalise' ou 'dollars')."
    )


def assigner_groupes_taille(donnees, colonne_capitalisation, definition,
                            colonne_mois='annee_mois'):
    """Assigne un groupe de taille a chaque titre-mois.

    Deux regimes, selon `definition['type_seuils']` :

    - 'percentile' : les coupures sont recalculees MOIS PAR MOIS sur la population de ce
      mois-la. Indispensable pour que le decoupage n'utilise aucune information du futur :
      une entreprise doit pouvoir changer de groupe en grandissant, et la classer selon sa
      taille moyenne ou finale reviendrait a savoir des 1990 qu'elle deviendra une grande
      capitalisation.

    - 'dollars' : les coupures sont des montants FIXES, identiques a toutes les dates. Pas
      de fuite d'information non plus (un seuil fixe ne depend d'aucune donnee), mais les
      groupes se vident et se remplissent au fil du temps sous l'effet de l'inflation et de
      la croissance des marches -- voir l'avertissement de config.py.

    Un nom valant None dans `definition['noms']` designe une ZONE TAMPON : les titres-mois
    correspondants recoivent NaN et sont exclus de l'analyse par groupe (ils restent
    comptes dans la ligne de reference "univers complet").

    Retourne une Series alignee sur `donnees` (NaN = hors analyse).
    """
    seuils = definition['seuils']
    noms = definition['noms']
    capitalisation = donnees[colonne_capitalisation]

    if definition['type_seuils'] == 'percentile':
        coupures = [
            donnees.groupby(colonne_mois, observed=True)[colonne_capitalisation]
                   .transform(lambda x, s=s: x.quantile(s))
            for s in seuils
        ]
    else:  # 'dollars' : la meme valeur a toutes les dates
        coupures = [pd.Series(float(s), index=donnees.index) for s in seuils]

    # np.select n'accepte pas None comme valeur : on passe par un jeton, converti en NaN.
    JETON_EXCLU = '\x00exclu'
    etiquettes = [n if n is not None else JETON_EXCLU for n in noms]

    conditions = [capitalisation <= coupure for coupure in coupures]
    groupe = pd.Series(
        np.select(conditions, etiquettes[:-1], default=etiquettes[-1]),
        index=donnees.index, dtype=object,
    )
    return groupe.where(groupe != JETON_EXCLU, other=np.nan)


# ============================================================
# Section 1 -- Chargement : predictions + colonnes de taille du panel
# ============================================================

def charger_predictions_et_taille(rap, definition):
    """Charge les predictions de test des modeles disponibles (format LONG : une ligne par
    (modele, permno, annee_mois)), y rattache la capitalisation brute du panel, puis
    ASSIGNE les groupes de taille.

    Le rattachement se fait en 'inner' sur (permno, annee_mois) : on ne garde donc que les
    titres-mois presents A LA FOIS dans les predictions et dans le panel -- ce qui doit
    etre 100% des predictions, puisque celles-ci en sont issues. Un ecart signalerait que
    le panel a ete regenere avec d'autres filtres depuis le dernier entrainement, ce qu'on
    verifie et affiche explicitement.

    ⚠️ Les groupes sont calcules sur la POPULATION EVALUEE (les titres-mois presents dans
    les predictions), et non sur le panel entier. C'est le bon choix : les percentiles
    doivent decrire l'univers sur lequel porte reellement l'analyse. Comme les predictions
    couvrent tous les titres du panel sur la periode de test, les deux coincident de toute
    facon a la periode de test pres.
    """
    cible = config.CIBLE
    colonne_groupe = config.COLONNE_GROUPE_TAILLE
    colonne_mvel1 = config.COLONNE_MVEL1_BRUT
    noms_analyses = definition['noms_analyses']

    # --- Panel : uniquement les 3 colonnes utiles (le panel complet est volumineux) ---
    panel = pd.read_parquet(
        config.FICHIER_PANEL_MODELISATION,
        columns=['permno', 'annee_mois', colonne_mvel1],
    )
    panel['annee_mois'] = panel['annee_mois'].astype(str)
    print(f"Panel (capitalisation brute) : {panel.shape}")

    # --- Predictions de chaque modele disponible ---
    tables = []
    modeles_absents = []
    for nom, chemin in MODELES:
        if not Path(chemin).exists():
            modeles_absents.append(nom)
            continue
        table = pd.read_parquet(chemin)[['permno', 'annee_mois', cible, 'prediction']].copy()
        table['annee_mois'] = table['annee_mois'].astype(str)
        table['modele'] = nom
        tables.append(table)
        print(f"{nom:22s} : {table.shape}, periode {table['annee_mois'].min()} "
              f"a {table['annee_mois'].max()}")

    if modeles_absents:
        print("\n⚠️ Modeles ignores (fichier de predictions absent) :", ", ".join(modeles_absents))
        print("   Lance le(s) script(s) correspondant(s) puis relance cette etape "
              "si tu veux les inclure.")
    if not tables:
        raise FileNotFoundError(
            "Aucun fichier outputs/predictions_*.parquet trouve : lance d'abord au moins un "
            "des scripts scripts/etape04 a scripts/etape07."
        )

    predictions = pd.concat(tables, ignore_index=True)
    n_avant = len(predictions)

    predictions = pd.merge(predictions, panel, on=['permno', 'annee_mois'], how='inner')
    n_apres = len(predictions)

    if n_apres < n_avant:
        print(f"\n⚠️ {n_avant - n_apres} lignes de predictions "
              f"({(n_avant - n_apres) / n_avant * 100:.2f}%) n'ont pas trouve de "
              "correspondance dans le panel.")
        print("   Cause la plus probable : le panel a ete regenere (etape 03) avec d'autres "
              "filtres DEPUIS le dernier entrainement. Relance 04 a 07 pour que predictions "
              "et panel decrivent le meme univers.")
    if predictions[colonne_mvel1].isna().any():
        raise ValueError(
            f"La colonne '{colonne_mvel1}' contient des valeurs manquantes : relance "
            "`python scripts/construction_panel.py --partie-b-seulement`."
        )

    # --- Capitalisation en dollars, puis assignation des groupes ---
    # ⚠️ La conversion doit preceder l'assignation : en mode 'dollars', les seuils de
    # config.py sont exprimes en dollars, donc compares a la capitalisation CONVERTIE.
    predictions['capitalisation_dollars'] = (
        predictions[colonne_mvel1] * config.MULTIPLICATEUR_MVEL1_EN_DOLLARS)
    predictions[colonne_groupe] = assigner_groupes_taille(
        predictions, 'capitalisation_dollars', definition)

    # ⚠️ `predictions` est au format LONG : un meme titre-mois y figure une fois PAR MODELE.
    # Les effectifs annonces doivent donc etre comptes sur les titres-mois UNIQUES, sinon
    # ils sont multiplies par le nombre de modeles et ne s'additionnent plus avec ceux du
    # tableau descriptif (qui, lui, deduplique).
    titres_mois = predictions.drop_duplicates(subset=['permno', 'annee_mois'])
    n_titres_mois = len(titres_mois)
    n_exclus = int(titres_mois[colonne_groupe].isna().sum())
    if n_exclus:
        print(f"\nZone tampon : {n_exclus} titres-mois "
              f"({n_exclus / n_titres_mois * 100:.1f}%) n'appartiennent a aucun groupe "
              "(un nom vaut None dans config.py) -- exclus de l'analyse par groupe, mais "
              "toujours comptes dans la ligne de reference 'univers complet'.")

    # Garde-fou : hors zone tampon, les groupes partitionnent l'univers evalue.
    # ℹ️ A ne pas confondre avec le nombre d'ENTREPRISES DISTINCTES, qui lui ne s'additionne
    # pas et n'a pas a le faire : le groupe etant reattribue chaque mois, une entreprise qui
    # grandit est comptee dans plusieurs groupes (voir avertissement 3 du notebook 10).
    n_dans_groupes = int(titres_mois[colonne_groupe].isin(noms_analyses).sum())
    if n_dans_groupes + n_exclus != n_titres_mois:
        raise ValueError(
            f"{n_titres_mois - n_dans_groupes - n_exclus} titres-mois n'appartiennent ni "
            f"a un groupe connu ({noms_analyses}) ni a la zone tampon : incoherence interne "
            "de l'assignation."
        )

    rap.valeur('modeles_evalues', sorted(predictions['modele'].unique().tolist()))
    rap.valeur('modeles_absents', modeles_absents)
    rap.valeur('n_lignes_predictions', int(n_apres))
    rap.valeur('pct_lignes_non_appariees', float((n_avant - n_apres) / n_avant * 100))
    rap.valeur('n_titres_mois_zone_tampon', n_exclus)
    rap.valeur('pct_titres_mois_zone_tampon', float(n_exclus / n_titres_mois * 100))
    rap.valeur('periode', [str(predictions['annee_mois'].min()), str(predictions['annee_mois'].max())])
    rap.valeur('n_mois', int(predictions['annee_mois'].nunique()))
    rap.valeur('n_entreprises', int(predictions['permno'].nunique()))
    rap.valeur('cible', cible)

    return predictions


# ============================================================
# Section 2 -- Descriptif des groupes : capitalisations et effectifs
#
# C'est cette section qui permet d'ecrire, dans le memoire, une phrase du type
# "le modele a ete evalue separement sur les entreprises dont la capitalisation depasse
#  X dollars, soit Y entreprises en moyenne chaque mois".
# ============================================================

def decrire_groupes(predictions, noms_groupes, rap):
    colonne_groupe = config.COLONNE_GROUPE_TAILLE
    colonne_mvel1 = config.COLONNE_MVEL1_BRUT

    # On travaille sur les titres-MOIS uniques (un titre observe par 4 modeles ne doit pas
    # etre compte 4 fois dans les statistiques de capitalisation).
    univers = (
        predictions[['permno', 'annee_mois', colonne_mvel1, 'capitalisation_dollars',
                     colonne_groupe]]
        .drop_duplicates(subset=['permno', 'annee_mois'])
        .copy()
    )

    # --- 2.a Descriptif global, un groupe par ligne (+ la ligne "univers complet") ---
    def _decrire(sous_ensemble, nom):
        capi = sous_ensemble['capitalisation_dollars']
        effectifs_mensuels = sous_ensemble.groupby('annee_mois', observed=True).size()
        return {
            'groupe': nom,
            'n_titres_mois': int(len(sous_ensemble)),
            'n_entreprises_distinctes': int(sous_ensemble['permno'].nunique()),
            'n_mois': int(sous_ensemble['annee_mois'].nunique()),
            'n_entreprises_moyen_par_mois': float(effectifs_mensuels.mean()),
            'n_entreprises_min_par_mois': int(effectifs_mensuels.min()),
            'n_entreprises_max_par_mois': int(effectifs_mensuels.max()),
            'capi_min': float(capi.min()),
            'capi_p25': float(capi.quantile(0.25)),
            'capi_mediane': float(capi.median()),
            'capi_moyenne': float(capi.mean()),
            'capi_p75': float(capi.quantile(0.75)),
            'capi_max': float(capi.max()),
            'part_capitalisation_totale_pct': float(
                capi.sum() / univers['capitalisation_dollars'].sum() * 100),
        }

    lignes = [_decrire(univers, config.NOM_GROUPE_UNIVERS_COMPLET)]
    for nom in noms_groupes:
        sous = univers[univers[colonne_groupe] == nom]
        if len(sous) == 0:
            print(f"⚠️ Groupe '{nom}' vide sur la periode de test : ignore.")
            continue
        lignes.append(_decrire(sous, nom))

    descriptif = pd.DataFrame(lignes).set_index('groupe')

    # --- 2.b Bornes mois par mois (l'inflation et la croissance des marches rendent tout
    #         seuil moyen sur 40 ans peu parlant : c'est l'evolution qu'il faut montrer) ---
    bornes = (
        univers.dropna(subset=[colonne_groupe])
        .groupby(['annee_mois', colonne_groupe], observed=True)['capitalisation_dollars']
        .agg(capi_min='min', capi_mediane='median', capi_max='max', n_entreprises='size')
        .reset_index()
        .rename(columns={colonne_groupe: 'groupe'})
        .sort_values(['annee_mois', 'groupe'])
    )

    dernier_mois = univers['annee_mois'].max()
    rap.valeur('multiplicateur_mvel1_en_dollars', float(config.MULTIPLICATEUR_MVEL1_EN_DOLLARS))
    rap.valeur('dernier_mois', str(dernier_mois))
    rap.table('descriptif_groupes', descriptif)
    rap.table('bornes_mensuelles', bornes)
    rap.table('apercu_mvel1_brut',
              univers[['permno', 'annee_mois', colonne_mvel1, 'capitalisation_dollars',
                       colonne_groupe]].sort_values('capitalisation_dollars', ascending=False).head(10))

    descriptif.to_parquet(config.FICHIER_TAILLE_DESCRIPTIF)
    bornes.to_parquet(config.FICHIER_TAILLE_BORNES_MENSUELLES, index=False)
    print("\nDescriptif des groupes sauvegarde :", config.FICHIER_TAILLE_DESCRIPTIF)
    print("Bornes mensuelles sauvegardees   :", config.FICHIER_TAILLE_BORNES_MENSUELLES)
    print()
    print(descriptif[['n_entreprises_moyen_par_mois', 'capi_mediane',
                      'part_capitalisation_totale_pct']].round(1))

    return descriptif, bornes


# ============================================================
# Section 3 -- Evaluation : R2_oos, rank-IC et portefeuilles, par (modele x groupe)
# ============================================================

def evaluer_par_groupe(predictions, noms_groupes, rap):
    cible = config.CIBLE
    colonne_groupe = config.COLONNE_GROUPE_TAILLE
    nb_deciles = config.NB_DECILES

    # L'univers complet d'abord : c'est la ligne de reference, et elle doit reproduire
    # EXACTEMENT les chiffres du notebook 08 (memes predictions, memes deciles, memes
    # mesures) -- si ce n'est pas le cas, c'est qu'un filtre a bouge quelque part.
    groupes_a_evaluer = [(config.NOM_GROUPE_UNIVERS_COMPLET, None)] + [(n, n) for n in noms_groupes]

    lignes_performance = []
    lignes_deciles = []
    lignes_ls = []
    lignes_ic = []

    for nom_modele, sous_predictions in predictions.groupby('modele', sort=True):
        for nom_groupe, filtre in groupes_a_evaluer:
            donnees = (sous_predictions if filtre is None
                       else sous_predictions[sous_predictions[colonne_groupe] == filtre])
            if len(donnees) == 0:
                continue

            resultat = portefeuilles.evaluer_sous_univers(
                donnees, colonne_prediction='prediction', colonne_cible=cible,
                nb_deciles=nb_deciles,
            )

            # ⚠️ Un mois comptant moins de `nb_deciles` titres dans ce groupe ne permet pas
            # de former les deciles : il est ecarte du portefeuille (voir
            # portefeuilles.assigner_decile). Cas frequent en mode 'dollars', ou un groupe
            # peut etre quasi vide en debut ou en fin de periode. On le chiffre plutot que
            # de le laisser passer inapercu.
            n_mois_groupe = int(donnees['annee_mois'].nunique())
            n_mois_portefeuille = int(len(resultat['rendement_long_short'].dropna()))

            ligne = {'modele': nom_modele, 'groupe': nom_groupe}
            ligne['n_mois_disponibles'] = n_mois_groupe
            ligne['n_mois_sans_portefeuille'] = n_mois_groupe - n_mois_portefeuille
            # ⚠️ Le R2_oos est calcule ICI sur le sous-univers : son denominateur (somme des
            # rendements au carre) change donc d'un groupe a l'autre. Voir l'avertissement
            # du notebook 10 : il n'est PAS comparable entre groupes, contrairement au
            # rank-IC juste a cote.
            ligne['r2_oos'] = float(fenetres.r2_oos(donnees[cible], donnees['prediction']))
            ligne['n_titres_mois'] = int(len(donnees))
            ligne['n_entreprises'] = int(donnees['permno'].nunique())
            ligne.update(resultat['metriques'])
            lignes_performance.append(ligne)

            moyenne_deciles = resultat['rendements_deciles'].mean()
            for decile, valeur in moyenne_deciles.items():
                lignes_deciles.append({'modele': nom_modele, 'groupe': nom_groupe,
                                       'decile': int(decile), 'rendement_moyen': float(valeur)})

            ls = resultat['rendement_long_short']
            lignes_ls.append(pd.DataFrame({
                'annee_mois': ls.index.astype(str), 'modele': nom_modele,
                'groupe': nom_groupe, 'rendement_long_short': ls.values,
            }))

            ic = resultat['ic_mensuel']
            lignes_ic.append(pd.DataFrame({
                'annee_mois': ic.index.astype(str), 'modele': nom_modele,
                'groupe': nom_groupe, 'rank_ic': ic.values,
            }))

            if ligne['n_mois_sans_portefeuille'] > 0:
                print(f"  ⚠️ {nom_modele} / {nom_groupe} : "
                      f"{ligne['n_mois_sans_portefeuille']} mois sur {n_mois_groupe} ecartes "
                      f"(moins de {nb_deciles} titres dans le groupe ce mois-la)")

            print(f"{nom_modele:22s} | {nom_groupe:18s} : "
                  f"R2_oos = {ligne['r2_oos']:+.4f} | "
                  f"rank-IC = {ligne['ic_moyen']:+.4f} (t = {ligne['ic_t_stat']:+.2f}) | "
                  f"Sharpe = {ligne['sharpe_ratio']:+.3f}")

    performance = pd.DataFrame(lignes_performance)
    rendement_par_decile = pd.DataFrame(lignes_deciles)
    rendements_ls = pd.concat(lignes_ls, ignore_index=True)
    ic_mensuel = pd.concat(lignes_ic, ignore_index=True)

    # Ordre d'affichage : univers complet en premier, puis du plus petit au plus grand
    ordre = [config.NOM_GROUPE_UNIVERS_COMPLET] + list(noms_groupes)
    performance['groupe'] = pd.Categorical(performance['groupe'], categories=ordre, ordered=True)
    performance = performance.sort_values(['modele', 'groupe']).reset_index(drop=True)
    performance['groupe'] = performance['groupe'].astype(str)

    performance.to_parquet(config.FICHIER_TAILLE_PERFORMANCE, index=False)
    rendement_par_decile.to_parquet(config.FICHIER_TAILLE_DECILES, index=False)
    rendements_ls.to_parquet(config.FICHIER_TAILLE_RENDEMENTS_LS, index=False)
    ic_mensuel.to_parquet(config.FICHIER_TAILLE_IC_MENSUEL, index=False)

    print("\nPerformance par (modele x groupe) sauvegardee :", config.FICHIER_TAILLE_PERFORMANCE)
    print("Rendement par decile sauvegarde              :", config.FICHIER_TAILLE_DECILES)
    print("Rendements long-short mensuels sauvegardes   :", config.FICHIER_TAILLE_RENDEMENTS_LS)
    print("Rank-IC mensuel sauvegarde                   :", config.FICHIER_TAILLE_IC_MENSUEL)

    rap.table('performance', performance)
    rap.table('rendement_par_decile', rendement_par_decile)
    rap.table('rendements_long_short', rendements_ls)
    rap.table('ic_mensuel', ic_mensuel)

    return performance

NOM_BENCHMARK_EQUIPONDERE = "Univers equipondere"
NOM_BENCHMARK_CAPI = "Univers pondere capi"


def evaluer_benchmarks_par_groupe(predictions, noms_groupes, rap):
    """Deux repères LONG ONLY calcules A L'INTERIEUR de chaque groupe de taille.

    Ils n'utilisent AUCUNE prediction : c'est le rendement du groupe lui-meme, une fois
    equipondere, une fois pondere par la capitalisation. Ils repondent a la question
    "qu'aurait rapporte le fait de detenir simplement toutes les small caps ?", et donnent
    donc l'echelle a laquelle lire les long-short des modeles dans ce groupe.

    ⚠️ Ce sont des portefeuilles LONG ONLY : ils portent l'exposition au marche que le
    long-short neutralise. Ils ne se comparent pas terme a terme aux portefeuilles des
    modeles -- voir les avertissements du notebook 10.
    """
    cible = config.CIBLE
    colonne_groupe = config.COLONNE_GROUPE_TAILLE
    colonne_mvel1 = config.COLONNE_MVEL1_BRUT

    # ⚠️ `predictions` est au format LONG : un meme titre-mois y figure une fois PAR MODELE.
    # Les benchmarks ne dependent d'aucun modele -> on deduplique, sinon chaque titre
    # pesera autant de fois qu'il y a de modeles (et le poids capi serait multiplie).
    univers = predictions.drop_duplicates(subset=['permno', 'annee_mois']).copy()

    groupes_a_evaluer = ([(config.NOM_GROUPE_UNIVERS_COMPLET, None)]
                         + [(n, n) for n in noms_groupes])

    lignes_performance, lignes_rendements = [], []

    for nom_groupe, filtre in groupes_a_evaluer:
        donnees = univers if filtre is None else univers[univers[colonne_groupe] == filtre]
        if len(donnees) == 0:
            continue

        equipondere = donnees.groupby('annee_mois')[cible].mean().sort_index()

        poids = donnees[colonne_mvel1]
        mois = donnees['annee_mois']
        capi = ((donnees[cible] * poids).groupby(mois).sum()
                / poids.groupby(mois).sum()).sort_index()

        for nom_benchmark, serie in [(NOM_BENCHMARK_EQUIPONDERE, equipondere),
                                     (NOM_BENCHMARK_CAPI, capi)]:
            ligne = {'benchmark': nom_benchmark, 'groupe': nom_groupe}
            ligne.update(portefeuilles.calculer_metriques(serie))
            lignes_performance.append(ligne)
            lignes_rendements.append(pd.DataFrame({
                'annee_mois': serie.index.astype(str),
                'benchmark': nom_benchmark,
                'groupe': nom_groupe,
                'rendement': serie.values,
            }))

    benchmarks = pd.DataFrame(lignes_performance)
    rendements_benchmarks = pd.concat(lignes_rendements, ignore_index=True)

    rap.table('benchmarks_performance', benchmarks)
    rap.table('rendements_benchmarks', rendements_benchmarks)

    print(f"\nBenchmarks calcules : {len(benchmarks)} lignes "
          f"({len(groupes_a_evaluer)} groupes x 2 constructions, long only).")
    return benchmarks

# ============================================================
def main():
    config.assurer_dossiers()
    rapports.assurer_dossier()
    rap = rapports.Rapport('10_taille')

    definition = definition_groupes_taille()
    noms_groupes = definition['noms_analyses']

    print("=" * 70)
    print(f"ETAPE 10 -- Analyse par segment de taille ({config.MODE_GROUPES_TAILLE})")
    print("=" * 70)
    print(f"Groupes analyses : {noms_groupes}")
    if definition['type_seuils'] == 'percentile':
        print("Coupures : percentiles recalcules CHAQUE MOIS -> "
              + ", ".join(f"{s:.1%}" for s in definition['seuils']))
    else:
        print("Coupures : seuils en dollars FIXES sur toute la periode -> "
              + ", ".join(f"{s:,.0f} $".replace(',', ' ') for s in definition['seuils']))
        print("⚠️ Un seuil fixe n'est pas comparable d'un bout a l'autre de l'echantillon "
              "(inflation, croissance des marches) :")
        print("   verifie les effectifs mensuels par groupe (section 2 du notebook 10) "
              "avant de conclure.")
    if any(n is None for n in definition['noms']):
        tranches = []
        bornes_lisibles = [None] + list(definition['seuils']) + [None]
        for i, nom in enumerate(definition['noms']):
            if nom is None:
                bas, haut = bornes_lisibles[i], bornes_lisibles[i + 1]
                tranches.append(f"entre {bas:,.0f} $ et {haut:,.0f} $".replace(',', ' ')
                                if bas is not None and haut is not None else "une tranche")
        print(f"Zone tampon EXCLUE de l'analyse : {', '.join(tranches)}")
    print(f"Nombre de deciles par portefeuille : {config.NB_DECILES}")
    print()

    rap.valeur('mode_groupes_taille', config.MODE_GROUPES_TAILLE)
    rap.valeur('type_seuils', definition['type_seuils'])
    rap.valeur('seuils', [float(s) for s in definition['seuils']])
    rap.valeur('noms_groupes_avec_zone_tampon',
               ['(zone tampon exclue)' if n is None else n for n in definition['noms']])
    rap.valeur('noms_groupes', list(noms_groupes))
    rap.valeur('nom_groupe_univers_complet', config.NOM_GROUPE_UNIVERS_COMPLET)
    rap.valeur('nb_deciles', config.NB_DECILES)
    rap.valeur('seuil_percentile_taille_univers', config.SEUIL_PERCENTILE_TAILLE)
    rap.valeur('seuil_percentile_liquidite_univers', config.SEUIL_PERCENTILE_LIQUIDITE)

    predictions = charger_predictions_et_taille(rap, definition)
    decrire_groupes(predictions, noms_groupes, rap)
    evaluer_par_groupe(predictions, noms_groupes, rap)
    evaluer_benchmarks_par_groupe(predictions, noms_groupes, rap)


    rap.sauvegarder()

    print("\n" + "=" * 70)
    print("ETAPE 10 TERMINEE.")
    print("  -> ouvre notebooks/10_analyse_par_taille.ipynb pour visualiser les resultats")
    print("=" * 70)



if __name__ == "__main__":
    main()
