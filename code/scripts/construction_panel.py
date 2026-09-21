"""
ETAPE 03 -- Construction du panel final (ex-notebook 03).

Ce script contient TOUT le calcul qui se trouvait auparavant dans les cellules du
notebook `03_construction_panel.ipynb`. Le notebook ne fait plus que LIRE et AFFICHER
ce que ce script a produit (voir rapports.py).

Lancement, depuis la RACINE du projet :

    python scripts/construction_panel.py

    # ou, pour ne rejouer QUE la partie B (filtres/imputation/rank transform) a partir
    # de data/processed/panel_final.parquet, sans refaire la fusion (partie A) :
    python scripts/construction_panel.py --partie-b-seulement

Deux parties SEQUENTIELLES (contrairement a l'etape 02) :
  A -- Fusion des 3 fichiers de data/interim/ + cible excess_return
       -> data/processed/panel_final.parquet
  B -- Preparation pour la modelisation (filtres taille/liquidite, imputation,
       winsorizing, rank transform)
       -> data/processed/panel_pret_modelisation.parquet

⚠️ Pre-requis : avoir deja lance `python scripts/nettoyage_donnees.py`.

Parametres, tous dans config.py : SEUIL_PERCENTILE_TAILLE, SEUIL_PERCENTILE_LIQUIDITE,
(+ CARACTERISTIQUES_RETENUES, lu automatiquement dans le manifeste
ecrit par l'etape 02).

La partie B ajoute au passage une colonne purement DESCRIPTIVE au panel (section B.2bis),
utilisee par la seule etape 10 :
  - `mvel1_brut` : la capitalisation boursiere AVANT winsorizing et rank transform (apres
                   quoi mvel1 vit dans [-1, 1] et n'est plus interpretable en dollars)
⚠️ Ce n'est JAMAIS un predicteur : les etapes 04 a 07 selectionnent explicitement
config.PREDICTEURS, elles ignorent donc cette colonne. L'ajouter ne change aucun resultat
de modelisation.

Ni decoupage temporel ni standardisation macro ici : les deux dependent de la fenetre
en cours et sont recalcules a la volee par fenetres.py (etapes 04 a 06).
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
import horizon
import rapports


# ============================================================
# Partie A -- Fusion des 3 bases nettoyees
# ============================================================

def mois_suivant(annee_mois):
    """Renvoie le mois suivant au format AAAAMM (texte).

    Sert a decaler les predicteurs macro (Welch-Goyal, PAS pre-decales contrairement
    a datashare.parquet) : un predicteur connu au mois m doit servir a predire le
    rendement du mois m+1, jamais celui du mois m lui-meme.
    """
    annee = int(annee_mois[:4])
    mois = int(annee_mois[4:6])
    mois += 1
    if mois > 12:
        mois = 1
        annee += 1
    return f"{annee:04d}{mois:02d}"


def construire_panel_fusionne(rap):
    print("\n" + "=" * 70)
    print("PARTIE A -- Fusion des 3 bases nettoyees")
    print("=" * 70)

    # --- A.1 Chargement ---
    chars = pd.read_parquet(config.FICHIER_CARACTERISTIQUES_CLEAN)
    returns = pd.read_parquet(config.FICHIER_RETURNS_CLEAN)
    macro = pd.read_parquet(config.FICHIER_MACRO_CLEAN)

    print("Caracteristiques :", chars.shape)
    print("Rendements       :", returns.shape)
    print("Macro            :", macro.shape)
    rap.valeur('A_shape_chars', list(chars.shape))
    rap.valeur('A_shape_returns', list(returns.shape))
    rap.valeur('A_shape_macro', list(macro.shape))

    # --- A.2 Cle de fusion mensuelle (aucun decalage : datashare est deja pre-decale) ---
    chars['annee_mois'] = chars['DATE'].astype(str).str[:6]

    # --- A.3 Chevauchement AVANT de fusionner ---
    permnos_chars = set(chars['permno'].unique())
    permnos_returns = set(returns['permno'].unique())
    communs = permnos_chars & permnos_returns

    rap.valeur('A_n_entreprises_chars', len(permnos_chars))
    rap.valeur('A_n_entreprises_returns', len(permnos_returns))
    rap.valeur('A_n_entreprises_communes', len(communs))
    rap.valeur('A_pct_entreprises_communes',
               float(len(communs) / max(len(permnos_chars), 1) * 100))
    print(f"Entreprises communes : {len(communs)} "
          f"({len(communs) / max(len(permnos_chars), 1) * 100:.1f}% de chars)")

    periodes_chars = set(chars['annee_mois'])
    periodes_returns = set(returns['annee_mois'])
    periodes_macro = set(macro['annee_mois'])
    rap.valeur('A_periodes_chars', [len(periodes_chars), min(periodes_chars, default='-'), max(periodes_chars, default='-')])
    rap.valeur('A_periodes_returns', [len(periodes_returns), min(periodes_returns, default='-'), max(periodes_returns, default='-')])
    rap.valeur('A_periodes_macro', [len(periodes_macro), min(periodes_macro, default='-'), max(periodes_macro, default='-')])

    # --- A.4 Fusion 1 : caracteristiques + rendements ---
    avant_chars, avant_returns = len(chars), len(returns)
    panel = pd.merge(
        chars, returns,
        on=['permno', 'annee_mois'],
        how='inner',
        validate='one_to_one',
    )
    rap.valeur('A_lignes_chars_avant_fusion', int(avant_chars))
    rap.valeur('A_lignes_returns_avant_fusion', int(avant_returns))
    rap.valeur('A_lignes_apres_fusion1', int(len(panel)))
    print(f"Lignes apres fusion caracteristiques + rendements : {len(panel)}")

    # --- A.5 Fusion 2 : macro (predicteurs decales d'un mois, Rfree non decale) ---
    macro_predicteurs_decales = macro[['annee_mois'] + config.MACRO_PREDICTEURS].copy()
    macro_predicteurs_decales['annee_mois'] = macro_predicteurs_decales['annee_mois'].apply(mois_suivant)

    panel = pd.merge(panel, macro_predicteurs_decales, on='annee_mois', how='left')
    rap.valeur('A_lignes_apres_fusion_predicteurs_macro', int(len(panel)))

    rfree_avec_cle = macro[['annee_mois', 'Rfree']]
    panel = pd.merge(panel, rfree_avec_cle, on='annee_mois', how='left')
    rap.valeur('A_lignes_apres_fusion_rfree', int(len(panel)))

    # --- A.5bis Calcul de la cible composee sur HORIZON_PREDICTION_MOIS mois ---
    #
    # ⚠️ EMPLACEMENT CRITIQUE, a deux titres.
    # (a) Le calcul porte sur `returns`, l'historique de rendements COMPLET issu de l'etape
    #     02, et surtout PAS sur `panel` : la fusion A.4 est un inner join
    #     (caracteristiques ∩ rendements) et la partie B retirera encore des titres-mois.
    #     Composer sur `panel` reviendrait a enjamber silencieusement les mois que nos
    #     propres filtres ont retires, donc a calculer un rendement sur 12 mois NON
    #     CONSECUTIFS -- sans la moindre erreur ni le moindre avertissement.
    # (b) Ici, et pas plus bas : `returns` est libere juste apres. Seul le petit DataFrame
    #     de resultat est conserve, pour le rattachement en A.6bis.
    cible_longue = horizon.construire_cible_horizon(returns, macro, rap=rap)

    del chars, returns
    gc.collect()

    colonnes_macro_predicteurs = config.MACRO_PREDICTEURS + ['Rfree']
    manquant_macro = int(panel[colonnes_macro_predicteurs].isna().any(axis=1).sum())
    rap.valeur('A_lignes_sans_macro', manquant_macro)
    rap.valeur('A_pct_lignes_sans_macro', float(manquant_macro / len(panel) * 100))
    print(f"Lignes sans donnees macro correspondantes : {manquant_macro} "
          f"({manquant_macro / len(panel) * 100:.2f}%)")

    if manquant_macro > 0:
        panel = panel.dropna(subset=colonnes_macro_predicteurs)
        print(f"Lignes apres suppression : {len(panel)}")
    rap.valeur('A_lignes_apres_suppression_sans_macro', int(len(panel)))

    # --- Cible : rendement excedentaire ---
    panel['excess_return'] = panel['RET'] - panel['Rfree']
    rap.table('A_apercu_cible',
              panel[['permno', 'annee_mois', 'RET', 'Rfree', 'excess_return']].head())
    rap.table('A_describe_cible', panel['excess_return'].describe().rename('excess_return'))

    # --- A.5ter Rattachement de la cible longue (calculee en A.5bis) ---
    #
    # ⚠️ Cette colonne S'AJOUTE a `excess_return`, elle ne la remplace pas : les deux cibles
    # cohabitent dans le panel. Les etapes 04 a 07 continuent d'utiliser `excess_return` et
    # ignorent celle-ci ; les etapes 11 a 14 font l'inverse. Une seule construction de panel
    # suffit donc pour les deux pistes, et la piste a 1 mois n'est en rien affectee.
    nom_cible_longue = config.nom_cible_horizon()

    panel = panel.merge(cible_longue.drop(columns='statut_titre'),
                        on=['permno', 'annee_mois'], how='left')
    n_avec_cible = int(panel[nom_cible_longue].notna().sum())
    print(f"Cible {nom_cible_longue} rattachee au panel : {n_avec_cible} lignes "
          f"({n_avec_cible / len(panel) * 100:.1f} %) -- les autres ont un horizon "
          "incomplet (trou au milieu de l'historique, ou censure de fin d'echantillon).")
    rap.valeur('A_nom_cible_longue', nom_cible_longue)
    rap.valeur('A_n_lignes_avec_cible_longue', n_avec_cible)
    rap.valeur('A_pct_lignes_avec_cible_longue', float(n_avec_cible / len(panel) * 100))
    rap.table('A_apercu_cible_longue',
              panel.loc[panel[nom_cible_longue].notna(),
                        ['permno', 'annee_mois', 'excess_return', nom_cible_longue,
                         'n_mois_horizon_observes']].head())

    del macro, macro_predicteurs_decales, rfree_avec_cle
    del permnos_chars, permnos_returns, communs, periodes_chars, periodes_returns, periodes_macro
    gc.collect()

    # --- A.6 Verifications finales ---
    doublons = int(panel.duplicated(subset=['permno', 'annee_mois']).sum())
    rap.valeur('A_doublons_restants', doublons)
    if doublons > 0:
        panel = panel.drop_duplicates(subset=['permno', 'annee_mois'], keep='first')
        print("Doublons retires (on garde la premiere occurrence).")

    rap.valeur('A_shape_panel_final', list(panel.shape))
    rap.valeur('A_n_entreprises_panel', int(panel['permno'].nunique()))
    rap.valeur('A_periode_panel', [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])
    manquants = panel.isna().sum()
    rap.table('A_missing_panel', manquants[manquants > 0].rename('nb_manquant')
              if manquants.sum() > 0 else pd.Series(dtype='int64', name='nb_manquant'))
    print("Dimensions finales du panel :", panel.shape)

    # --- A.6bis Optimisation memoire ---
    panel['annee_mois'] = panel['annee_mois'].astype('category').cat.as_ordered()
    panel['permno'] = panel['permno'].astype('category')
    rap.valeur('A_memoire_panel_mo', float(panel.memory_usage(deep=True).sum() / 1e6))

    # --- A.7 Sauvegarde ---
    panel.to_parquet(config.FICHIER_PANEL_FINAL, index=False)
    print("Fichier sauvegarde :", config.FICHIER_PANEL_FINAL)

    return panel


# ============================================================
# Partie B -- Preparation pour la modelisation
# ============================================================

def imputer_par_mediane_mois(serie):
    return serie.fillna(serie.median())


def winsorize_groupe(serie, borne_bas=0.01, borne_haut=0.99):
    """Coupe les valeurs extremes d'une serie a ses percentiles bas/haut.
    Si la serie est entierement vide (tout NaN), on la renvoie telle quelle."""
    if serie.notna().sum() == 0:
        return serie
    lo = serie.quantile(borne_bas)
    hi = serie.quantile(borne_haut)
    return serie.clip(lo, hi)


def rank_transform(serie):
    n = len(serie)
    if n <= 1:
        # un seul rang possible : on ne peut pas dire si l'entreprise est "haute" ou "basse"
        return pd.Series(0.0, index=serie.index)
    return 2 * (serie.rank(method='average') - 1) / (n - 1) - 1


def preparer_panel_modelisation(panel, rap):
    print("\n" + "=" * 70)
    print("PARTIE B -- Preparation pour la modelisation")
    print("=" * 70)

    caracteristiques = config.CARACTERISTIQUES_RETENUES
    macro_predicteurs = config.MACRO_PREDICTEURS
    cible = config.CIBLE

    colonnes_attendues = caracteristiques + macro_predicteurs + [cible]
    colonnes_manquantes = [c for c in colonnes_attendues if c not in panel.columns]
    rap.valeur('B_colonnes_manquantes', colonnes_manquantes)
    rap.valeur('B_n_caracteristiques', len(caracteristiques))
    rap.valeur('B_n_macro_predicteurs', len(macro_predicteurs))
    if colonnes_manquantes:
        print("ATTENTION, colonnes introuvables dans le panel :", colonnes_manquantes)

    # --- B.2.1 Filtre taille (mvel1), mois par mois ---
    avant_dropna_mvel1 = len(panel)
    entreprises_avant_dropna_mvel1 = panel['permno'].nunique()
    panel = panel.dropna(subset=['mvel1'])
    rap.valeur('B_lignes_avant_dropna_mvel1', int(avant_dropna_mvel1))
    rap.valeur('B_lignes_apres_dropna_mvel1', int(len(panel)))
    rap.valeur('B_entreprises_avant_dropna_mvel1', int(entreprises_avant_dropna_mvel1))
    rap.valeur('B_entreprises_apres_dropna_mvel1', int(panel['permno'].nunique()))

    SEUIL_PERCENTILE_TAILLE = config.SEUIL_PERCENTILE_TAILLE
    rap.valeur('B_seuil_percentile_taille', SEUIL_PERCENTILE_TAILLE)

    seuils_taille_mensuels = panel.groupby('annee_mois', observed=True)['mvel1'].transform(
        lambda x: x.quantile(SEUIL_PERCENTILE_TAILLE)
    )
    avant_filtre_taille = len(panel)
    entreprises_avant = panel['permno'].nunique()

    masque_micro_cap = panel['mvel1'] < seuils_taille_mensuels
    panel = panel[~masque_micro_cap]
    del seuils_taille_mensuels, masque_micro_cap
    gc.collect()

    rap.valeur('B_lignes_avant_filtre_taille', int(avant_filtre_taille))
    rap.valeur('B_lignes_apres_filtre_taille', int(len(panel)))
    rap.valeur('B_entreprises_avant_filtre_taille', int(entreprises_avant))
    rap.valeur('B_entreprises_apres_filtre_taille', int(panel['permno'].nunique()))
    print(f"Filtre taille : {avant_filtre_taille} -> {len(panel)} lignes "
          f"({(avant_filtre_taille - len(panel)) / avant_filtre_taille * 100:.2f}% retirees)")

    # --- B.2.2 Filtre liquidite (ill), mois par mois ---
    SEUIL_PERCENTILE_LIQUIDITE = config.SEUIL_PERCENTILE_LIQUIDITE
    rap.valeur('B_seuil_percentile_liquidite', SEUIL_PERCENTILE_LIQUIDITE)

    avant_dropna_ill = len(panel)
    entreprises_avant_dropna_ill = panel['permno'].nunique()
    panel = panel.dropna(subset=['ill'])
    rap.valeur('B_lignes_avant_dropna_ill', int(avant_dropna_ill))
    rap.valeur('B_lignes_apres_dropna_ill', int(len(panel)))
    rap.valeur('B_entreprises_avant_dropna_ill', int(entreprises_avant_dropna_ill))
    rap.valeur('B_entreprises_apres_dropna_ill', int(panel['permno'].nunique()))

    seuils_liquidite_mensuels = panel.groupby('annee_mois', observed=True)['ill'].transform(
        lambda x: x.quantile(SEUIL_PERCENTILE_LIQUIDITE)
    )
    avant_filtre_liquidite = len(panel)
    entreprises_avant_liquidite = panel['permno'].nunique()

    masque_illiquide = panel['ill'] > seuils_liquidite_mensuels
    panel = panel[~masque_illiquide]
    del seuils_liquidite_mensuels, masque_illiquide
    gc.collect()

    rap.valeur('B_lignes_avant_filtre_liquidite', int(avant_filtre_liquidite))
    rap.valeur('B_lignes_apres_filtre_liquidite', int(len(panel)))
    rap.valeur('B_entreprises_avant_filtre_liquidite', int(entreprises_avant_liquidite))
    rap.valeur('B_entreprises_apres_filtre_liquidite', int(panel['permno'].nunique()))
    rap.valeur('B_pct_retire_total_filtres',
               float((avant_dropna_mvel1 - len(panel)) / avant_dropna_mvel1 * 100))
    print(f"Filtre liquidite : {avant_filtre_liquidite} -> {len(panel)} lignes")
    print(f"Effet cumule des filtres : {avant_dropna_mvel1} -> {len(panel)} lignes "
          f"({(avant_dropna_mvel1 - len(panel)) / avant_dropna_mvel1 * 100:.2f}% retirees au total)")

    # --- B.2bis Capitalisation boursiere brute (pour l'etape 10) ---
    #
    # ⚠️ EMPLACEMENT CRITIQUE : ici, mvel1 est encore la capitalisation boursiere BRUTE
    # (en unites monetaires), et l'univers est deja celui qui servira reellement a la
    # modelisation (post-filtres taille et liquidite). Quelques lignes plus bas, le
    # winsorizing puis le rank transform ecrasent mvel1 dans [-1, 1] : la capitalisation
    # en dollars est alors DEFINITIVEMENT perdue, et il devient impossible de dire "les
    # entreprises de plus de X dollars". D'ou cette copie, la SEULE chose que l'etape 03
    # ait besoin de conserver pour l'etape 10.
    #
    # ℹ️ Le decoupage en segments de taille (mediane / terciles / seuils en dollars...) n'est
    # PAS calcule ici : ce n'est qu'une derivation de cette colonne, et l'etape 10 la
    # recalcule elle-meme a partir de config.MODE_GROUPES_TAILLE. Changer de decoupage
    # n'oblige donc PAS a relancer l'etape 03 -- et surtout, il ne peut pas y avoir de
    # desynchronisation entre un decoupage fige dans le panel et celui de config.py.
    #
    # Cette colonne n'est JAMAIS un predicteur : les etapes 04 a 07 selectionnent
    # explicitement config.PREDICTEURS, donc elles l'ignorent. L'ajouter au panel ne change
    # STRICTEMENT RIEN aux modeles deja entraines.
    colonne_mvel1_brut = config.COLONNE_MVEL1_BRUT
    panel[colonne_mvel1_brut] = panel['mvel1'].astype('float64')

    rap.table('B_capitalisation_brute', panel[colonne_mvel1_brut].describe())
    print(f"Capitalisation brute conservee dans '{colonne_mvel1_brut}' "
          f"(mediane = {panel[colonne_mvel1_brut].median():,.0f}, "
          f"avant winsorizing et rank transform)")

    # --- B.3 Imputation (mediane du mois, sur la population post-filtres) ---
    rap.table('B_missing_avant_imputation',
              (panel[caracteristiques].isna().mean() * 100).sort_values(ascending=False).rename('pct_manquant'))

    caracteristiques_a_imputer = [c for c in caracteristiques if c != 'mvel1']

    groupes_mois = panel.groupby('annee_mois', observed=True)[caracteristiques_a_imputer]
    panel[caracteristiques_a_imputer] = groupes_mois.transform(imputer_par_mediane_mois)
    del groupes_mois
    gc.collect()

    colonnes_entierement_vides = []
    for col in caracteristiques_a_imputer:
        if panel[col].isna().any():
            panel[col] = panel[col].fillna(panel[col].median())
        if panel[col].isna().any():
            colonnes_entierement_vides.append(col)
            panel[col] = panel[col].fillna(0)

    rap.valeur('B_n_caracteristiques_imputees', len(caracteristiques_a_imputer))
    rap.valeur('B_colonnes_entierement_vides', colonnes_entierement_vides)
    print(f"Imputation terminee sur {len(caracteristiques_a_imputer)} caracteristiques (hors mvel1).")

    total_manquant_chars = int(panel[caracteristiques].isna().sum().sum())
    rap.valeur('B_missing_restant_apres_imputation', total_manquant_chars)
    assert total_manquant_chars == 0, "Il reste des valeurs manquantes parmi les caracteristiques !"
    assert panel['mvel1'].isna().sum() == 0, "mvel1 ne devrait jamais avoir de valeurs manquantes ici !"

    # --- B.4 Winsorizing (1er/99e centile, par mois) ---
    groupes_mois = panel.groupby('annee_mois', observed=True)[caracteristiques]
    panel[caracteristiques] = groupes_mois.transform(winsorize_groupe)
    del groupes_mois
    gc.collect()
    print("Winsorizing termine (sur", len(caracteristiques), "caracteristiques).")

    # --- B.5 Rank transform (par mois, dans [-1, 1]) ---
    groupes_mois = panel.groupby('annee_mois', observed=True)[caracteristiques]
    panel[caracteristiques] = groupes_mois.transform(rank_transform)
    del groupes_mois
    gc.collect()
    print("Transformation en rang terminee.")

    # Diagnostics du rank transform (affiches par le notebook)
    rap.table('B_ecart_type_apres_rank',
              panel[caracteristiques].std().sort_values().rename('ecart_type'))
    rap.table('B_bornes_apres_rank', panel[caracteristiques].agg(['min', 'max']).T)

    # --- B.7 Verifications finales et sauvegarde ---
    rap.valeur('B_shape_finale', list(panel.shape))
    rap.valeur('B_periode', [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])
    rap.valeur('B_n_mois_distincts', int(panel['annee_mois'].nunique()))

    manquants = panel[caracteristiques + macro_predicteurs + [cible]].isna().sum()
    rap.table('B_missing_final', manquants[manquants > 0].rename('nb_manquant')
              if manquants.sum() > 0 else pd.Series(dtype='int64', name='nb_manquant'))

    # Retour au dtype d'origine avant sauvegarde (les etapes 04 a 08 ne savent rien
    # de la conversion en 'category', qui ne sert qu'a alleger la memoire ici).
    for col in ['annee_mois', 'permno']:
        if isinstance(panel[col].dtype, pd.CategoricalDtype):
            panel[col] = panel[col].astype(panel[col].cat.categories.dtype)

    rap.table('B_apercu_final',
              panel[['permno', 'annee_mois', config.COLONNE_MVEL1_BRUT]
                    + caracteristiques[:5] + macro_predicteurs[:2] + [cible]].head())

    panel.to_parquet(config.FICHIER_PANEL_MODELISATION, index=False)
    print("Fichier sauvegarde :", config.FICHIER_PANEL_MODELISATION)


# ============================================================
def main():
    partie_b_seulement = '--partie-b-seulement' in sys.argv

    config.assurer_dossiers()
    rapports.assurer_dossier()

    # On repart du rapport existant si on ne rejoue que la partie B, pour ne pas perdre
    # les diagnostics de la partie A affiches par le notebook.
    if partie_b_seulement and rapports.existe('03_panel'):
        rap = rapports.charger('03_panel')
    else:
        rap = rapports.Rapport('03_panel')

    if partie_b_seulement:
        print("Mode --partie-b-seulement : rechargement de", config.FICHIER_PANEL_FINAL)
        panel = pd.read_parquet(config.FICHIER_PANEL_FINAL)
    else:
        panel = construire_panel_fusionne(rap)
        rap.sauvegarder()  # la partie A n'est pas perdue si la partie B plante

    preparer_panel_modelisation(panel, rap)
    rap.sauvegarder()

    print("\n" + "=" * 70)
    print("ETAPE 03 TERMINEE.")
    print("  -> ouvre notebooks/03_construction_panel.ipynb pour visualiser les resultats")
    print("=" * 70)


if __name__ == "__main__":
    main()
