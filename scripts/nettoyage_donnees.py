"""
ETAPE 02 -- Nettoyage des 3 sources de donnees (ex-notebook 02).

Ce script contient TOUT le calcul qui se trouvait auparavant dans les cellules du
notebook `02_nettoyage_donnees.ipynb`. Le notebook, lui, ne fait plus que LIRE et
AFFICHER ce que ce script a produit (voir rapports.py).

Lancement, depuis la RACINE du projet :

    python scripts/nettoyage_donnees.py

Quatre parties independantes entre elles (aucune ne lit le resultat d'une autre) :
  A -- Caracteristiques (datashare.parquet)  -> data/interim/characteristics_clean.parquet
                                                + data/interim/caracteristiques_retenues.json
  B -- Rendements (StockReturn.parquet)      -> data/interim/returns_clean.parquet
  C -- Macro (MacroData.parquet)             -> data/interim/macro_clean.parquet
  D -- Facteurs FF5 + MOM (2 CSV Ken French) -> data/interim/facteurs_clean.parquet

⚠️ La partie D est la seule dont la sortie ne sert PAS a predire : les facteurs de Fama-
French n'entrent jamais dans le panel de modelisation, ils servent uniquement a EVALUER
les portefeuilles a l'etape 08 (alphas, comparaison de Sharpe). Elle est ici parce que
c'est l'etape ou l'on transforme du brut en propre, pas parce qu'elle nourrit les modeles.

Tous les parametres viennent de config.py (ANNEE_DEBUT, CARACTERISTIQUES,
SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES, MACRO_PREDICTEURS) : modifie-les LA-BAS,
puis relance ce script.

En plus des 3 fichiers ci-dessus, il ecrit le rapport `02_nettoyage`
(outputs/rapports/) : tous les compteurs et petits tableaux de diagnostic que le
notebook 02 affichait autrefois au fil des cellules.
"""

import json
import sys
from pathlib import Path

# Permet d'importer config.py / rapports.py (a la RACINE) quel que soit le repertoire
# courant depuis lequel ce script est lance.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
import facteurs
import rapports


# ============================================================
# Partie A -- Caracteristiques d'entreprise (datashare.parquet)
# ============================================================

def nettoyer_caracteristiques(rap):
    print("\n" + "=" * 70)
    print("PARTIE A -- Caracteristiques (datashare.parquet)")
    print("=" * 70)

    # --- A.1 Chargement ---
    chars = pd.read_parquet(config.FICHIER_CARACTERISTIQUES_BRUT)
    print("Dimensions de depart :", chars.shape)
    rap.valeur('A_shape_depart', list(chars.shape))
    rap.table('A_apercu_brut', chars.head())

    # --- A.2 Periode de depart : evolution du taux de missing par decennie ---
    chars['annee'] = chars['DATE'] // 10000
    chars['decennie'] = (chars['annee'] // 10) * 10

    colonnes_predicteurs = [c for c in chars.columns if c not in
                            ['permno', 'DATE', 'sic2', 'annee', 'decennie']]

    missing_par_decennie = (
        chars.groupby('decennie')[colonnes_predicteurs]
             .apply(lambda g: g.isna().mean().mean())
             .rename("taux_missing_moyen")
    )
    rap.table('A_missing_par_decennie', (missing_par_decennie * 100).round(1))
    print("\nTaux moyen de valeurs manquantes par decennie (%) :")
    print((missing_par_decennie * 100).round(1))

    ANNEE_DEBUT = config.ANNEE_DEBUT
    print(f"\nANNEE_DEBUT (depuis config.py) : {ANNEE_DEBUT}")
    chars = chars[chars['annee'] >= ANNEE_DEBUT].copy()
    print(f"Dimensions apres filtre sur annee >= {ANNEE_DEBUT} :", chars.shape)
    rap.valeur('A_annee_debut', ANNEE_DEBUT)
    rap.valeur('A_shape_apres_filtre_annee', list(chars.shape))

    # --- A.3 Univers candidat ---
    caracteristiques_candidates = config.CARACTERISTIQUES
    print(f"\nUnivers candidat : {len(caracteristiques_candidates)} caracteristiques "
          "(sur 94 dans Gu, Kelly & Xiu 2020)")
    rap.valeur('A_caracteristiques_candidates', list(caracteristiques_candidates))

    manquantes_dans_fichier = [c for c in caracteristiques_candidates if c not in chars.columns]
    rap.valeur('A_colonnes_introuvables', manquantes_dans_fichier)
    if manquantes_dans_fichier:
        print("ATTENTION, colonnes introuvables dans le fichier :", manquantes_dans_fichier)
    else:
        print("OK : toutes les colonnes candidates existent dans le fichier.")

    # --- A.3bis Filtre des caracteristiques trop incompletes ---
    SEUIL_MISSING = config.SEUIL_MAX_PCT_MANQUANT_CARACTERISTIQUES

    taux_missing = chars[caracteristiques_candidates].isna().mean().sort_values(ascending=False)

    caracteristiques_gardees = [c for c in caracteristiques_candidates if taux_missing[c] <= SEUIL_MISSING]
    caracteristiques_exclues = [c for c in caracteristiques_candidates if taux_missing[c] > SEUIL_MISSING]

    print(f"\nSeuil d'exclusion : {SEUIL_MISSING*100:.0f}% de valeurs manquantes "
          f"(sur annee >= {ANNEE_DEBUT})")
    print(f"Caracteristiques retenues : {len(caracteristiques_gardees)} / {len(caracteristiques_candidates)}")
    if caracteristiques_exclues:
        print("Caracteristiques EXCLUES (trop incompletes) :")
        for c in caracteristiques_exclues:
            print(f"  - {c:<18s} {taux_missing[c]*100:5.1f}% manquant")
    else:
        print("Aucune caracteristique exclue : toutes les candidates passent le seuil.")

    rap.valeur('A_seuil_missing', SEUIL_MISSING)
    rap.valeur('A_caracteristiques_gardees', caracteristiques_gardees)
    rap.valeur('A_caracteristiques_exclues', caracteristiques_exclues)
    # Tableau complet (les 94), c'est lui qui alimente le graphique du notebook
    rap.table('A_taux_missing', (taux_missing * 100).rename('pct_manquant'))

    # Manifeste lu ensuite par config.charger_caracteristiques_retenues()
    manifeste = {
        'seuil_max_pct_manquant': SEUIL_MISSING,
        'annee_debut': ANNEE_DEBUT,
        'caracteristiques_retenues': caracteristiques_gardees,
        'caracteristiques_exclues': caracteristiques_exclues,
        'taux_missing_pct': {c: round(float(taux_missing[c]) * 100, 3) for c in caracteristiques_candidates},
    }
    with open(config.FICHIER_CARACTERISTIQUES_RETENUES, 'w') as f:
        json.dump(manifeste, f, indent=2, ensure_ascii=False)
    print("Manifeste sauvegarde :", config.FICHIER_CARACTERISTIQUES_RETENUES)

    # --- Reduction aux colonnes utiles ---
    chars = chars[['permno', 'DATE', 'annee', 'sic2'] + caracteristiques_gardees].copy()
    print("Dimensions apres reduction des colonnes :", chars.shape)
    rap.valeur('A_shape_apres_reduction', list(chars.shape))

    # --- A.4 Valeurs manquantes restantes ---
    manquants = chars[caracteristiques_gardees].isna().sum()
    manquants = manquants[manquants > 0].sort_values(ascending=False)
    rap.table('A_missing_restant_pct', (manquants / len(chars) * 100).round(2).rename('pct_manquant'))
    print(f"\nValeurs manquantes restantes parmi les {len(caracteristiques_gardees)} "
          "caracteristiques retenues (imputees au notebook 03, partie B) :")
    print((manquants / len(chars) * 100).round(2) if len(manquants) else "Aucune.")

    # --- A.5 Nettoyage du secteur ---
    def nettoyer_secteur(valeur):
        if pd.isna(valeur):
            return "Inconnu"
        return str(int(valeur))

    chars['secteur'] = chars['sic2'].apply(nettoyer_secteur)
    chars = chars.drop(columns=['sic2'])
    rap.table('A_repartition_secteurs', chars['secteur'].value_counts().head(10).rename('nb_lignes'))

    # --- A.6 Verifications finales et sauvegarde ---
    print("\nDimensions finales :", chars.shape)
    rap.valeur('A_shape_finale', list(chars.shape))
    rap.valeur('A_colonnes_finales', list(chars.columns))
    rap.table('A_missing_final', chars.isna().sum().rename('nb_manquant'))
    rap.table('A_describe', chars.describe().T)
    rap.table('A_apercu_final', chars.head())

    assert chars['permno'].isna().sum() == 0
    assert chars['DATE'].isna().sum() == 0

    chars.to_parquet(config.FICHIER_CARACTERISTIQUES_CLEAN, index=False)
    print("Fichier sauvegarde :", config.FICHIER_CARACTERISTIQUES_CLEAN)

    return caracteristiques_gardees


# ============================================================
# Partie B -- Rendements (StockReturn.parquet)
# ============================================================

def nettoyer_rendements(rap):
    print("\n" + "=" * 70)
    print("PARTIE B -- Rendements (StockReturn.parquet)")
    print("=" * 70)

    # --- B.1 Chargement ---
    returns = pd.read_parquet(config.FICHIER_RETURNS_BRUT)
    print("Dimensions de depart :", returns.shape)
    print("Type de la colonne RET :", returns['RET'].dtype)
    rap.valeur('B_shape_depart', list(returns.shape))
    rap.valeur('B_dtype_ret_depart', str(returns['RET'].dtype))
    rap.table('B_apercu_brut', returns.head(10))

    # --- B.2 Codes CRSP non numeriques ---
    ret_numerique_test = pd.to_numeric(returns['RET'], errors='coerce')
    codes_speciaux = returns.loc[returns['RET'].notna() & ret_numerique_test.isna(), 'RET']

    rap.table('B_codes_speciaux', codes_speciaux.value_counts().rename('nb_lignes'))
    rap.valeur('B_nb_codes_speciaux', int(len(codes_speciaux)))
    rap.valeur('B_pct_codes_speciaux', float(len(codes_speciaux) / len(returns) * 100))
    print(f"\nCodes non numeriques trouves dans RET : {len(codes_speciaux)} lignes "
          f"({len(codes_speciaux) / len(returns) * 100:.2f} %)")
    print(codes_speciaux.value_counts())

    # --- B.3 Conversion definitive ---
    returns['RET'] = pd.to_numeric(returns['RET'], errors='coerce')
    pct_manquant = returns['RET'].isna().mean() * 100
    rap.valeur('B_pct_manquant_apres_conversion', float(pct_manquant))
    print(f"\n% de RET manquant apres conversion : {pct_manquant:.2f} %")

    # --- B.4 Suppression des rendements manquants (jamais imputes : c'est la cible) ---
    avant = len(returns)
    returns = returns.dropna(subset=['RET']).copy()
    apres = len(returns)
    rap.valeur('B_lignes_avant_dropna', int(avant))
    rap.valeur('B_lignes_apres_dropna', int(apres))
    rap.valeur('B_pct_lignes_supprimees', float((avant - apres) / avant * 100))
    print(f"Lignes avant suppression : {avant}")
    print(f"Lignes apres suppression : {apres} "
          f"({(avant - apres) / avant * 100:.2f} % supprimees)")

    # --- B.5 Valeurs extremes (diagnostic seulement, rien n'est supprime) ---
    rap.table('B_describe_ret', returns['RET'].describe().rename('RET'))
    rap.table('B_quantiles_ret', returns['RET'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).rename('RET'))

    seuil_extreme = 5.0  # +500%
    rap.valeur('B_seuil_extreme', seuil_extreme)
    rap.valeur('B_nb_extremes_hauts', int((returns['RET'] > seuil_extreme).sum()))
    rap.valeur('B_nb_extremes_bas', int((returns['RET'] < -0.95).sum()))
    print(f"Rendements > +{seuil_extreme*100:.0f}% : {(returns['RET'] > seuil_extreme).sum()} lignes")
    print(f"Rendements < -95% : {(returns['RET'] < -0.95).sum()} lignes")

    # --- B.6 Harmonisation des identifiants et cle temporelle ---
    returns = returns.rename(columns={'PERMNO': 'permno'})
    returns['annee_mois'] = returns['date'].astype(str).str[:6]

    doublons = int(returns.duplicated(subset=['permno', 'annee_mois']).sum())
    rap.valeur('B_doublons_permno_mois', doublons)
    print(f"Nombre de doublons (permno, annee_mois) : {doublons}")
    if doublons > 0:
        print("ATTENTION : a investiguer avant la fusion (etape 03, partie A).")

    # --- B.7 Verification finale et sauvegarde ---
    returns_finales = returns[['permno', 'annee_mois', 'RET']]
    rap.valeur('B_shape_finale', list(returns_finales.shape))
    rap.valeur('B_colonnes_finales', list(returns_finales.columns))
    rap.valeur('B_periode', [str(returns['annee_mois'].min()), str(returns['annee_mois'].max())])
    rap.table('B_missing_final', returns_finales.isna().sum().rename('nb_manquant'))
    rap.table('B_apercu_final', returns_finales.head())

    returns_finales.to_parquet(config.FICHIER_RETURNS_CLEAN, index=False)
    print("Fichier sauvegarde :", config.FICHIER_RETURNS_CLEAN)


# ============================================================
# Partie C -- Variables macro (MacroData.parquet)
# ============================================================

def nettoyer_macro(rap):
    print("\n" + "=" * 70)
    print("PARTIE C -- Macro (MacroData.parquet)")
    print("=" * 70)

    # --- C.1 Chargement ---
    macro = pd.read_parquet(config.FICHIER_MACRO_BRUT)
    print("Dimensions de depart :", macro.shape)
    rap.valeur('C_shape_depart', list(macro.shape))
    rap.valeur('C_colonnes_depart', list(macro.columns))
    rap.valeur('C_periode_depart', [int(macro['yyyymm'].min()), int(macro['yyyymm'].max())])
    rap.table('C_apercu_brut', macro.head())

    # --- C.1bis Conversion numerique forcee (separateurs de milliers) ---
    colonnes_a_convertir = [c for c in macro.columns if c != 'yyyymm']
    for col in colonnes_a_convertir:
        macro[col] = pd.to_numeric(
            macro[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    rap.table('C_dtypes_apres_conversion',
              macro.dtypes.astype(str).rename('dtype'))

    # --- C.2 Restriction a la periode utile ---
    ANNEE_DEBUT = config.ANNEE_DEBUT
    macro['annee'] = macro['yyyymm'] // 100
    macro = macro[macro['annee'] >= ANNEE_DEBUT].copy()
    print(f"Dimensions apres filtre sur annee >= {ANNEE_DEBUT} :", macro.shape)
    rap.valeur('C_annee_debut', ANNEE_DEBUT)
    rap.valeur('C_shape_apres_filtre_annee', list(macro.shape))

    # --- C.3 Valeurs manquantes : ffill puis bfill ---
    colonnes_macro = [c for c in macro.columns if c not in ['yyyymm', 'annee']]
    rap.table('C_missing_avant_comblement',
              (macro[colonnes_macro].isna().mean() * 100).round(1).sort_values(ascending=False).rename('pct_manquant'))

    colonnes_entierement_vides = []
    for col in colonnes_macro:
        macro[col] = macro[col].ffill().bfill()
        if macro[col].isna().any():
            colonnes_entierement_vides.append(col)

    rap.valeur('C_colonnes_entierement_vides', colonnes_entierement_vides)
    print("Comblement des trous termine (ffill puis bfill).")
    if colonnes_entierement_vides:
        print("ATTENTION - colonnes ENTIEREMENT vides sur la periode choisie :", colonnes_entierement_vides)

    # --- C.4 Construction des 8 predicteurs macro de GKX ---
    valeurs_non_positives = {}
    for col in ['Index', 'D12', 'E12']:
        nb_non_positif = int((macro[col] <= 0).sum())
        valeurs_non_positives[col] = nb_non_positif
        if nb_non_positif > 0:
            print(f"ATTENTION : {nb_non_positif} valeurs <= 0 dans '{col}', le log sera NaN.")
    rap.valeur('C_valeurs_non_positives', valeurs_non_positives)

    macro['macro_dp'] = np.log(macro['D12']) - np.log(macro['Index'])
    macro['macro_ep'] = np.log(macro['E12']) - np.log(macro['Index'])
    macro['macro_bm'] = macro['b/m']
    macro['macro_ntis'] = macro['ntis']
    macro['macro_tbl'] = macro['tbl']
    macro['macro_tms'] = macro['lty'] - macro['tbl']
    macro['macro_dfy'] = macro['BAA'] - macro['AAA']
    macro['macro_svar'] = macro['svar']

    predicteurs_gkx = config.MACRO_PREDICTEURS
    rap.table('C_apercu_predicteurs', macro[predicteurs_gkx].head())
    rap.table('C_missing_predicteurs_gkx', macro[predicteurs_gkx].isna().sum().rename('nb_manquant'))

    # --- C.5 Cle temporelle et selection finale ---
    macro['annee_mois'] = macro['yyyymm'].astype(str)
    macro_final = macro[['annee_mois'] + predicteurs_gkx + ['Rfree']].copy()

    # --- C.6 Verification finale et sauvegarde ---
    print("Dimensions finales :", macro_final.shape)
    rap.valeur('C_shape_finale', list(macro_final.shape))
    rap.valeur('C_colonnes_finales', list(macro_final.columns))
    rap.table('C_missing_final', macro_final.isna().sum().rename('nb_manquant'))
    rap.table('C_apercu_final', macro_final.head())

    macro_final.to_parquet(config.FICHIER_MACRO_CLEAN, index=False)
    print("Fichier sauvegarde :", config.FICHIER_MACRO_CLEAN)


# ============================================================
# Partie D -- Facteurs de risque FF5 + momentum (Ken French)
# ============================================================

def _trouver_csv_french(chemin_attendu):
    """Retrouve un CSV de Ken French meme si son extension n'a pas la casse attendue.

    Les archives publiees contiennent des fichiers en `.CSV` MAJUSCULE
    (`F-F_Momentum_Factor.CSV`). Sur Linux et macOS, `Path.exists()` est sensible a la
    casse : un dezippage fidele produit donc un FileNotFoundError alors que le fichier est
    bien la, sous le nez de l'utilisateur. Plutot que d'imposer un renommage manuel, on
    balaye le dossier en comparant les noms en minuscules.

    Renvoie le chemin reellement trouve, ou leve une erreur qui nomme le fichier a
    telecharger et l'endroit ou le mettre.
    """
    if chemin_attendu.exists():
        return chemin_attendu

    dossier = chemin_attendu.parent
    if dossier.exists():
        cible = chemin_attendu.name.lower()
        for candidat in dossier.iterdir():
            if candidat.name.lower() == cible:
                print(f"ℹ️ {candidat.name} utilise (casse differente de {chemin_attendu.name})")
                return candidat

    raise FileNotFoundError(
        f"{chemin_attendu} est introuvable.\n"
        "  Telecharge depuis la Data Library de Ken French :\n"
        "    - Fama/French 5 Factors (2x3)  -> F-F_Research_Data_5_Factors_2x3_CSV.zip\n"
        "    - Momentum Factor (Mom)        -> F-F_Momentum_Factor_CSV.zip\n"
        f"  puis dezippe les DEUX dans {dossier}/ (les variantes _daily ne conviennent pas)."
    )


def nettoyer_facteurs_de_risque(rap):
    """PARTIE D -- Les 6 series de facteurs, fusionnees et remises en decimal.

    Tout le detail du nettoyage vit dans `facteurs.nettoyer_facteurs` (module de la racine,
    au meme titre que portefeuilles.py) : cette fonction n'est qu'un point d'entree, pour
    que la partie D se lance et se rapporte exactement comme A, B et C.

    ⚠️ Pre-requis : les DEUX CSV dezippes dans data/raw/, tels que telecharges depuis la
    Data Library de Ken French. Ne pas les ouvrir puis les re-enregistrer avec un tableur --
    celui-ci reformate la premiere colonne et le bloc mensuel n'est plus reconnu.
    """
    print("\n" + "=" * 70)
    print("PARTIE D -- FACTEURS DE RISQUE (FF5 + MOM)")
    print("=" * 70)

    chemin_ff5 = _trouver_csv_french(config.FICHIER_FACTEURS_FF5_BRUT)
    chemin_mom = _trouver_csv_french(config.FICHIER_FACTEURS_MOM_BRUT)

    facteurs.nettoyer_facteurs(
        chemin_ff5,
        chemin_mom,
        chemin_sortie=config.FICHIER_FACTEURS_CLEAN,
        rap=rap,
    )


# ============================================================
def main():
    config.assurer_dossiers()
    rapports.assurer_dossier()

    rap = rapports.Rapport('02_nettoyage')

    caracteristiques_gardees = nettoyer_caracteristiques(rap)
    rap.sauvegarder()  # sauvegarde intermediaire : la partie A n'est pas perdue si B ou C plante

    nettoyer_rendements(rap)
    rap.sauvegarder()

    nettoyer_macro(rap)
    rap.sauvegarder()

    nettoyer_facteurs_de_risque(rap)
    rap.sauvegarder()

    print("\n" + "=" * 70)
    print("ETAPE 02 TERMINEE.")
    print(f"  {len(caracteristiques_gardees)} caracteristiques retenues")
    print("  4 fichiers ecrits dans data/interim/")
    print("  -> ouvre notebooks/02_nettoyage_donnees.ipynb pour visualiser les resultats")
    print("=" * 70)


if __name__ == "__main__":
    main()
