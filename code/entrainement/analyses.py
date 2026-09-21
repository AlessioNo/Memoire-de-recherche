"""
Analyses annexes : ce qui vient APRES l'entrainement et ne concerne qu'un modele.

Chacune de ces fonctions vivait dans le script du modele correspondant (section 6bis des
anciens etape04 a etape07). Elles sont regroupees ici pour la meme raison que la boucle
d'entrainement l'est dans `boucle.py` : SHAP est ecrit une seule fois pour LightGBM et le
Random Forest, alors qu'il l'etait deux fois.

⚠️ Ces analyses ne tournent qu'a l'horizon 1 mois. C'est le comportement d'origine (les
anciens scripts etape11 a etape14 n'appelaient que `entrainer()`), et il est justifie :
- la regle de lags de Newey-West de `fama_macbeth` est calibree pour des rendements
  mensuels NON chevauchants ; sur une cible a 12 mois, les residus suivent un MA(11) et
  les t-stats seraient largement surevaluees ;
- les tableaux `importance_*.parquet` sont lus par les notebooks 04 a 07, qui n'affichent
  que la piste a 1 mois ; les ecraser avec des valeurs a 12 mois melangerait les deux.
"""

import time

import numpy as np
import pandas as pd

import chemins
import config
import fenetres


# ============================================================
# Regression lineaire -- significativite de Fama-MacBeth (1973) + Newey-West
# ============================================================

def fama_macbeth(panel, rap):
    """Une regression cross-sectionnelle PAR MOIS sur les caracteristiques uniquement.

    Les predicteurs macro sont exclus : constants dans un mois donne, donc colineaires avec
    la constante de la regression cross-sectionnelle -- coefficient non identifiable.

    Utilise TOUT l'historique du panel, pas seulement le train d'une fenetre : l'objectif
    est descriptif (quelles caracteristiques comptent en moyenne, de facon stable dans le
    temps), pas une nouvelle evaluation hors-echantillon.

    Sauvegarde en plus la matrice mois x predicteurs des coefficients ET celle de leurs
    ecarts-types, pour la figure 6ter du notebook 04 (frequence de significativite et
    coherence de signe, fenetre par fenetre).
    """
    import statsmodels.api as sm

    print("\n--- Section 6bis : significativite Fama-MacBeth ---")
    predicteurs = config.PREDICTEURS
    caracteristiques = config.CARACTERISTIQUES_RETENUES
    cible = config.CIBLE

    predicteurs_fm = [p for p in predicteurs if p in caracteristiques]

    coefs_mensuels = []
    erreurs_mensuelles = []
    mois_utilises = []
    for mois, g in panel.groupby('annee_mois', sort=True, observed=True):
        # Securite : il faut strictement plus d'observations que de parametres a estimer
        if len(g) <= len(predicteurs_fm) + 1:
            continue
        X_design = np.column_stack([np.ones(len(g)), g[predicteurs_fm].values])
        y = g[cible].values
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        # Ecarts-types OLS de la coupe transversale de CE mois. Avec n ~ 3000 entreprises
        # pour ~90 parametres, ils sont bien estimes -- contrairement a un ecart-type
        # calcule sur les 12 coefficients d'une fenetre.
        residus = y - X_design @ beta
        n_obs, n_param = X_design.shape
        variance = (residus @ residus) / (n_obs - n_param)
        erreurs = np.sqrt(variance * np.diag(np.linalg.pinv(X_design.T @ X_design)))

        coefs_mensuels.append(beta[1:])      # on jette la constante, on garde les pentes
        erreurs_mensuelles.append(erreurs[1:])
        mois_utilises.append(mois)

    coefs_mensuels = pd.DataFrame(coefs_mensuels, columns=predicteurs_fm, index=mois_utilises)
    print(f"{len(coefs_mensuels)} regressions cross-sectionnelles mensuelles estimees "
          f"({len(predicteurs_fm)} predicteurs, periode {mois_utilises[0]} a {mois_utilises[-1]}).")

    # <-- AJOUT : matrices mois x predicteurs sur disque, pour la section 6ter du notebook 04
    coefs_mensuels.index = coefs_mensuels.index.astype(str)
    coefs_mensuels.index.name = 'annee_mois'
    coefs_mensuels.to_parquet(chemins.sortie('coefficients_fm_mensuels'))

    erreurs_mensuelles = pd.DataFrame(erreurs_mensuelles,
                                      columns=predicteurs_fm,
                                      index=coefs_mensuels.index)
    erreurs_mensuelles.index.name = 'annee_mois'
    erreurs_mensuelles.to_parquet(chemins.sortie('erreurs_types_fm_mensuels'))

    print("Coefficients mensuels sauvegardes :", chemins.sortie('coefficients_fm_mensuels'))
    print("Ecarts-types mensuels sauvegardes :", chemins.sortie('erreurs_types_fm_mensuels'))
    # fin AJOUT -->

    # Moyenne temporelle + t-stat de Newey-West
    T_mois = len(coefs_mensuels)
    nb_lags_nw = max(1, int(np.floor(4 * (T_mois / 100) ** (2 / 9))))
    print(f"Nombre de lags Newey-West : {nb_lags_nw} (regle de Newey-West 1994, T={T_mois} mois)")

    lignes = []
    for p in predicteurs_fm:
        serie = coefs_mensuels[p].values
        resultat_nw = sm.OLS(serie, np.ones(len(serie))).fit(
            cov_type='HAC', cov_kwds={'maxlags': nb_lags_nw})
        lignes.append({
            'predicteur': p,
            'coef_moyen_fm': resultat_nw.params[0],
            'erreur_type_nw': resultat_nw.bse[0],
            't_stat_nw': resultat_nw.tvalues[0],
            'p_value_nw': resultat_nw.pvalues[0],
            'pct_mois_signe_positif': (serie > 0).mean() * 100,
        })

    significativite = pd.DataFrame(lignes).set_index('predicteur')
    significativite['significatif_5pct'] = significativite['p_value_nw'] < 0.05
    significativite = significativite.sort_values('t_stat_nw', key=abs, ascending=False)

    nb_significatifs = int(significativite['significatif_5pct'].sum())
    print(f"{nb_significatifs} / {len(significativite)} predicteurs significatifs a 5%.")

    destination = chemins.importance('regression_lineaire')
    significativite.to_parquet(destination)
    print("Tableau de significativite sauvegarde :", destination)

    rap.valeur('fm_n_regressions', int(len(coefs_mensuels)))
    rap.valeur('fm_n_predicteurs', len(predicteurs_fm))
    rap.valeur('fm_periode', [str(mois_utilises[0]), str(mois_utilises[-1])])
    rap.valeur('fm_nb_lags_nw', nb_lags_nw)
    rap.valeur('fm_nb_significatifs', nb_significatifs)
    rap.valeur('fm_n_teste', int(len(significativite)))

# ============================================================
# Elastic Net -- stabilite de la selection de variables
# ============================================================

def stabilite_selection(sortie, rap):
    """Stability selection (Meinshausen & Buhlmann, 2010) : sur TOUTES les fenetres
    entrainees, a quelle frequence chaque predicteur est-il retenu (coefficient != 0),
    avec quelle magnitude moyenne et quelle coherence de signe."""
    print("\n--- Section 6bis : stabilite de la selection de variables ---")

    coefficients = pd.DataFrame(
        sortie['coefficients_par_fenetre'],
        index=sortie['resultats_par_fenetre']['fenetre'],
    )
    print(f"Coefficients sauvegardes pour {len(coefficients)} fenetres "
          f"x {coefficients.shape[1]} predicteurs.")

    def moyenne_si_selectionne(colonne):
        non_nuls = colonne[colonne != 0]
        return non_nuls.mean() if len(non_nuls) > 0 else 0.0

    def signe_majoritaire(colonne):
        non_nuls = colonne[colonne != 0]
        if len(non_nuls) == 0:
            return 0
        return 1 if (non_nuls > 0).mean() >= 0.5 else -1

    def pct_accord_signe(colonne):
        non_nuls = colonne[colonne != 0]
        if len(non_nuls) == 0:
            return np.nan
        majorite = 1 if (non_nuls > 0).mean() >= 0.5 else -1
        return (np.sign(non_nuls) == majorite).mean() * 100

    stabilite = pd.DataFrame({
        'frequence_selection_pct': (coefficients != 0).mean() * 100,
        'coef_moyen_si_selectionne': coefficients.apply(moyenne_si_selectionne),
        'signe_dominant': coefficients.apply(signe_majoritaire),
        'coherence_signe_pct': coefficients.apply(pct_accord_signe),
    }).sort_values(['frequence_selection_pct', 'coef_moyen_si_selectionne'],
                   key=abs, ascending=False)

    n_toujours = int((stabilite['frequence_selection_pct'] == 100).sum())
    n_jamais = int((stabilite['frequence_selection_pct'] == 0).sum())
    print(f"{n_toujours} / {len(stabilite)} predicteurs selectionnes dans TOUTES les fenetres.")
    print(f"{n_jamais} / {len(stabilite)} predicteurs jamais selectionnes.")

    destination = chemins.importance('elastic_net')
    stabilite.to_parquet(destination)
    print("Tableau de stabilite sauvegarde :", destination)

    rap.valeur('stab_n_toujours_selectionnes', n_toujours)
    rap.valeur('stab_n_jamais_selectionnes', n_jamais)
    rap.valeur('stab_n_predicteurs', int(len(stabilite)))
    # Coefficients bruts par fenetre : utile pour un diagnostic fin dans le notebook
    rap.table('coefficients_par_fenetre', coefficients)


# ============================================================
# Modeles d'arbres -- importance agregee sur toutes les fenetres
# ============================================================

def _annoncer_importance(importance, n_fenetres, destination):
    n_jamais = int((importance['jamais_utilisee_pct_fenetres'] == 100).sum())
    print(f"Importance agregee sur {n_fenetres} fenetres.")
    print(f"{n_jamais} / {len(importance)} predicteurs ne sont utilises dans AUCUN arbre "
          "d'aucune fenetre.")
    importance.to_parquet(destination)
    print("Tableau d'importance sauvegarde :", destination)
    return n_jamais


def importance_agregee_lightgbm(sortie, rap):
    """Gain et split moyens sur toutes les fenetres."""
    print("\n--- Section 6bis : importance agregee sur toutes les fenetres ---")

    index_fenetres = sortie['resultats_par_fenetre']['fenetre']
    gain = pd.DataFrame(sortie['importances_gain_par_fenetre'], index=index_fenetres)
    split = pd.DataFrame(sortie['importances_split_par_fenetre'], index=index_fenetres)

    importance = pd.DataFrame({
        'gain_moyen_pct': gain.mean(),
        'gain_ecart_type_pct': gain.std(),
        'split_moyen': split.mean(),
        'split_ecart_type': split.std(),
        'jamais_utilisee_pct_fenetres': (split == 0).mean() * 100,
    }).sort_values('gain_moyen_pct', ascending=False)

    n_jamais = _annoncer_importance(importance, len(gain), chemins.importance('lightgbm'))

    rap.valeur('imp_n_jamais_utilisees', n_jamais)
    rap.valeur('imp_n_predicteurs', int(len(importance)))
    rap.table('gain_par_fenetre', gain)
    rap.table('split_par_fenetre', split)


def importance_agregee_random_forest(sortie, rap):
    """Importance MDI (Mean Decrease in Impurity) moyennee sur toutes les fenetres.

    ⚠️ Limite connue de cette mesure (Strobl et al. 2007) : elle est biaisee en faveur des
    variables a forte cardinalite et, entre deux variables correlees, en distribue
    arbitrairement le credit. Les caracteristiques de GKX etant fortement correlees entre
    elles, la lire comme un classement absolu serait une erreur -- c'est aussi pour ca que
    la section SHAP existe juste apres.
    """
    print("\n--- Section 6bis : importance agregee sur toutes les fenetres ---")

    index_fenetres = sortie['resultats_par_fenetre']['fenetre']
    par_fenetre = pd.DataFrame(sortie['importances_par_fenetre'], index=index_fenetres)

    importance = pd.DataFrame({
        'importance_moyenne_pct': par_fenetre.mean(),
        'importance_ecart_type_pct': par_fenetre.std(),
        'rang_moyen': par_fenetre.rank(axis=1, ascending=False).mean(),
        'jamais_utilisee_pct_fenetres': (par_fenetre == 0).mean() * 100,
    }).sort_values('importance_moyenne_pct', ascending=False)

    n_jamais = _annoncer_importance(importance, len(par_fenetre),
                                    chemins.importance('random_forest'))

    rap.valeur('imp_n_jamais_utilisees', n_jamais)
    rap.valeur('imp_n_predicteurs', int(len(importance)))
    rap.table('importance_par_fenetre', par_fenetre)


# ============================================================
# Modeles d'arbres -- valeurs SHAP du modele de la derniere fenetre
# ============================================================

def calculer_shap(panel, sortie, rap, taille_max, saute=False,
                  check_additivity=True, chronometre=False):
    """Valeurs SHAP du modele de la derniere fenetre, sur un echantillon de son test.

    Elles sont enregistrees dans le rapport : le notebook trace ensuite le summary_plot a
    partir de ces valeurs, sans jamais relancer l'explainer (couteux).

    `taille_max` : plafond de l'echantillon. 2000 pour LightGBM, 500 pour le Random Forest
    -- le cout de TreeExplainer croit avec le nombre ET la taille des arbres, or une foret
    a des arbres bien plus gros qu'un booster.
    """
    if saute:
        print("\n(--sans-shap : calcul SHAP saute)")
        rap.valeur('shap_calcule', False)
        return

    print("\n--- Section 6bis : valeurs SHAP (modele de la derniere fenetre) ---")
    try:
        import shap
    except ImportError:
        print("Le package 'shap' n'est pas installe -- calcul SHAP saute "
              "(le reste de l'etape n'est pas affecte). Installe-le avec : pip install shap")
        rap.valeur('shap_calcule', False)
        return

    predicteurs = config.PREDICTEURS
    derniere_fenetre = sortie['liste_fenetres'][-1]
    donnees = fenetres.preparer_fenetre(
        panel, derniere_fenetre, predicteurs, config.MACRO_PREDICTEURS, config.CIBLE)

    X_test = donnees['X_test']
    taille = min(taille_max, len(X_test))
    X_echantillon = X_test.sample(n=taille, random_state=0)

    debut = time.perf_counter()
    explainer = shap.TreeExplainer(sortie['modele_final'])
    if check_additivity:
        valeurs_shap = explainer.shap_values(X_echantillon)
    else:
        valeurs_shap = explainer.shap_values(X_echantillon, check_additivity=False)

    rap.valeur('shap_calcule', True)
    rap.valeur('shap_taille_echantillon', int(taille))
    rap.valeur('shap_fenetre', int(derniere_fenetre['numero']))
    rap.table('shap_valeurs', pd.DataFrame(valeurs_shap, columns=predicteurs,
                                           index=X_echantillon.index))
    rap.table('shap_echantillon_X', X_echantillon)

    duree = f" ({time.perf_counter() - debut:.1f} s)" if chronometre else ""
    print(f"Valeurs SHAP calculees sur {taille} lignes du test de la derniere fenetre{duree}.")
