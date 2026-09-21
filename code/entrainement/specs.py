"""
Une SPECIFICATION par modele : tout ce qui distingue un modele des trois autres, et
rien d'autre.

Le principe
-----------
Les quatre modeles du projet suivent exactement le meme protocole : memes fenetres, meme
recherche d'hyperparametres sur la validation de chaque fenetre, meme R2_oos pooled, memes
fichiers de sortie, meme ligne au journal. Cette mecanique-la vit dans `boucle.py`, ecrite
UNE FOIS.

Ce qui varie tient en une poignee de points, et c'est ce que chaque classe ci-dessous
declare :

  - `grille()`          : les combinaisons d'hyperparametres a essayer
  - `construire(hp)`    : comment fabriquer l'estimateur a partir d'une combinaison
  - `ajuster(...)`      : comment l'entrainer (LightGBM a besoin d'un eval_set)
  - `ligne_grille(...)` : la ligne de diagnostic ecrite pour CHAQUE combinaison essayee
  - `ligne_fenetre(...)`: la ligne de resultat ecrite pour la combinaison RETENUE
  - `apres_fenetre(...)`: ce qu'on garde de chaque fenetre (coefficients, importances)
  - `analyses(...)`     : les analyses propres au modele (Fama-MacBeth, SHAP, stabilite)

⚠️ Les colonnes et leur ORDRE reproduisent exactement ceux des anciens scripts etape04 a
etape07 : les notebooks 04 a 07 lisent ces tableaux tels quels.

ℹ️ La regression lineaire est traitee comme un cas particulier trivial : une grille a UNE
seule combinaison, vide. Elle n'a donc plus besoin de son propre code d'entrainement.
"""

import warnings
from itertools import product

import config
import chemins

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression

import config
from fenetres import r2_oos

from . import analyses


def _grille(**axes):
    """{'alpha': [1, 2], 'l1_ratio': [.1, .9]} -> [{'alpha': 1, 'l1_ratio': .1}, ...]

    Le produit cartesien est parcouru dans l'ordre des axes declares, donc dans le MEME
    ordre que les boucles `for` imbriquees des anciens scripts : les tableaux de grille
    sortent ligne pour ligne identiques.
    """
    noms = list(axes)
    return [dict(zip(noms, valeurs)) for valeurs in product(*axes.values())]


# ============================================================
# Classe de base : le comportement commun, surcharge au besoin
# ============================================================

class Modele:
    """Contrat que `boucle.py` attend d'un modele. Chaque methode a un defaut raisonnable ;
    une sous-classe ne surcharge que ce qui la distingue."""

    cle = None          # 'elastic_net' -- nom canonique : fichiers, rapports, config
    libelle = None      # 'Elastic Net' -- nom affiche, et cle du journal des experiences
    rapport_h1 = None       # nom du rapport a l'horizon 1 mois
    rapport_horizon = None  # nom du rapport a l'horizon long
    avec_grille = True  # False = pas de tableau `grille_complete` dans le rapport
    recherche_par_etapes = False
    # True = la recherche d'hyperparametres se fait en plusieurs passes successives (voir
    # `etape_suivante` ci-dessous), et `boucle` ajoute alors une colonne 'etape' au tableau
    # `grille_complete`. Seul le Random Forest s'en sert.

    # ---------- grille et estimateur ----------

    def grille(self):
        """Liste de dicts d'hyperparametres a essayer sur chaque fenetre.

        Avec `recherche_par_etapes = True`, c'est la grille de la PREMIERE etape seulement.
        """
        raise NotImplementedError

    def etape_suivante(self, numero_etape, hp_meilleur, deja_evaluees):
        """Combinaisons a essayer APRES l'etape `numero_etape` (0 = la grille de depart).

        Appelee UNE FOIS PAR FENETRE ET PAR ETAPE, avec :
          - `hp_meilleur`     : la meilleure combinaison de la fenetre toutes etapes
                                confondues jusqu'ici (meilleur R2_oos de validation) ;
          - `deja_evaluees`   : toutes les combinaisons deja essayees dans cette fenetre.

        Renvoie une liste de combinaisons, ou None / [] pour arreter la recherche. Le defaut
        arrete tout de suite : une seule etape, donc le produit cartesien classique, et le
        comportement historique du projet est strictement inchange pour les 3 autres modeles.
        """
        return None

    def construire(self, hp, final=False, conserve=False):
        """Estimateur non entraine pour la combinaison `hp`.

        `final`    : True s'il s'agit du modele RETENU de la fenetre, reconstruit apres la
                     grille. `conserve` : True si les candidats de la grille sont gardes en
                     memoire (auquel cas l'un d'eux DEVIENDRA le modele retenu).
        Seul le Random Forest s'en sert, pour savoir s'il doit calculer son score
        out-of-bag au moment du fit -- les trois autres ignorent ces deux drapeaux.
        """
        raise NotImplementedError

    def ajuster(self, modele, d):
        """Entraine `modele` sur le train de la fenetre. `d` est le dict renvoye par
        `fenetres.preparer_fenetre` (X_train/y_train/X_validation/...)."""
        modele.fit(d['X_train'], d['y_train'])

    def conserver_candidats(self, contexte):
        """True = les modeles de la grille sont gardes en memoire et la gagnante est
        reprise telle quelle ; False = seuls les scores sont gardes et la gagnante est
        ré-entrainee. Aucun effet sur les resultats (random_state fixe partout)."""
        return False

    # ---------- lignes de diagnostic ----------

    def ligne_grille(self, hp, modele, scores):
        """Ligne du tableau `grille_complete`, pour UNE combinaison essayee.
        `boucle` prefixe deja 'fenetre' et 'annee_test', et ajoute 'selectionnee' apres."""
        return dict(hp, **scores)

    def ligne_fenetre(self, ligne, modele, d, r2):
        """Ligne du tableau `resultats_par_fenetre`, pour la combinaison RETENUE.
        `boucle` prefixe deja 'fenetre', 'annee_test' et 'n_train'.
        `r2` = {'r2_oos_train', 'r2_oos_validation', 'r2_oos_test'}."""
        return dict(r2)

    def ligne_journal(self, ligne, modele, d):
        """Partie VARIABLE de la ligne affichee a l'ecran a la fin de chaque fenetre.
        `boucle` l'encadre deja par "Fenetre N (test AAAA) : " et "R2_oos test = ...".
        Doit se terminer par " | " si elle n'est pas vide."""
        return ""

    # ---------- collecte et rapport ----------

    def preparer(self, panel, liste_fenetres, combinaisons, rap):
        """Appelee une fois AVANT la boucle : prints d'annonce, valeurs de rapport,
        et retour d'un `contexte` libre passe ensuite a `conserver_candidats`."""
        return {}

    def apres_fenetre(self, etat, modele, d, predicteurs):
        """Accumule ce qu'on veut garder de chaque fenetre dans `etat` (un dict de listes,
        cree vide par boucle et reverse tel quel dans la `sortie`)."""

    def apres_boucle(self, sortie, grille_complete, rap, predicteurs):
        """Diagnostics finaux : importance/coefficients de la derniere fenetre, statistiques
        sur la grille complete."""

    def params_specifiques(self):
        """Hyperparametres propres a CE modele, tels qu'enregistres au journal des
        experiences. Ils entrent dans la cle d'unicite de l'experience."""
        return {}

    # ---------- analyses annexes ----------

    def analyses(self, panel, sortie, rap, options):
        """Analyses propres au modele, executees APRES la sauvegarde (Fama-MacBeth,
        stabilite de selection, importance agregee, SHAP).

        ⚠️ Elles ne tournent qu'a l'horizon 1 mois -- voir boucle.executer. C'est le
        comportement des anciens scripts etape11 a etape14, qui n'appelaient que
        `entrainer()`.
        """


# ============================================================
# Regression lineaire (ex-etape04)
# ============================================================

class RegressionLineaire(Modele):
    cle = 'regression_lineaire'
    libelle = 'Regression lineaire'
    rapport_h1 = '04_regression_lineaire'
    rapport_horizon = '11_horizon_lineaire'
    avec_grille = False   # OLS n'a aucun hyperparametre : pas de tableau de grille

    def grille(self):
        # Une seule "combinaison", vide : la boucle generique s'occupe du reste.
        return [{}]

    def construire(self, hp, final=False, conserve=False):
        return LinearRegression()

    def ligne_journal(self, ligne, modele, d):
        return f"n_train={len(d['y_train']):>8,} | "

    def apres_fenetre(self, etat, modele, d, predicteurs):
        # AJOUT : on garde les coefficients de CHAQUE fenetre (et pas seulement de la
        # derniere), pour la figure 5bis du notebook 04. Inclut les predicteurs macro.
        etat.setdefault('coefficients_par_fenetre', []).append(
            dict(zip(predicteurs, modele.coef_)))

    def apres_boucle(self, sortie, grille_complete, rap, predicteurs):
        coefficients = pd.Series(sortie['modele_final'].coef_, index=predicteurs)
        coefficients = coefficients.sort_values(key=abs, ascending=False)
        rap.table('coefficients_derniere_fenetre', coefficients.rename('coefficient'))
        rap.valeur('n_coefficients_a_zero', int((coefficients == 0).sum()))

    def analyses(self, panel, sortie, rap, options):
        # AJOUT : matrice fenetre x predicteur des coefficients, sur disque
        coefficients_fenetres = pd.DataFrame(sortie['coefficients_par_fenetre'])
        coefficients_fenetres.index.name = 'numero_fenetre'
        coefficients_fenetres.to_parquet(chemins.sortie('coefficients_par_fenetre_lineaire'))
        print("Coefficients par fenetre sauvegardes :",
              chemins.sortie('coefficients_par_fenetre_lineaire'))

        if options.get('sans_fama_macbeth'):
            print("\n(--sans-fama-macbeth : section 6bis sautee)")
            rap.valeur('fama_macbeth_execute', False)
            return
        analyses.fama_macbeth(panel, rap)
        rap.valeur('fama_macbeth_execute', True)


# ============================================================
# Elastic Net (ex-etape05)
# ============================================================

class ElasticNetModele(Modele):
    cle = 'elastic_net'
    libelle = 'Elastic Net'
    rapport_h1 = '05_elastic_net'
    rapport_horizon = '12_horizon_elastic_net'

    def grille(self):
        return _grille(alpha=config.GRILLE_ALPHA_ELASTIC_NET,
                       l1_ratio=config.GRILLE_L1_RATIO_ELASTIC_NET)

    def preparer(self, panel, liste_fenetres, combinaisons, rap):
        # sklearn previent a chaque modele que la descente de coordonnees n'a pas converge
        # sur les alphas les plus faibles : le message serait repete des milliers de fois.
        warnings.filterwarnings('ignore', category=UserWarning)
        return {}

    def construire(self, hp, final=False, conserve=False):
        return ElasticNet(alpha=hp['alpha'], l1_ratio=hp['l1_ratio'],
                          max_iter=config.MAX_ITER_ELASTIC_NET, random_state=0)

    def ligne_grille(self, hp, modele, scores):
        return {
            'alpha': hp['alpha'],
            'l1_ratio': hp['l1_ratio'],
            **scores,
            'n_coefficients_non_nuls': int((modele.coef_ != 0).sum()),
        }

    def ligne_fenetre(self, ligne, modele, d, r2):
        return {'alpha': ligne['alpha'], 'l1_ratio': ligne['l1_ratio'], **r2}

    def ligne_journal(self, ligne, modele, d):
        return f"alpha={ligne['alpha']:.1e}, l1_ratio={ligne['l1_ratio']:.1f} | "

    def apres_fenetre(self, etat, modele, d, predicteurs):
        etat.setdefault('coefficients_par_fenetre', []).append(
            dict(zip(predicteurs, modele.coef_)))

    def apres_boucle(self, sortie, grille_complete, rap, predicteurs):
        coefficients = pd.Series(sortie['modele_final'].coef_, index=predicteurs)
        coefficients = coefficients.sort_values(key=abs, ascending=False)
        rap.table('coefficients_derniere_fenetre', coefficients.rename('coefficient'))
        rap.valeur('n_coefficients_a_zero', int((coefficients == 0).sum()))

    def params_specifiques(self):
        return {
            'grille_alpha': config.GRILLE_ALPHA_ELASTIC_NET,
            'grille_l1_ratio': config.GRILLE_L1_RATIO_ELASTIC_NET,
            'max_iter': config.MAX_ITER_ELASTIC_NET,
        }

    def analyses(self, panel, sortie, rap, options):
        analyses.stabilite_selection(sortie, rap)


# ============================================================
# LightGBM (ex-etape06)
# ============================================================

class LightGBM(Modele):
    cle = 'lightgbm'
    libelle = 'LightGBM'
    rapport_h1 = '06_lightgbm'
    rapport_horizon = '13_horizon_lightgbm'

    def grille(self):
        return _grille(num_leaves=config.GRILLE_NUM_LEAVES_LIGHTGBM,
                       learning_rate=config.GRILLE_LEARNING_RATE_LIGHTGBM,
                       min_child_samples=config.GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM,
                       n_estimators=config.GRILLE_N_ESTIMATORS_LIGHTGBM)

    def preparer(self, panel, liste_fenetres, combinaisons, rap):
        n = len(combinaisons) * len(liste_fenetres)
        print(f"Grille : {len(combinaisons)} combinaisons x {len(liste_fenetres)} fenetres "
              f"= {n} entrainements. Sois patient.")
        return {}

    def construire(self, hp, final=False, conserve=False):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=hp['n_estimators'],   # budget max d'arbres, arret anticipe possible
            num_leaves=hp['num_leaves'],
            learning_rate=hp['learning_rate'],
            min_child_samples=hp['min_child_samples'],
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            importance_type='gain',
            random_state=0,
            verbose=-1,
        )

    def ajuster(self, modele, d):
        import lightgbm as lgb
        modele.fit(
            d['X_train'], d['y_train'],
            eval_set=[(d['X_validation'], d['y_validation'])],
            eval_metric='l2',
            callbacks=[lgb.early_stopping(
                stopping_rounds=config.STOPPING_ROUNDS_LIGHTGBM, verbose=False)],
        )

    def conserver_candidats(self, contexte):
        # Les boosters sont petits : les garder evite un ré-entrainement par fenetre.
        return True

    def ligne_grille(self, hp, modele, scores):
        nb_arbres = modele.best_iteration_ or hp['n_estimators']
        return {
            'num_leaves': hp['num_leaves'],
            'learning_rate': hp['learning_rate'],
            'min_child_samples': hp['min_child_samples'],
            'n_estimators': hp['n_estimators'],
            'nb_arbres_utilises': nb_arbres,
            'budget_atteint': nb_arbres >= hp['n_estimators'],
            **scores,
        }

    def ligne_fenetre(self, ligne, modele, d, r2):
        return {
            'num_leaves': ligne['num_leaves'],
            'learning_rate': ligne['learning_rate'],
            'min_child_samples': ligne['min_child_samples'],
            'n_estimators': ligne['n_estimators'],
            'nb_arbres_utilises': ligne['nb_arbres_utilises'],
            **r2,
        }

    def ligne_journal(self, ligne, modele, d):
        return (f"num_leaves={int(ligne['num_leaves']):>2d}, "
                f"lr={ligne['learning_rate']:.3f}, "
                f"arbres={int(ligne['nb_arbres_utilises']):>4d} | ")

    def apres_fenetre(self, etat, modele, d, predicteurs):
        # Le gain est normalise a 100 par fenetre pour rester comparable malgre des
        # nombres d'arbres differents d'une fenetre a l'autre.
        gain = modele.booster_.feature_importance(importance_type='gain').astype(float)
        if gain.sum() > 0:
            gain = gain / gain.sum() * 100
        split = modele.booster_.feature_importance(importance_type='split').astype(float)
        etat.setdefault('importances_gain_par_fenetre', []).append(dict(zip(predicteurs, gain)))
        etat.setdefault('importances_split_par_fenetre', []).append(dict(zip(predicteurs, split)))

    def apres_boucle(self, sortie, grille_complete, rap, predicteurs):
        # Diagnostic : si le budget d'arbres n'est JAMAIS atteint, toutes les valeurs de
        # GRILLE_N_ESTIMATORS_LIGHTGBM donnent le meme modele (l'arret anticipe coupe avant).
        rap.valeur('pct_budget_arbres_atteint',
                   float(grille_complete['budget_atteint'].mean() * 100))
        importances = pd.Series(sortie['modele_final'].feature_importances_,
                                index=predicteurs).sort_values(ascending=False)
        rap.table('importance_derniere_fenetre', importances.rename('importance_gain'))

    def params_specifiques(self):
        return {
            'grille_num_leaves': config.GRILLE_NUM_LEAVES_LIGHTGBM,
            'grille_learning_rate': config.GRILLE_LEARNING_RATE_LIGHTGBM,
            'grille_min_child_samples': config.GRILLE_MIN_CHILD_SAMPLES_LIGHTGBM,
            'grille_n_estimators': config.GRILLE_N_ESTIMATORS_LIGHTGBM,
            'stopping_rounds': config.STOPPING_ROUNDS_LIGHTGBM,
        }

    def analyses(self, panel, sortie, rap, options):
        analyses.importance_agregee_lightgbm(sortie, rap)
        analyses.calculer_shap(panel, sortie, rap, taille_max=2000,
                               saute=options.get('sans_shap'), check_additivity=True)


# ============================================================
# Random Forest (ex-etape07)
# ============================================================

class RandomForest(Modele):
    """⚠️ Seul modele du projet dont la recherche d'hyperparametres n'est PAS un produit
    cartesien : `max_features` est cherche dans une SECONDE etape, une fois les trois autres
    axes arbitres. Voir config.py, section "max_features : recherche EN DEUX TEMPS", pour la
    justification et le calcul du cout.
    """

    cle = 'random_forest'
    libelle = 'Random Forest'
    rapport_h1 = '07_random_forest'
    rapport_horizon = '14_horizon_random_forest'
    recherche_par_etapes = True

    # ---- la recherche en deux etapes ----

    @staticmethod
    def _valeurs_etape_b():
        """Valeurs de `max_features` a essayer a l'etape B, celle de l'etape A retiree.

        La comparaison passe par `str` : elle traite correctement les valeurs textuelles
        ('sqrt', 'log2') a cote des fractions, et evite de comparer deux flottants avec `==`.
        """
        pivot = str(config.MAX_FEATURES_ETAPE_A_RANDOM_FOREST)
        vues = {pivot}
        valeurs = []
        for v in config.GRILLE_MAX_FEATURES_RANDOM_FOREST:
            if str(v) not in vues:      # ne JAMAIS refaire une valeur deja evaluee
                vues.add(str(v))
                valeurs.append(v)
        return valeurs

    def grille(self):
        """ETAPE A : `max_features` fige au pivot, produit cartesien des trois autres axes."""
        return _grille(n_estimators=config.GRILLE_N_ESTIMATORS_RANDOM_FOREST,
                       max_depth=config.GRILLE_MAX_DEPTH_RANDOM_FOREST,
                       max_features=[config.MAX_FEATURES_ETAPE_A_RANDOM_FOREST],
                       min_samples_leaf=config.GRILLE_MIN_SAMPLES_LEAF_RANDOM_FOREST)

    def etape_suivante(self, numero_etape, hp_meilleur, deja_evaluees):
        """ETAPE B : le triplet gagnant de l'etape A est fige, seul `max_features` varie.

        ⚠️ `hp_meilleur` est le gagnant DE CETTE FENETRE : le triplet fige a l'etape B peut
        donc differer d'une fenetre a l'autre, ce qui est bien le comportement voulu (chaque
        fenetre ré-arbitre ses hyperparametres sur SA propre validation).

        Il n'y a que deux etapes : au-dela, on renvoie None et la recherche s'arrete.
        """
        if numero_etape >= 1:
            return None
        return [dict(hp_meilleur, max_features=v) for v in self._valeurs_etape_b()]

    # ---- estimation memoire : garder toute la grille en RAM, ou ré-entrainer ? ----

    @staticmethod
    def _memoire_estimee_mo(n_train, triplets):
        """Estime (en Mo) la RAM occupee si TOUTES les forets de la grille sont gardees en
        memoire pendant une fenetre de `n_train` lignes.

        1. scikit-learn stocke un arbre comme 8 tableaux paralleles de 8 octets par noeud :
           64 octets par noeud, exactement.
        2. Un arbre binaire a F feuilles a 2F - 1 noeuds.
        3. F est plafonne par DEUX contraintes, et c'est la PLUS SERREE qui s'applique :
           la profondeur (F <= 2^max_depth) et min_samples_leaf (les feuilles se partagent
           les lignes du train sans recouvrement, et une coupure n'est acceptee que si elle
           donne DEUX enfants d'au moins min_samples_leaf lignes -- d'ou le facteur 2).

        Estimation volontairement GENEREUSE : mieux vaut basculer sur le ré-entrainement
        pour rien que planter au milieu d'une fenetre.
        """
        total_octets = 0
        for n_estimators, max_depth, min_samples_leaf in triplets:
            feuilles = n_train / (2 * max(min_samples_leaf, 1))
            if max_depth is not None:
                feuilles = min(feuilles, 2 ** min(max_depth, 30))  # garde-fou anti-overflow
            noeuds = 2 * feuilles - 1
            total_octets += n_estimators * noeuds * 64
        return total_octets / 1e6

    def preparer(self, panel, liste_fenetres, combinaisons, rap):
        n_etape_a = len(combinaisons)
        valeurs_b = self._valeurs_etape_b()
        n_etape_b = len(valeurs_b)
        n_combinaisons = n_etape_a + n_etape_b   # forets construites par fenetre

        # La plus grande fenetre d'entrainement decide du pire cas memoire (en "expanding"
        # c'est la derniere ; on prend le max pour ne dependre d'aucun mode).
        n_train_maximum = max(
            int(panel['annee_mois'].isin(f['train']).sum()) for f in liste_fenetres)
        # ⚠️ max_features ne figure pas dans le triplet : il ne change ni la taille des
        # arbres ni leur nombre, seulement le choix des coupures. C'est precisement ce qui
        # permet d'estimer la memoire de l'etape B sans savoir quel triplet elle heritera.
        triplets = [(hp['n_estimators'], hp['max_depth'], hp['min_samples_leaf'])
                    for hp in _grille(n_estimators=config.GRILLE_N_ESTIMATORS_RANDOM_FOREST,
                                      max_depth=config.GRILLE_MAX_DEPTH_RANDOM_FOREST,
                                      min_samples_leaf=config.GRILLE_MIN_SAMPLES_LEAF_RANDOM_FOREST)]
        # Les forets de l'etape B reprennent le triplet gagnant de l'etape A, inconnu a ce
        # stade : on suppose le PIRE (le triplet le plus gourmand), fidele a l'esprit
        # volontairement genereux de cette estimation.
        if n_etape_b:
            pire = max(triplets, key=lambda t: self._memoire_estimee_mo(n_train_maximum, [t]))
            triplets = triplets + [pire] * n_etape_b
        memoire_mo = self._memoire_estimee_mo(n_train_maximum, triplets)
        plafond_mo = config.PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO

        garder = config.GARDER_CANDIDATS_RANDOM_FOREST
        if garder and memoire_mo > plafond_mo:
            garder = False
            print(f"⚠️ GARDER_CANDIDATS_RANDOM_FOREST = True, mais garder toute la grille en "
                  f"memoire est estime a ~{memoire_mo:,.0f} Mo sur la plus grande fenetre "
                  f"({n_train_maximum:,} lignes), au-dela du plafond de {plafond_mo:,} Mo.")
            print("   -> Bascule automatique sur le ré-entrainement de la gagnante (1 entrainement")
            print("      de plus par fenetre, une seule foret en memoire a la fois). Resultats")
            print("      strictement identiques. Pour forcer l'autre comportement : augmente")
            print("      PLAFOND_MEMOIRE_CANDIDATS_RANDOM_FOREST_MO dans config.py.")

        if garder:
            entrainements = n_combinaisons * len(liste_fenetres)
            mode = (f"les {n_combinaisons} forets de chaque fenetre sont gardees en memoire "
                    f"(~{memoire_mo:,.0f} Mo estimes), la gagnante est reprise telle quelle")
        else:
            entrainements = (n_combinaisons + 1) * len(liste_fenetres)
            mode = ("une seule foret en memoire a la fois ; la gagnante de chaque fenetre est "
                    "ré-entrainee, soit 1 entrainement de plus par fenetre")

        n_cartesien = n_etape_a * (1 + n_etape_b)
        print(f"Recherche en DEUX ETAPES (max_features cherche a part) :")
        print(f"  etape A : max_features fige a {config.MAX_FEATURES_ETAPE_A_RANDOM_FOREST}, "
              f"{n_etape_a} combinaisons (n_estimators x max_depth x min_samples_leaf)")
        if n_etape_b:
            print(f"  etape B : triplet gagnant fige, max_features parmi "
                  f"{valeurs_b} -> {n_etape_b} combinaisons")
        else:
            print("  etape B : aucune valeur supplementaire de max_features -> etape sautee "
                  "(GRILLE_MAX_FEATURES_RANDOM_FOREST ne contient que la valeur de l'etape A)")
        print(f"  soit {n_combinaisons} forets par fenetre, contre {n_cartesien} en produit "
              f"cartesien complet.")
        print(f"Grille : {n_combinaisons} combinaisons x {len(liste_fenetres)} fenetres "
              f"= {entrainements} forets a construire. Sois patient.")
        print(f"Mode : {mode}.")
        rap.valeur('recherche_max_features_deux_temps', True)
        rap.valeur('n_combinaisons_etape_a', int(n_etape_a))
        rap.valeur('n_combinaisons_etape_b', int(n_etape_b))
        rap.valeur('n_combinaisons_produit_cartesien', int(n_cartesien))
        rap.valeur('max_features_etape_a', str(config.MAX_FEATURES_ETAPE_A_RANDOM_FOREST))
        rap.valeur('garder_candidats', bool(garder))
        rap.valeur('garder_candidats_demande', bool(config.GARDER_CANDIDATS_RANDOM_FOREST))
        rap.valeur('memoire_grille_estimee_mo', float(memoire_mo))
        rap.valeur('n_train_maximum', int(n_train_maximum))
        return {'garder_candidats': garder}

    def conserver_candidats(self, contexte):
        return contexte['garder_candidats']

    def construire(self, hp, final=False, conserve=False):
        """Foret configuree comme le reste du projet : meme random_state partout (resultats
        reproductibles), bootstrap actif (indispensable au bagging et au calcul out-of-bag).

        `oob_score` n'est demande pour un candidat de la grille que si AUCUN ré-entrainement
        ne viendra ensuite : il doit etre calcule au moment du fit, et ne sert que pour la
        foret finalement retenue.
        """
        oob = config.OOB_SCORE_RANDOM_FOREST and (final or conserve)
        return RandomForestRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            max_features=hp['max_features'],
            min_samples_leaf=hp['min_samples_leaf'],
            max_samples=config.MAX_SAMPLES_RANDOM_FOREST,
            bootstrap=True,
            oob_score=oob,
            n_jobs=config.N_JOBS_RANDOM_FOREST,
            random_state=0,
        )

    def ligne_grille(self, hp, modele, scores):
        return {
            'n_estimators': hp['n_estimators'],
            # None (arbres complets) n'est pas representable en Parquet a cote d'entiers :
            # -1 sert de code pour "aucune limite", et le notebook le reaffiche comme 'None'.
            'max_depth': -1 if hp['max_depth'] is None else int(hp['max_depth']),
            'max_features': str(hp['max_features']),
            'min_samples_leaf': hp['min_samples_leaf'],
            'profondeur_moyenne_arbres': float(np.mean(
                [arbre.get_depth() for arbre in modele.estimators_])),
            'feuilles_moyennes_arbres': float(np.mean(
                [arbre.get_n_leaves() for arbre in modele.estimators_])),
            **scores,
        }

    def ligne_fenetre(self, ligne, modele, d, r2):
        # R2_oos out-of-bag : chaque ligne du train predite UNIQUEMENT par les arbres qui ne
        # l'ont pas vue. Estimation hors-echantillon gratuite, propre au bagging.
        r2_oob = np.nan
        if config.OOB_SCORE_RANDOM_FOREST and hasattr(modele, 'oob_prediction_'):
            oob = modele.oob_prediction_
            valides = ~np.isnan(oob)   # une ligne vue par TOUS les arbres n'a pas de prediction oob
            if valides.sum() > 0:
                r2_oob = r2_oos(d['y_train'].values[valides], oob[valides])
        return {
            'n_estimators': int(ligne['n_estimators']),
            'max_depth': int(ligne['max_depth']),
            'max_features': str(ligne['max_features']),
            'min_samples_leaf': int(ligne['min_samples_leaf']),
            'profondeur_moyenne_arbres': float(ligne['profondeur_moyenne_arbres']),
            'r2_oos_train': r2['r2_oos_train'],
            'r2_oos_oob': r2_oob,
            'r2_oos_validation': r2['r2_oos_validation'],
            'r2_oos_test': r2['r2_oos_test'],
        }

    def ligne_journal(self, ligne, modele, d):
        # max_features est affiche parce que c'est desormais l'axe interessant : il permet de
        # voir d'un coup d'oeil si l'etape B a fait bouger quelque chose, fenetre par fenetre.
        return (f"max_depth={ligne['max_depth']:>3d}, "
                f"arbres={int(ligne['n_estimators']):>4d}, "
                f"min_leaf={int(ligne['min_samples_leaf']):>5d}, "
                f"max_feat={str(ligne['max_features']):>5s} | ")

    def apres_fenetre(self, etat, modele, d, predicteurs):
        # Importance MDI, normalisee a 100 (scikit-learn la renvoie normalisee a 1) : memes
        # ordres de grandeur que le gain LightGBM, lui aussi ramene a 100 par fenetre.
        etat.setdefault('importances_par_fenetre', []).append(
            dict(zip(predicteurs, modele.feature_importances_.astype(float) * 100)))

    def apres_boucle(self, sortie, grille_complete, rap, predicteurs):
        rap.valeur('oob_score_actif', bool(config.OOB_SCORE_RANDOM_FOREST))
        # Diagnostic : une profondeur moyenne systematiquement inferieure au plafond demande
        # signifie que max_depth ne mord jamais (min_samples_leaf arrete les arbres avant),
        # donc que toutes les valeurs de la grille donnent le meme modele.
        sans_limite = grille_complete['max_depth'] == -1
        mord = (~sans_limite) & (grille_complete['profondeur_moyenne_arbres']
                                 >= grille_complete['max_depth'] - 0.05)
        rap.valeur('pct_max_depth_mord', float(mord[~sans_limite].mean() * 100)
                   if (~sans_limite).any() else float('nan'))

        # Diagnostic PROPRE A LA RECHERCHE EN DEUX TEMPS : dans combien de fenetres l'etape B
        # a-t-elle reellement fait mieux que le pivot ? Une valeur nulle ou tres basse veut
        # dire que MAX_FEATURES_ETAPE_A_RANDOM_FOREST domine partout -- soit c'est un bon
        # choix, soit les valeurs de l'etape B sont mal placees. Une valeur elevee justifie
        # au contraire d'avoir cherche ce parametre. Dans les deux cas, c'est une phrase pour
        # le memoire.
        if 'etape' in grille_complete.columns:
            gagnantes = grille_complete[grille_complete['selectionnee']]
            rap.valeur('pct_fenetres_gagnees_par_etape_b',
                       float((gagnantes['etape'] != 'A').mean() * 100))
            rap.table('choix_par_fenetre_max_features',
                      gagnantes.set_index('fenetre')[
                          ['annee_test', 'etape', 'max_features', 'max_depth',
                           'min_samples_leaf', 'r2_oos_validation']])
        importances = pd.Series(sortie['modele_final'].feature_importances_ * 100,
                                index=predicteurs).sort_values(ascending=False)
        rap.table('importance_derniere_fenetre', importances.rename('importance_mdi_pct'))

    def params_specifiques(self):
        # ⚠️ 'recherche_max_features' et 'max_features_etape_a' entrent dans la cle d'unicite
        # de l'experience : sans eux, une recherche en deux temps et un produit cartesien sur
        # les MEMES grilles produiraient la meme cle et se confondraient au journal, alors
        # qu'ils n'explorent pas le meme espace et peuvent retenir des modeles differents.
        return {
            'grille_n_estimators': config.GRILLE_N_ESTIMATORS_RANDOM_FOREST,
            'grille_max_depth': [str(v) for v in config.GRILLE_MAX_DEPTH_RANDOM_FOREST],
            'recherche_max_features': 'deux_temps',
            'max_features_etape_a': str(config.MAX_FEATURES_ETAPE_A_RANDOM_FOREST),
            'grille_max_features': [str(v) for v in self._valeurs_etape_b()],
            'grille_min_samples_leaf': config.GRILLE_MIN_SAMPLES_LEAF_RANDOM_FOREST,
            'max_samples': config.MAX_SAMPLES_RANDOM_FOREST,
        }

    def analyses(self, panel, sortie, rap, options):
        analyses.importance_agregee_random_forest(sortie, rap)
        # ⚠️ SHAP est bien plus lent ici que sur LightGBM (arbres bien plus gros) : d'ou un
        # echantillon plus petit, et --sans-shap a utiliser sans hesiter.
        analyses.calculer_shap(panel, sortie, rap, taille_max=500,
                               saute=options.get('sans_shap'), check_additivity=False,
                               chronometre=True)


# ============================================================
# Registre : la liste des modeles connus du projet
#
# Ajouter un 5e modele = ajouter une classe ci-dessus et une entree ici. Rien d'autre :
# ni chemins (chemins.py les engendre), ni script d'entrainement (boucle.py est generique),
# ni ligne dans config.py. Seuls deux fichiers de lancement sont a creer, de 6 lignes.
# ============================================================

LINEAIRE = RegressionLineaire()
ELASTIC_NET = ElasticNetModele()
LIGHTGBM = LightGBM()
RANDOM_FOREST = RandomForest()

TOUS = [LINEAIRE, ELASTIC_NET, LIGHTGBM, RANDOM_FOREST]
PAR_CLE = {m.cle: m for m in TOUS}
