"""
LA boucle d'entrainement du projet -- ecrite une fois, pour les 4 modeles et les 2 horizons.

Ce que ce module remplace
-------------------------
Les fonctions `entrainer()` / `sauvegarder()` / `main()` des anciens scripts etape04 a
etape07 (1 679 lignes, dont environ 70 % strictement identiques d'un fichier a l'autre), et
la section C de l'ancien horizon.py, qui rejouait la meme sauvegarde une seconde fois pour
les scripts etape11 a etape14.

Le protocole -- identique pour tous, c'est bien le probleme qu'il posait :

    pour chaque fenetre :
        essayer chaque combinaison d'hyperparametres, la noter sur la validation
        retenir la meilleure, predire le test
    empiler les predictions de test de toutes les fenetres -> une serie hors-echantillon
    calculer le R2_oos pooled, sauvegarder, inscrire l'experience au journal

Tout ce qui distingue un modele d'un autre est declare dans `specs.py`. Un cinquieme modele
ne demande donc AUCUNE ligne ici.

L'horizon est un simple argument
--------------------------------
`horizon=1` (piste principale) et `horizon=12` (piste longue) empruntent litteralement le
meme chemin de code : seuls la cible, l'embargo aux frontieres des blocs et le suffixe des
fichiers changent, et les trois sont derives de ce seul argument. Une correction apportee a
la boucle profite donc automatiquement aux deux pistes -- ce qui n'etait pas le cas quand
elles vivaient dans huit fichiers separes.

⚠️ Les analyses annexes (Fama-MacBeth, stabilite, importance agregee, SHAP) ne tournent
qu'a l'horizon 1 mois. Voir `analyses.py` pour la justification methodologique ; c'est
aussi le comportement des anciens scripts etape11 a etape14.
"""

import sys
import time

import joblib
import numpy as np
import pandas as pd

import chemins
import config
import fenetres
import horizon as horizon_mod
import journal
import rapports

from . import specs as specs_module


# ============================================================
# 1. La boucle sur les fenetres
# ============================================================

def _libelle_etape(numero):
    """0 -> 'A', 1 -> 'B', ... Sert a etiqueter les lignes du tableau `grille_complete`
    quand un modele cherche ses hyperparametres en plusieurs etapes."""
    if numero < 26:
        return chr(ord('A') + numero)
    return f"E{numero + 1}"


def entrainer(spec, panel, rap):
    """Entraine `spec` fenetre par fenetre et renvoie tout ce qui en sort.

    Retourne un dict `sortie` contenant au minimum : liste_fenetres, resultats_par_fenetre,
    predictions, modele_final, r2 (train, validation, test), duree. Chaque modele peut y
    ajouter ce qu'il a accumule (coefficients ou importances par fenetre) via
    `spec.apres_fenetre`.
    """
    predicteurs = config.PREDICTEURS
    macro_predicteurs = config.MACRO_PREDICTEURS
    cible = config.CIBLE

    liste_fenetres = fenetres.generer_fenetres(
        panel['annee_mois'].unique(),
        type_fenetre=config.TYPE_FENETRE,
        annees_train_initial=config.ANNEES_TRAIN_INITIAL,
        annees_validation=config.ANNEES_VALIDATION,
        annees_test_par_fenetre=config.ANNEES_TEST_PAR_FENETRE,
        reduction_validation_par_fenetre=config.REDUCTION_VALIDATION_PAR_FENETRE,
        fenetre_debut_reduction_validation=config.FENETRE_DEBUT_REDUCTION_VALIDATION,
        annees_validation_minimum=config.ANNEES_VALIDATION_MINIMUM,
    )
    rap.valeur('type_fenetre', config.TYPE_FENETRE)
    rap.valeur('n_fenetres', len(liste_fenetres))
    rap.table('resume_fenetres', fenetres.resumer_fenetres(liste_fenetres))
    print(f"Mode : {config.TYPE_FENETRE} | {len(liste_fenetres)} fenetres generees")

    combinaisons = spec.grille()
    if spec.avec_grille:
        rap.valeur('n_combinaisons_grille', len(combinaisons))

    contexte = spec.preparer(panel, liste_fenetres, combinaisons, rap)

    # Faut-il garder les modeles de la grille en memoire (et reprendre la gagnante telle
    # quelle), ou ne garder que leurs scores et ré-entrainer la gagnante ?
    # ⚠️ Ce choix ne change AUCUN resultat (random_state=0 est fixe partout) : seulement le
    # temps de calcul et la memoire. Avec une grille a UNE combinaison, la question ne se
    # pose pas : ré-entrainer produirait deux fois le meme modele pour rien.
    # ⚠️ Sauf en recherche par etapes : la grille de depart peut compter une seule
    # combinaison et les etapes suivantes en ajouter, auquel cas ce raccourci court-circuiterait
    # le garde-fou memoire du modele.
    conserver = spec.conserver_candidats(contexte) or (
        len(combinaisons) == 1 and not spec.recherche_par_etapes)

    debut_entrainement = time.perf_counter()

    resultats_par_fenetre = []
    predictions_toutes_fenetres = []
    grilles_par_fenetre = []
    pool_train = {'y': [], 'pred': []}
    pool_validation = {'y': [], 'pred': []}
    etat = {}            # ce que chaque modele accumule au fil des fenetres
    modele_final = None

    for f in liste_fenetres:
        donnees = fenetres.preparer_fenetre(panel, f, predicteurs, macro_predicteurs, cible)

        # --- recherche d'hyperparametres sur la validation de CETTE fenetre ---
        # On garde le score de CHAQUE combinaison, sur le train ET sur la validation : le
        # tableau complet est sauvegarde dans le rapport et affiche au notebook (heatmaps +
        # tableau), y compris l'ecart train-validation, qui mesure le sur-apprentissage de
        # chaque combinaison.
        # ⚠️ La recherche se fait EN ETAPES. Par defaut il n'y en a qu'une seule (le produit
        # cartesien complet renvoye par `spec.grille()`), et le comportement est exactement
        # celui d'avant. Un modele peut en declarer d'autres via `spec.etape_suivante` :
        # chaque etape voit alors le MEILLEUR resultat de toutes les etapes deja jouees et
        # decide quelles combinaisons essayer ensuite (descente par coordonnees). Voir
        # specs.RandomForest, qui s'en sert pour `max_features`.
        lignes_grille = []
        candidats = [] if conserver else None
        combinaisons_evaluees = []   # reste aligne, index pour index, avec lignes_grille

        combinaisons_etape = combinaisons
        numero_etape = 0
        while combinaisons_etape:
            libelle_etape = _libelle_etape(numero_etape)

            for hp in combinaisons_etape:
                candidat = spec.construire(hp, conserve=conserver)
                spec.ajuster(candidat, donnees)

                score_train = fenetres.r2_oos(donnees['y_train'],
                                              candidat.predict(donnees['X_train']))
                score_validation = fenetres.r2_oos(donnees['y_validation'],
                                                   candidat.predict(donnees['X_validation']))
                scores = {
                    'r2_oos_train': score_train,
                    'r2_oos_validation': score_validation,
                    'ecart_train_validation': score_train - score_validation,
                }
                ligne = {'fenetre': f['numero'], 'annee_test': f['annee_test']}
                if spec.recherche_par_etapes:
                    ligne['etape'] = libelle_etape
                ligne.update(spec.ligne_grille(hp, candidat, scores))
                lignes_grille.append(ligne)
                combinaisons_evaluees.append(hp)
                if conserver:
                    candidats.append(candidat)

            # Meilleure combinaison TOUTES ETAPES CONFONDUES : c'est elle que l'etape
            # suivante prend pour point de depart, et c'est elle qui sera retenue si aucune
            # etape suivante ne fait mieux.
            index_provisoire = int(np.argmax([l['r2_oos_validation'] for l in lignes_grille]))
            combinaisons_etape = spec.etape_suivante(
                numero_etape, combinaisons_evaluees[index_provisoire], combinaisons_evaluees)
            numero_etape += 1

        grille_fenetre = pd.DataFrame(lignes_grille)
        # Combinaison retenue pour CETTE fenetre : le meilleur R2_oos de validation.
        # ⚠️ Pour selectionner sur l'ecart train-validation plutot que sur le R2_oos brut,
        # c'est CETTE ligne qu'il faut changer -- une seule fois, pour les 4 modeles.
        index_meilleur = grille_fenetre['r2_oos_validation'].idxmax()
        grille_fenetre['selectionnee'] = False
        grille_fenetre.loc[index_meilleur, 'selectionnee'] = True
        grilles_par_fenetre.append(grille_fenetre)

        ligne_retenue = grille_fenetre.loc[index_meilleur]

        # --- modele final de CETTE fenetre ---
        if conserver:
            # Deja entraine ci-dessus : `candidats` est construit dans le MEME ordre que
            # `lignes_grille`, donc les deux restent alignes sur index_meilleur.
            modele = candidats[index_meilleur]
        else:
            # Seuls les scores ont ete gardes : on reconstruit la gagnante a partir de sa
            # combinaison d'origine et on la ré-entraine (identique a la premiere).
            # ⚠️ `combinaisons_evaluees` et non `combinaisons` : avec une recherche par
            # etapes, la gagnante peut venir d'une etape posterieure a la grille de depart.
            modele = spec.construire(combinaisons_evaluees[index_meilleur], final=True)
            spec.ajuster(modele, donnees)

        pred_train = modele.predict(donnees['X_train'])
        pred_validation = modele.predict(donnees['X_validation'])
        pred_test = modele.predict(donnees['X_test'])

        r2 = {
            'r2_oos_train': fenetres.r2_oos(donnees['y_train'], pred_train),
            'r2_oos_validation': fenetres.r2_oos(donnees['y_validation'], pred_validation),
            'r2_oos_test': fenetres.r2_oos(donnees['y_test'], pred_test),
        }
        resultats_par_fenetre.append({
            'fenetre': f['numero'],
            'annee_test': f['annee_test'],
            'n_train': len(donnees['y_train']),
            **spec.ligne_fenetre(ligne_retenue, modele, donnees, r2),
        })

        preds = donnees['id_test'].copy()
        preds[cible] = donnees['y_test'].values
        preds['prediction'] = pred_test
        preds['fenetre'] = f['numero']
        predictions_toutes_fenetres.append(preds)

        pool_train['y'].append(donnees['y_train'].values)
        pool_train['pred'].append(pred_train)
        pool_validation['y'].append(donnees['y_validation'].values)
        pool_validation['pred'].append(pred_validation)

        spec.apres_fenetre(etat, modele, donnees, predicteurs)
        modele_final = modele
        # Les candidates non retenues ne servent plus a rien : on les libere avant de
        # passer a la fenetre suivante, pour ne jamais cumuler DEUX grilles en memoire.
        candidats = None

        print(f"Fenetre {f['numero']:2d} (test {f['annee_test']:>9s}) : "
              f"{spec.ligne_journal(ligne_retenue, modele, donnees)}"
              f"R2_oos test = {r2['r2_oos_test']:.4f}")

    duree = time.perf_counter() - debut_entrainement

    resultats_par_fenetre = pd.DataFrame(resultats_par_fenetre)
    predictions_toutes_fenetres = pd.concat(predictions_toutes_fenetres, ignore_index=True)
    grille_complete = pd.concat(grilles_par_fenetre, ignore_index=True)

    # --- R2_oos pooled (toutes les fenetres mises bout a bout) ---
    r2_train = fenetres.r2_oos(np.concatenate(pool_train['y']),
                               np.concatenate(pool_train['pred']))
    r2_validation = fenetres.r2_oos(np.concatenate(pool_validation['y']),
                                    np.concatenate(pool_validation['pred']))
    r2_test = fenetres.r2_oos(predictions_toutes_fenetres[cible],
                              predictions_toutes_fenetres['prediction'])

    print(f"\nR2_oos train      (pooled) : {r2_train:.4f}")
    print(f"R2_oos validation (pooled) : {r2_validation:.4f}")
    print(f"R2_oos test       (pooled) : {r2_test:.4f}")
    detail = " (recherche d'hyperparametres incluse)" if spec.avec_grille else " (toutes fenetres)"
    print(f"Duree totale d'entrainement{detail} : {duree:.1f} s")

    rap.valeur('duree_entrainement_secondes', float(duree))
    rap.valeur('n_predicteurs', len(predicteurs))
    rap.valeur('n_predictions_test', int(len(predictions_toutes_fenetres)))
    rap.table('apercu_predictions', predictions_toutes_fenetres.head())

    if spec.avec_grille:
        # Une ligne par (fenetre, combinaison) : c'est ce tableau qu'affichent les
        # notebooks 05, 06 et 07 en section 3bis (heatmaps + tableau).
        rap.table('grille_complete', grille_complete)
        rap.valeur('n_modeles_entraines_grille', int(len(grille_complete)))

    sortie = {
        'liste_fenetres': liste_fenetres,
        'resultats_par_fenetre': resultats_par_fenetre,
        'predictions': predictions_toutes_fenetres,
        'grille_complete': grille_complete,
        'modele_final': modele_final,
        'r2': (r2_train, r2_validation, r2_test),
        'duree': duree,
        'params_specifiques': spec.params_specifiques(),
        **etat,
    }

    spec.apres_boucle(sortie, grille_complete, rap, predicteurs)
    return sortie


# ============================================================
# 2. Sauvegardes et journal des experiences
# ============================================================

def sauvegarder(spec, sortie, rap, horizon):
    """Ecrit modele, predictions et resultats, et inscrit l'experience au journal.

    ℹ️ La cle d'experience differe automatiquement entre les deux horizons :
    `journal.params_generaux_actuels()` ajoute la cible et l'embargo a la signature des
    qu'on s'ecarte de `excess_return`. Les deux pistes cohabitent donc dans le journal sans
    jamais se confondre, et le notebook 09 les distingue seul.
    """
    r2_train, r2_validation, r2_test = sortie['r2']
    fichiers = chemins.fichiers_modele(spec.cle, horizon)

    joblib.dump(sortie['modele_final'], fichiers['modele'])
    print("\nModele final (derniere fenetre) sauvegarde :", fichiers['modele'])

    sortie['predictions'].to_parquet(fichiers['predictions'], index=False)
    print("Predictions (toutes fenetres) sauvegardees :", fichiers['predictions'])

    params_specifiques = sortie['params_specifiques']
    # Meme cle que celle calculee par enregistrer_experience ci-dessous : c'est elle que
    # les notebooks relisent pour relier Sharpe/Sortino a la bonne ligne de leur tableau.
    cle_experience = journal.cle_experience_actuelle(spec.libelle, params_specifiques)

    ligne = {'modele': spec.libelle}
    if horizon != 1:
        ligne['horizon_mois'] = horizon
        ligne['cible'] = config.CIBLE
    ligne.update({
        'type_fenetre': config.TYPE_FENETRE,
        'n_fenetres': len(sortie['liste_fenetres']),
        'r2_oos_train': r2_train,
        'r2_oos_validation': r2_validation,
        'r2_oos_test': r2_test,
        'cle_experience': cle_experience,
    })

    pd.DataFrame([ligne]).to_parquet(fichiers['resultats'], index=False)
    sortie['resultats_par_fenetre'].to_parquet(fichiers['resultats_fenetre'], index=False)
    print("Resultats pooled sauvegardes      :", fichiers['resultats'])
    print("Resultats par fenetre sauvegardes :", fichiers['resultats_fenetre'])

    ajoutee = journal.enregistrer_experience(
        modele=spec.libelle,
        params_specifiques=params_specifiques,
        resultats={
            'n_fenetres': len(sortie['liste_fenetres']),
            'r2_oos_train': r2_train,
            'r2_oos_validation': r2_validation,
            'r2_oos_test': r2_test,
        },
        duree_entrainement_secondes=sortie['duree'],
    )

    rap.valeur('cle_experience', cle_experience)
    rap.valeur('experience_ajoutee_au_journal', bool(ajoutee))
    rap.valeur('params_specifiques', params_specifiques)
    rap.valeur('r2_oos_train', float(r2_train))
    rap.valeur('r2_oos_validation', float(r2_validation))
    rap.valeur('r2_oos_test', float(r2_test))
    rap.table('resultats_par_fenetre', sortie['resultats_par_fenetre'])


# ============================================================
# 3. L'execution complete d'un modele -- le seul point d'entree
# ============================================================

def executer(spec, horizon=1, options=None):
    """Entraine `spec` a l'horizon donne, de bout en bout.

    C'est l'unique fonction appelee par les 8 fichiers de scripts/entrainer/.
    """
    options = options or {}
    nom_rapport = spec.rapport_h1 if horizon == 1 else spec.rapport_horizon

    chemins.assurer_dossiers()
    rap = rapports.Rapport(nom_rapport)

    titre = f"{spec.libelle} -- horizon {horizon} mois"
    print("=" * 70)
    print(titre)
    print("=" * 70)

    # ⚠️ Tout se passe DANS ce contexte : c'est lui qui positionne config.CIBLE et
    # l'embargo, et qui les restaure a la sortie meme en cas d'exception. A horizon=1 il
    # ne change rien -- les deux pistes suivent le meme chemin de code.
    with horizon_mod.contexte(horizon):
        panel = pd.read_parquet(chemins.PANEL_MODELISATION)
        print(f"Panel complet : {panel.shape}")
        rap.valeur('shape_panel', list(panel.shape))
        rap.valeur('n_caracteristiques', len(config.CARACTERISTIQUES_RETENUES))
        rap.valeur('n_macro_predicteurs', len(config.MACRO_PREDICTEURS))

        # Ecarte les lignes sans cible calculable (horizon long uniquement ; sans effet a
        # 1 mois). Doit avoir lieu AVANT le fenetrage, sinon les R2 et le decompte des
        # mois disponibles seraient fausses.
        panel = horizon_mod.preparer_panel(panel, horizon, rap)
        print(f"Periode couverte : {panel['annee_mois'].min()} a {panel['annee_mois'].max()}")
        rap.valeur('periode_panel',
                   [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])
        rap.valeur('cible', config.CIBLE)
        rap.valeur('horizon_mois', horizon)
        rap.valeur('mois_embargo', fenetres.MOIS_EMBARGO_PAR_DEFAUT)

        panel = fenetres.restreindre_debut_entrainement(panel, config.ANNEE_DEBUT_ENTRAINEMENT)
        rap.valeur('annee_debut_entrainement', config.ANNEE_DEBUT_ENTRAINEMENT)
        rap.valeur('shape_panel_entrainement', list(panel.shape))
        rap.valeur('periode_entrainement',
                   [str(panel['annee_mois'].min()), str(panel['annee_mois'].max())])

        sortie = entrainer(spec, panel, rap)
        sauvegarder(spec, sortie, rap, horizon)
        rap.sauvegarder()   # l'entrainement (le plus long) n'est pas perdu si la suite plante

        if horizon == 1:
            spec.analyses(panel, sortie, rap, options)
            rap.sauvegarder()

    print("\n" + "=" * 70)
    print(f"TERMINE : {titre}")
    print(f"  -> rapport '{nom_rapport}' ecrit dans outputs/rapports/")
    print(f"  -> ouvre {_notebook(spec, horizon)} pour visualiser les resultats")
    print("=" * 70)
    return sortie


def _notebook(spec, horizon):
    """Quel notebook affiche les resultats de ce modele a cet horizon."""
    if horizon != 1:
        return "notebooks/11_horizon_12_mois.ipynb"
    return {
        'regression_lineaire': "notebooks/04_modele_lineaire.ipynb",
        'elastic_net': "notebooks/05_modele_elastic_net.ipynb",
        'lightgbm': "notebooks/06_modele_lightgbm.ipynb",
        'random_forest': "notebooks/07_modele_random_forest.ipynb",
    }[spec.cle]


# ============================================================
# 4. Point d'entree en ligne de commande
# ============================================================

DRAPEAUX = {
    '--sans-shap': 'sans_shap',                    # LightGBM, Random Forest
    '--sans-fama-macbeth': 'sans_fama_macbeth',    # Regression lineaire
}


def lancer(spec, horizon=1, argv=None):
    """Ce qu'appellent les 8 fichiers de scripts/entrainer/ : lit les drapeaux de la ligne
    de commande, puis execute.

        python scripts/entrainer/lightgbm_h1.py --sans-shap
    """
    argv = sys.argv[1:] if argv is None else argv
    inconnus = [a for a in argv if a not in DRAPEAUX]
    if inconnus:
        raise SystemExit(
            f"Option(s) inconnue(s) : {' '.join(inconnus)}\n"
            f"Options acceptees : {' '.join(DRAPEAUX)}"
        )
    options = {nom: (drapeau in argv) for drapeau, nom in DRAPEAUX.items()}
    return executer(spec, horizon=horizon, options=options)


def lancer_par_cle(cle, horizon=1, argv=None):
    """Meme chose, a partir de la cle textuelle du modele ('lightgbm', ...). Pratique pour
    un script maison ou une boucle shell."""
    return lancer(specs_module.PAR_CLE[cle], horizon=horizon, argv=argv)
