"""
Rapports d'execution : le pont entre les SCRIPTS (scripts/etape02 a etape07, qui
CALCULENT) et les NOTEBOOKS (02 a 07, qui AFFICHENT).

Pourquoi ce fichier existe
--------------------------
Depuis la reorganisation du projet, les notebooks ne calculent plus rien : tout le
travail lourd (nettoyage, construction du panel, entrainement des 3 modeles) est fait
par les scripts de scripts/, lances a la main. Mais un notebook doit quand meme pouvoir
afficher les diagnostics de ce calcul (nombre de lignes retirees a chaque filtre, taux
de valeurs manquantes, R2_oos par fenetre, coefficients...) SANS refaire le calcul.

Les gros resultats du projet ont deja leur fichier attitre dans config.py
(panel_pret_modelisation.parquet, resultats_*.parquet, predictions_*.parquet...). Ce
module s'occupe de tout le RESTE : les petits tableaux et compteurs qui n'etaient
jusqu'ici qu'imprimes a l'ecran au fil des cellules, et qui seraient donc perdus si
personne ne les ecrivait sur disque.

Meme esprit que fenetres.py / journal.py : une seule version de cette plomberie, reutilisee
par tous les scripts et tous les notebooks, plutot que copiee-collee.

ℹ️ Ce module heberge aussi `nettoyer_pour_json` (ex-utils.py, supprime) : la conversion des
types numpy en types Python natifs, utilisee ici pour les valeurs de diagnostic et par
journal.py pour la cle de deduplication des experiences.

Ou ca va sur le disque
----------------------
    outputs/rapports/<nom>.json            -> les VALEURS (compteurs, listes, textes)
    outputs/rapports/<nom>/<cle>.parquet   -> les TABLEAUX (DataFrame / Series)

Le dossier est derive de config.OUTPUTS_DIR, jamais ecrit en dur (meme principe que le
reste du projet : config.py reste la source unique de verite pour les chemins).

Cote SCRIPT (ecriture)
----------------------
    import rapports

    rap = rapports.Rapport('02_nettoyage')
    rap.valeur('shape_depart', chars.shape)
    rap.table('taux_missing', taux_missing)
    ...
    rap.sauvegarder()

Cote NOTEBOOK (lecture)
-----------------------
    import rapports

    rap = rapports.charger('02_nettoyage')
    print(rap.valeur('shape_depart'))
    rap.table('taux_missing')

`charger` leve une erreur explicite (avec la commande exacte a lancer) si le script
correspondant n'a jamais tourne -- plutot qu'un `FileNotFoundError` illisible.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import chemins


# Dossier des rapports, derive de config.OUTPUTS_DIR (jamais ecrit en dur).
DOSSIER_RAPPORTS = chemins.RAPPORTS_DIR

# Quel script relancer quand un rapport manque -- sert au message d'erreur de `charger`.
SCRIPT_PAR_RAPPORT = {
    '02_nettoyage': 'scripts/nettoyage_donnees.py',
    '03_panel': 'scripts/construction_panel.py',
    '04_regression_lineaire': 'scripts/entrainer/lineaire_h1.py',
    '05_elastic_net': 'scripts/entrainer/elastic_net_h1.py',
    '06_lightgbm': 'scripts/entrainer/lightgbm_h1.py',
    '07_random_forest': 'scripts/entrainer/random_forest_h1.py',
    '10_taille': 'scripts/analyse_par_taille.py',
    '11_horizon_lineaire': 'scripts/entrainer/lineaire_h12.py',
    '12_horizon_elastic_net': 'scripts/entrainer/elastic_net_h12.py',
    '13_horizon_lightgbm': 'scripts/entrainer/lightgbm_h12.py',
    '14_horizon_random_forest': 'scripts/entrainer/random_forest_h12.py',
}


def nettoyer_pour_json(valeur):
    """Convertit recursivement les types numpy (array, float64, int64...) en types
    natifs Python, pour pouvoir serialiser en JSON. Utilise a chaque fois qu'on doit
    comparer ou sauvegarder des parametres issus de config.py :
    - journal.py  : cle de deduplication des experiences, colonne 'params_specifiques_json'
    - rapports.py : valeurs de diagnostic ecrites par les scripts de calcul, relues par les
      notebooks d'affichage (outputs/rapports/*.json)
    """
    if isinstance(valeur, dict):
        return {cle: nettoyer_pour_json(v) for cle, v in valeur.items()}
    if isinstance(valeur, (list, tuple, np.ndarray)):
        return [nettoyer_pour_json(v) for v in valeur]
    if isinstance(valeur, np.integer):
        return int(valeur)
    if isinstance(valeur, np.floating):
        return float(valeur)
    return valeur


# Sentinelle : permet de distinguer "aucun argument passe" (= lecture) de "argument
# passe qui vaut None/False/{}/0" (= ecriture) dans Rapport.valeur / Rapport.table.
_MANQUANT = object()


class Rapport:
    """Un rapport d'execution : des VALEURS (JSON) + des TABLEAUX (Parquet).

    Cree vide par un script, rempli au fil du calcul, puis `sauvegarder()` a la fin
    (ou plusieurs fois en cours de route : chaque appel reecrit le rapport complet,
    ce qui permet de ne pas tout perdre si un script long plante apres la partie A).
    """

    def __init__(self, nom):
        self.nom = nom
        self.valeurs = {}
        self.tables = {}

    # ---------- ecriture (scripts) / lecture (notebooks) ----------

    def valeur(self, cle, valeur=_MANQUANT):
        """ECRIT une valeur si `valeur` est fourni, la RELIT sinon.

        Valeurs acceptees : nombre, texte, booleen, liste, dict, tuple. Les types numpy
        sont convertis en types Python natifs (via nettoyer_pour_json) pour pouvoir
        etre serialises en JSON -- exactement comme le fait deja journal.py pour les
        parametres de config.py.

            rap.valeur('n_lignes', 12345)   # ecriture (script)
            rap.valeur('n_lignes')          # lecture  (notebook)
        """
        if valeur is _MANQUANT:
            if cle not in self.valeurs:
                raise KeyError(
                    f"La valeur '{cle}' n'existe pas dans le rapport '{self.nom}'. "
                    f"Valeurs disponibles : {sorted(self.valeurs)}"
                )
            return self.valeurs[cle]
        self.valeurs[cle] = nettoyer_pour_json(valeur)
        return valeur

    def table(self, cle, donnees=_MANQUANT):
        """ECRIT un tableau si `donnees` est fourni, le RELIT sinon.

        Une Series est convertie en DataFrame a une colonne (son index est conserve :
        c'est souvent le nom des caracteristiques, indispensable a l'affichage).
        """
        if donnees is _MANQUANT:
            if cle not in self.tables:
                raise KeyError(
                    f"Le tableau '{cle}' n'existe pas dans le rapport '{self.nom}'. "
                    f"Tableaux disponibles : {sorted(self.tables)}"
                )
            return self.tables[cle]
        if isinstance(donnees, pd.Series):
            donnees = donnees.to_frame(name=donnees.name if donnees.name else 'valeur')
        # Les noms de colonnes doivent etre du texte pour Parquet
        donnees = donnees.copy()
        donnees.columns = [str(c) for c in donnees.columns]
        self.tables[cle] = donnees
        return donnees

    def sauvegarder(self):
        """Ecrit le rapport complet sur disque (JSON + un Parquet par tableau)."""
        dossier_tables = DOSSIER_RAPPORTS / self.nom
        dossier_tables.mkdir(parents=True, exist_ok=True)

        for cle, df in self.tables.items():
            df.to_parquet(dossier_tables / f"{cle}.parquet")

        contenu = {
            'nom': self.nom,
            'horodatage': pd.Timestamp.now().isoformat(timespec='seconds'),
            'valeurs': self.valeurs,
            'tables': sorted(self.tables),
        }
        with open(DOSSIER_RAPPORTS / f"{self.nom}.json", 'w', encoding='utf-8') as f:
            json.dump(contenu, f, indent=2, ensure_ascii=False, default=str)

        print(f"[rapport] {self.nom} sauvegarde : {len(self.valeurs)} valeurs, "
              f"{len(self.tables)} tableaux -> {DOSSIER_RAPPORTS / self.nom}.json")

    # ---------- confort ----------

    @property
    def horodatage(self):
        return self.valeurs.get('_horodatage', '(inconnu)')

    def resume(self):
        """Une ligne pour l'en-tete d'un notebook : quand ce rapport a ete produit."""
        return f"Rapport '{self.nom}' produit le {self.horodatage}"


def charger(nom):
    """Charge un rapport deja produit par un script, pour l'afficher dans un notebook.

    Leve une erreur explicite (avec la commande a lancer) si le script correspondant
    n'a jamais tourne -- c'est le cas le plus frequent quand un notebook ne s'ouvre pas
    comme prevu apres un clone du depot ou un changement de parametre dans config.py.
    """
    chemin_json = DOSSIER_RAPPORTS / f"{nom}.json"
    if not chemin_json.exists():
        script = SCRIPT_PAR_RAPPORT.get(nom, f"scripts/etape{nom[:2]}_*.py")
        raise FileNotFoundError(
            f"Aucun rapport '{nom}' sur le disque ({chemin_json}).\n"
            f"-> Lance d'abord le script de calcul correspondant, depuis la RACINE du projet :\n"
            f"     python {script}\n"
            f"   puis re-execute ce notebook (qui ne fait qu'afficher ses resultats)."
        )

    with open(chemin_json, encoding='utf-8') as f:
        contenu = json.load(f)

    rap = Rapport(nom)
    rap.valeurs = contenu.get('valeurs', {})
    rap.valeurs['_horodatage'] = contenu.get('horodatage', '(inconnu)')

    dossier_tables = DOSSIER_RAPPORTS / nom
    for cle in contenu.get('tables', []):
        chemin = dossier_tables / f"{cle}.parquet"
        if chemin.exists():
            rap.tables[cle] = pd.read_parquet(chemin)

    return rap


def existe(nom):
    """True si le rapport `nom` est deja sur le disque (script deja lance au moins une fois)."""
    return (DOSSIER_RAPPORTS / f"{nom}.json").exists()


def assurer_dossier():
    """Cree outputs/rapports/ si absent -- appele par chaque script au demarrage."""
    Path(DOSSIER_RAPPORTS).mkdir(parents=True, exist_ok=True)
