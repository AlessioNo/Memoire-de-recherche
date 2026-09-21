"""
Random Forest -- horizon 1 mois.

Lancement, depuis la RACINE du projet :

    python scripts/entrainer/random_forest_h1.py --sans-shap

⚠️ Pre-requis : `python scripts/construction_panel.py` deja lance.

Ce fichier ne contient AUCUN calcul : tout le protocole (fenetres glissantes, recherche
d'hyperparametres sur la validation de chaque fenetre, R2_oos pooled, sauvegardes, journal
des experiences) vit dans `entrainement/boucle.py`, ecrit une seule fois pour les quatre
modeles et les deux horizons. Ce que ce modele-ci a de particulier est declare dans
`entrainement/specs.py`, classe `RandomForest`.

Volontairement lancable SEUL : les huit fichiers de ce dossier sont independants les uns
des autres, aucun n'a besoin qu'un autre ait tourne.

Fichiers produits (chemins engendres par chemins.py) :
  - modeles/random_forest.joblib              (modele de la DERNIERE fenetre)
  - outputs/predictions_random_forest.parquet
  - outputs/resultats_random_forest.parquet
  - outputs/resultats_random_forest_par_fenetre.parquet
  - outputs/importance_random_forest.parquet
  - 1 ligne dans outputs/journal_experiences.parquet  (sans doublon)
  - le rapport '07_random_forest' (outputs/rapports/), lu par notebooks/07_modele_random_forest.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entrainement import boucle, specs

if __name__ == "__main__":
    boucle.lancer(specs.RANDOM_FOREST, horizon=1)
