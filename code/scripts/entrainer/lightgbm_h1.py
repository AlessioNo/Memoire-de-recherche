"""
LightGBM -- horizon 1 mois.

Lancement, depuis la RACINE du projet :

    python scripts/entrainer/lightgbm_h1.py --sans-shap

⚠️ Pre-requis : `python scripts/construction_panel.py` deja lance.

Ce fichier ne contient AUCUN calcul : tout le protocole (fenetres glissantes, recherche
d'hyperparametres sur la validation de chaque fenetre, R2_oos pooled, sauvegardes, journal
des experiences) vit dans `entrainement/boucle.py`, ecrit une seule fois pour les quatre
modeles et les deux horizons. Ce que ce modele-ci a de particulier est declare dans
`entrainement/specs.py`, classe `LightGBM`.

Volontairement lancable SEUL : les huit fichiers de ce dossier sont independants les uns
des autres, aucun n'a besoin qu'un autre ait tourne.

Fichiers produits (chemins engendres par chemins.py) :
  - modeles/lightgbm.joblib                   (modele de la DERNIERE fenetre)
  - outputs/predictions_lightgbm.parquet
  - outputs/resultats_lightgbm.parquet
  - outputs/resultats_lightgbm_par_fenetre.parquet
  - outputs/importance_lightgbm.parquet
  - 1 ligne dans outputs/journal_experiences.parquet  (sans doublon)
  - le rapport '06_lightgbm' (outputs/rapports/), lu par notebooks/06_modele_lightgbm.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entrainement import boucle, specs

if __name__ == "__main__":
    boucle.lancer(specs.LIGHTGBM, horizon=1)
