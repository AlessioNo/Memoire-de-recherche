"""
Regression lineaire -- horizon 12 mois.

Lancement, depuis la RACINE du projet :

    python scripts/entrainer/lineaire_h12.py

⚠️ Pre-requis : `python scripts/construction_panel.py` deja lance.
   La cible composee sur 12 mois est calculee par cette meme etape 03
   (partie A, section A.6bis) : elle doit avoir tourne apres son ajout.

Ce fichier ne contient AUCUN calcul : tout le protocole (fenetres glissantes, recherche
d'hyperparametres sur la validation de chaque fenetre, R2_oos pooled, sauvegardes, journal
des experiences) vit dans `entrainement/boucle.py`, ecrit une seule fois pour les quatre
modeles et les deux horizons. Ce que ce modele-ci a de particulier est declare dans
`entrainement/specs.py`, classe `RegressionLineaire`.

Volontairement lancable SEUL : les huit fichiers de ce dossier sont independants les uns
des autres, aucun n'a besoin qu'un autre ait tourne.

Fichiers produits (chemins engendres par chemins.py) :
  - modeles/modele_regression_lineaire_h12.pkl (modele de la DERNIERE fenetre)
  - outputs/predictions_regression_lineaire_h12.parquet
  - outputs/resultats_regression_lineaire_h12.parquet
  - outputs/resultats_regression_lineaire_par_fenetre_h12.parquet
  - 1 ligne dans outputs/journal_experiences.parquet  (sans doublon)
  - le rapport '11_horizon_lineaire' (outputs/rapports/), lu par notebooks/11_horizon_12_mois.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entrainement import boucle, specs

if __name__ == "__main__":
    boucle.lancer(specs.LINEAIRE, horizon=12)
