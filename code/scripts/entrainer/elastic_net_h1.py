"""
Elastic Net -- horizon 1 mois.

Lancement, depuis la RACINE du projet :

    python scripts/entrainer/elastic_net_h1.py

⚠️ Pre-requis : `python scripts/construction_panel.py` deja lance.

Ce fichier ne contient AUCUN calcul : tout le protocole (fenetres glissantes, recherche
d'hyperparametres sur la validation de chaque fenetre, R2_oos pooled, sauvegardes, journal
des experiences) vit dans `entrainement/boucle.py`, ecrit une seule fois pour les quatre
modeles et les deux horizons. Ce que ce modele-ci a de particulier est declare dans
`entrainement/specs.py`, classe `ElasticNetModele`.

Volontairement lancable SEUL : les huit fichiers de ce dossier sont independants les uns
des autres, aucun n'a besoin qu'un autre ait tourne.

Fichiers produits (chemins engendres par chemins.py) :
  - modeles/elastic_net.joblib                (modele de la DERNIERE fenetre)
  - outputs/predictions_elastic_net.parquet
  - outputs/resultats_elastic_net.parquet
  - outputs/resultats_elastic_net_par_fenetre.parquet
  - outputs/importance_elastic_net.parquet
  - 1 ligne dans outputs/journal_experiences.parquet  (sans doublon)
  - le rapport '05_elastic_net' (outputs/rapports/), lu par notebooks/05_modele_elastic_net.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entrainement import boucle, specs

if __name__ == "__main__":
    boucle.lancer(specs.ELASTIC_NET, horizon=1)
