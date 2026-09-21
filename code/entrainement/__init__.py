"""
Entrainement des modeles : une boucle unique, quatre specifications.

    from entrainement import boucle, specs
    boucle.lancer(specs.LIGHTGBM, horizon=12)

Trois modules :
  - `specs.py`    : ce qui distingue chaque modele (grille, estimateur, diagnostics)
  - `boucle.py`   : le protocole commun (fenetres, grille, R2_oos pooled, sauvegardes)
  - `analyses.py` : les analyses annexes (Fama-MacBeth, stabilite, importances, SHAP)

Ajouter un 5e modele = une classe dans specs.py, une entree dans son registre, et deux
fichiers de lancement de 6 lignes dans scripts/entrainer/. Rien d'autre.
"""

from . import analyses, boucle, specs

__all__ = ['analyses', 'boucle', 'specs']
