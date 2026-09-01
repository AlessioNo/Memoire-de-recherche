"""
Genere la figure du protocole d'evaluation walk-forward (chapitre Cadre methodologique).

La figure est construite a partir de fenetres.generer_fenetres(), donc a partir des
valeurs REELLES de config.py : elle ne peut pas se desynchroniser du protocole
effectivement utilise par les etapes 04, 05 et 06.

Lancement, depuis la RACINE du projet :

    python scripts/figure_fenetres.py

Fichier produit : outputs/figure_protocole_fenetres.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import config
import fenetres

# Palette lisible aussi en impression noir et blanc
COULEURS = {
    "train": "#B8C4D9",
    "validation": "#6E8CB5",
    "test": "#2F4B6E",
}
TEXTE_SUR_BARRE = {"train": "#1A1A1A", "validation": "white", "test": "white"}
ETIQUETTES = {"train": "Entraînement", "validation": "Validation", "test": "Test"}


def mois_disponibles():
    """Liste des annee_mois reellement utilisables pour l'entrainement.

    Applique la MEME restriction que les scripts 04, 05 et 06 :
    fenetres.restreindre_debut_entrainement() avec config.ANNEE_DEBUT_ENTRAINEMENT.
    Sans cet appel, la figure partirait du debut du panel (ANNEE_DEBUT, etape 02) et
    ne correspondrait plus aux fenetres reellement estimees.
    """
    chemin = config.FICHIER_PANEL_MODELISATION
    if Path(chemin).exists():
        mois = pd.read_parquet(chemin, columns=["annee_mois"])
    else:
        print(f"Panel introuvable ({chemin}), repli sur une grille de mois synthetique.")
        mois = pd.DataFrame({"annee_mois": [
            f"{a}{m:02d}" for a in range(config.ANNEE_DEBUT, 2022) for m in range(1, 13)]})

    mois["annee_mois"] = mois["annee_mois"].astype(str)
    mois = fenetres.restreindre_debut_entrainement(mois, config.ANNEE_DEBUT_ENTRAINEMENT)
    return mois["annee_mois"].unique()


def annee_decimale(annee_mois):
    """'199801' -> 1998.0, pour positionner les barres au mois pres."""
    return int(annee_mois[:4]) + (int(annee_mois[4:]) - 1) / 12


def tracer(liste_fenetres, chemin_sortie):
    hauteur = 1.6 + 0.2 * len(liste_fenetres)
    fig, ax = plt.subplots(figsize=(9.5, hauteur))

    for rang, f in enumerate(liste_fenetres):
        y = len(liste_fenetres) - 1 - rang
        for bloc in ("train", "validation", "test"):
            debut = annee_decimale(f[bloc][0])
            fin = annee_decimale(f[bloc][-1]) + 1 / 12
            ax.barh(y, fin - debut, left=debut, height=0.80,
                    color=COULEURS[bloc], edgecolor="white", linewidth=0.8, zorder=3)
            if rang == 0:
                ax.text((debut + fin) / 2, y, ETIQUETTES[bloc],
                        ha="center", va="center", fontsize=8.5,
                        color=TEXTE_SUR_BARRE[bloc], zorder=4)

    annees = sorted({int(m[:4]) for f in liste_fenetres for m in f["train"] + f["test"]})
    premiere, derniere = annees[0], annees[-1] + 1

    ax.set_yticks(range(len(liste_fenetres)))
    ax.set_yticklabels([f"Fenêtre {f['numero']}" for f in reversed(liste_fenetres)],
                       fontsize=9.5)
    ax.set_ylim(-0.6, len(liste_fenetres) - 0.4)

    pas = 5 if derniere - premiere > 25 else 2
    ticks = list(range(premiere, derniere + 1, pas))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
    ax.set_xlim(premiere - 0.6, derniere + 0.6)
    ax.set_xlabel("Année", fontsize=9.5)

    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="both", length=0)

    """legende = [Patch(facecolor=COULEURS[b], label=t) for b, t in [
        ("train", "Entraînement (estimation des paramètres)"),
        ("validation", "Validation (choix des hyperparamètres)"),
        ("test", "Test (utilisé une seule fois)"),
    ]]
    ax.legend(handles=legende, loc="upper center", bbox_to_anchor=(0.5, 0),
              ncol=1, frameon=False, fontsize=9, handlelength=1.4, handleheight=1.0)"""

    fig.tight_layout()
    fig.savefig(chemin_sortie, bbox_inches="tight")
    fig.savefig(str(chemin_sortie).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("Figure ecrite :", chemin_sortie)


def main():
    config.assurer_dossiers()
    liste_fenetres = fenetres.generer_fenetres(
        mois_disponibles(),
        type_fenetre=config.TYPE_FENETRE,
        annees_train_initial=config.ANNEES_TRAIN_INITIAL,
        annees_validation=config.ANNEES_VALIDATION,
        annees_test_par_fenetre=config.ANNEES_TEST_PAR_FENETRE,
        reduction_validation_par_fenetre=config.REDUCTION_VALIDATION_PAR_FENETRE,
        fenetre_debut_reduction_validation=config.FENETRE_DEBUT_REDUCTION_VALIDATION,
        annees_validation_minimum=config.ANNEES_VALIDATION_MINIMUM,
    )
    print(f"Mode : {config.TYPE_FENETRE} | {len(liste_fenetres)} fenetres")
    tracer(liste_fenetres, config.OUTPUTS_DIR / "figure_protocole_fenetres.pdf")


if __name__ == "__main__":
    main()
