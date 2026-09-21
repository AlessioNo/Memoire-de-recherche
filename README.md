# Apprentissage automatique et prévision des rendements boursiers

**Mémoire de recherche — Master 2 Finance de Marché et Gestion des Risques**
Université Paris 1 Panthéon-Sorbonne — École de Management de la Sorbonne
Année académique 2025/2026

**Auteur :** Alessio Gino Nocera
**Directeurs :** Constantin Mellios et Mohamed El Fakir

📄 **[Lire le mémoire (PDF, 68 pages)](Memoire_Alessio_Nocera.pdf)**

---

## Résumé

Ce mémoire évalue quatre modèles de prévision du rendement excédentaire mensuel des actions américaines : une régression linéaire, un Elastic Net, une forêt aléatoire et un modèle de gradient boosting. Les quatre sont entraînés sur le même jeu de données de 2 615 807 observations titre-mois, couvrant 27 556 entreprises entre 1980 et 2021.

L'évaluation repose sur douze fenêtres glissantes dont les blocs de test, chronologiquement disjoints, couvrent 144 mois de 2010 à 2021. La performance est mesurée de deux façons indépendantes : le R² hors échantillon et les portefeuilles triés par décile. Le fil conducteur du travail tient à ce que ces deux mesures ne produisent pas le même classement.

## Principaux résultats

- **Les quatre modèles obtiennent un R² de test positif**, de 0,27 % à 0,49 %, les deux modèles à arbres devançant les deux modèles linéaires.
- **Les portefeuilles long-short dégagent tous un rendement moyen significatif**, avec des ratios de Sharpe de 0,766 à 1,220 — mais le modèle le moins précis produit le meilleur portefeuille.
- **Le momentum et la croissance de l'actif** ressortent comme les signaux les plus robustes.
- **L'ajout de prédicteurs macroéconomiques agrégés dégrade les quatre modèles.**
- **Un horizon de prévision de douze mois** améliore le classement des titres sans se traduire par un meilleur portefeuille.
- **Le pouvoir prédictif décroît avec la capitalisation.** Les alphas bruts, de 9,37 % à 14,02 %, ne survivent aux coûts de mise en œuvre que pour la forêt aléatoire en pondération par capitalisation.

## Plan du mémoire

| | |
|---|---|
| 1 | Introduction |
| 2 | Revue de littérature |
| 3 | Données |
| 4 | Cadre méthodologique |
| 5 | Résultats |
| 6 | Discussion |
| 7 | Conclusion |
| A | Annexes |

## Contenu du dépôt

- `Memoire_Alessio_Nocera.pdf` — le mémoire complet
- `code/` — l'intégralité du code de réplication (pipeline de données, entraînement des quatre modèles, construction et évaluation des portefeuilles). Voir le [README technique](code/README.md) pour l'installation et l'ordre d'exécution.

## Mots-clés

Apprentissage automatique · asset pricing empirique · coupe transversale des rendements · prévision hors échantillon · R² hors échantillon · forêt aléatoire · gradient boosting · Elastic Net · portefeuilles triés par décile · ratio de Sharpe · coûts de transaction
