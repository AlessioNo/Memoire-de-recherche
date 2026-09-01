"""
Turnover, couts de transaction et cout d'emprunt des portefeuilles long-short.

Pourquoi ce fichier existe
--------------------------
Les parties B et C mesurent des rendements BRUTS : aucun spread, aucun impact de marche,
aucun loyer sur la jambe vendeuse. C'est la convention de la litterature academique, et elle
se defend tant qu'on la dit. Elle ne se defend plus quand le portefeuille affiche un alpha a
deux chiffres sur un long-short equipondere a rotation mensuelle : c'est la premiere question
que pose un jury, et « c'est negligeable » n'est une reponse acceptable que chiffres en main.

Ce module fabrique ces chiffres. Il ne remplace rien : les series brutes restent la reference
du projet, les series nettes s'affichent a cote.

Les deux couts, et pourquoi ils ne se confondent pas
-----------------------------------------------------
  COUT DE TRANSACTION -- se paie quand on BOUGE. Spread et impact de marche, proportionnels
    au montant echange. Tenir une position trois mois ne coute rien pendant ces trois mois.
    Base de calcul : le turnover.

  COUT D'EMPRUNT -- se paie quand on RESTE. Pour vendre a decouvert il faut emprunter le
    titre, et le preteur facture un loyer mensuel du tant que la position est ouverte, meme
    sans passer le moindre ordre. Base de calcul : le notionnel vendu.

Ce n'est pas un cout qu'on ajoute par prudence : c'est une correction d'hypothese. Ecrire le
long-short comme une difference de rendements excedentaires suppose implicitement que le
produit de la vente est integralement restitue et remunere au taux sans risque. En pratique
le courtier en retient une commission, et le cout d'emprunt est exactement cet ecart.

⚠️ Les deux ne se reduisent pas par les memes leviers, d'ou deux parametres distincts. La
zone tampon fait baisser le turnover en allongeant la duree de detention -- donc elle allonge
d'autant la duree de location de la jambe vendeuse. Un seul parametre melangerait deux effets
de signe oppose.

Reperes de litterature
-----------------------
  - DeMiguel, Garlappi & Uppal (2009) : la definition du turnover reprise ici.
  - Novy-Marx & Velikov (2016) : couts effectifs par taille, et la zone tampon (`buffering`).
  - Frazzini, Israel & Moskowitz (2018) : les couts reels d'un grand gerant sont nettement
    inferieurs aux estimations academiques, ce qui plaide pour une analyse de SENSIBILITE
    plutot que pour un chiffre unique.
  - D'Avolio (2002) : les titres chers a emprunter sont precisement ceux qui peuplent les
    deciles d'anomalie extremes.
  - Garleanu & Pedersen (2013) : l'optimisation explicite sous cout, volontairement NON
    implementee ici (voir la note en tete de `zone_tampon`).
"""

import numpy as np
import pandas as pd


# ============================================================
# Poids mois par mois
# ============================================================

def poids_par_mois(donnees, colonne_decile, decile_cible, colonne_mois='annee_mois',
                   colonne_titre='permno', colonne_poids=None):
    """Poids de chaque titre, mois par mois, dans une jambe du portefeuille.

    Renvoie un DataFrame (index = mois, colonnes = titres) dont chaque ligne somme a 1.

    `colonne_poids = None` donne l'equipondere. Un nom de colonne (typiquement
    `config.COLONNE_MVEL1_BRUT`) donne la ponderation par ce critere, renormalisee a 1.

    ⚠️ Le turnover se mesure sur les POIDS, donc sur les titres individuels. Les rendements
    de decile agreges des parties B et C ne suffisent pas : deux mois consecutifs peuvent
    afficher le meme rendement de decile avec un portefeuille entierement renouvele.

    ⚠️ Le format est large et creux (un titre par colonne, la plupart a zero). Sur douze ans
    et plusieurs milliers de permnos ca reste tenable en memoire, mais c'est la raison pour
    laquelle ce module travaille jambe par jambe plutot que sur les dix deciles a la fois.
    """
    colonnes = [colonne_mois, colonne_titre, colonne_decile]
    if colonne_poids is not None:
        colonnes.append(colonne_poids)

    jambe = donnees.loc[donnees[colonne_decile] == decile_cible, colonnes].copy()
    jambe[colonne_mois] = jambe[colonne_mois].astype(str)

    if colonne_poids is None:
        jambe['_poids'] = 1.0
    else:
        jambe = jambe.dropna(subset=[colonne_poids])
        jambe = jambe[jambe[colonne_poids] > 0]
        jambe['_poids'] = jambe[colonne_poids].astype(float)

    if jambe.empty:
        return pd.DataFrame()

    tableau = jambe.pivot_table(index=colonne_mois, columns=colonne_titre,
                                values='_poids', aggfunc='sum', fill_value=0.0)
    totaux = tableau.sum(axis=1)
    return tableau.div(totaux.where(totaux > 0), axis=0).fillna(0.0).sort_index()


def _derive(poids_formation, rendements_titres):
    """Poids effectifs en FIN de mois, apres derive due aux rendements du mois.

    w+ = w (1 + r) / somme_j w_j (1 + r_j)

    Cette derive est GRATUITE : elle se produit sans passer le moindre ordre. C'est tout
    l'interet de la formule -- un portefeuille qu'on ne touche pas voit ses poids bouger, et
    facturer cet ecart reviendrait a inventer des transactions qui n'ont pas eu lieu. C'est
    l'erreur la plus courante dans les calculs de turnover faits a la main.
    """
    croissance = 1.0 + rendements_titres.reindex_like(poids_formation).fillna(0.0)
    valeur = poids_formation * croissance
    total = valeur.sum(axis=1)
    return valeur.div(total.where(total > 0), axis=0).fillna(0.0)


def decomposer_turnover(poids, rendements_titres):
    """Turnover mensuel d'une jambe, decompose par origine.

        TO_t+1 = somme_i | w_i,t+1 - w+_i,t |

    Un titre qui reste au meme poids ne coute rien. Un titre qui entre coute son poids
    entier, un titre qui sort coute le poids qu'il avait. Une jambe integralement renouvelee
    donne 2.0 : un pour tout vendre, un pour tout racheter.

    Trois colonnes, dont les deux dernieres somment exactement a la premiere :
      'turnover'       -- le total
      'entrees_sorties'-- la part due aux titres qui ENTRENT ou SORTENT de la jambe. C'est
                          la rotation imputable au SIGNAL du modele : elle mesure a quel
                          point ses classements bougent d'un mois sur l'autre.
      'rebalancement'  -- la part due aux titres presents les DEUX mois, dont le poids doit
                          etre ramene a sa cible.

    ⚠️ Cette seconde part n'est pas nulle meme a composition rigoureusement constante, et
    c'est un point que beaucoup de calculs faits a la main ratent : un portefeuille
    equipondere derive mecaniquement (les titres qui montent pesent plus lourd) et doit etre
    ramene a l'egalite chaque mois. Sur un decile de plusieurs centaines de titres, ce
    plancher tourne autour de 5 % par mois. Il n'est imputable a aucun modele -- c'est le
    prix de l'equiponderation elle-meme -- et le distinguer evite d'attribuer aux modeles
    une rotation qu'ils ne causent pas.

    ⚠️ Convention BIDIRECTIONNELLE. Une partie de la litterature divise ce chiffre par deux
    et parle de turnover unidirectionnel. Les deux se defendent, mais melanger les deux dans
    un meme tableau produit des couts faux d'un facteur deux : la note de tableau doit dire
    laquelle est retenue.

    ⚠️ Le premier mois construit la jambe a partir de rien : son turnover vaut 1.0 par
    construction et ne dit rien sur la strategie. Il est renvoye tel quel ; c'est
    `couts_mensuels` qui decide de l'ecarter ou non.
    """
    if poids.empty:
        return pd.DataFrame(columns=['turnover', 'entrees_sorties', 'rebalancement'])

    poids = poids.sort_index()
    mois = list(poids.index)
    lignes = {}

    for position, mois_courant in enumerate(mois):
        cible = poids.loc[mois_courant]
        if position == 0:
            total = float(cible.abs().sum())
            lignes[mois_courant] = (total, total, 0.0)
            continue

        mois_precedent = mois[position - 1]
        # Le rendement du mois PRECEDENT fait deriver les poids formes a cette date-la
        # jusqu'a la reformation d'aujourd'hui.
        if mois_precedent in rendements_titres.index:
            rendements = rendements_titres.loc[mois_precedent]
        else:
            rendements = pd.Series(dtype=float)
        precedent = poids.loc[[mois_precedent]]
        derive = _derive(precedent,
                         pd.DataFrame([rendements], index=[mois_precedent])).iloc[0]

        union = cible.index.union(derive.index)
        avant = derive.reindex(union).fillna(0.0)
        apres = cible.reindex(union).fillna(0.0)
        ecart = (apres - avant).abs()

        detenu_avant = avant > 0
        detenu_apres = apres > 0
        commun = detenu_avant & detenu_apres

        lignes[mois_courant] = (float(ecart.sum()),
                                float(ecart[~commun].sum()),
                                float(ecart[commun].sum()))

    return pd.DataFrame.from_dict(
        lignes, orient='index',
        columns=['turnover', 'entrees_sorties', 'rebalancement']).sort_index()


def turnover_jambe(poids, rendements_titres):
    """Turnover mensuel total d'une jambe. Voir `decomposer_turnover` pour le detail."""
    detail = decomposer_turnover(poids, rendements_titres)
    return detail['turnover'] if not detail.empty else pd.Series(dtype=float)


def rendements_titres_par_mois(donnees, colonne_rendement, colonne_mois='annee_mois',
                               colonne_titre='permno'):
    """Table (mois x titre) des rendements, pour le calcul de derive.

    ⚠️ Utiliser le rendement TOTAL (`RET`), pas `excess_return`. La derive porte sur la
    valeur de marche des positions, qui croit au rendement total ; retrancher le taux sans
    risque introduirait un biais faible mais systematique, et sans raison.
    """
    table = donnees[[colonne_mois, colonne_titre, colonne_rendement]].copy()
    table[colonne_mois] = table[colonne_mois].astype(str)
    return table.pivot_table(index=colonne_mois, columns=colonne_titre,
                             values=colonne_rendement, aggfunc='mean')


# ============================================================
# Zone tampon : amortir le turnover sans changer le signal
# ============================================================

def zone_tampon(donnees, colonne_decile, decile_entree, decile_sortie,
                colonne_mois='annee_mois', colonne_titre='permno', sens='haut'):
    """Regle d'appartenance amortie : on entre au decile `decile_entree`, on ne sort qu'au
    dela de `decile_sortie`.

    Le probleme : un titre qui oscille autour de la frontiere du decile 10 est achete et
    revendu tous les mois, pour un gain de signal nul. La zone tampon elargit la regle de
    SORTIE sans toucher a la regle d'ENTREE -- on entre au decile 10, mais on ne vend que si
    le titre tombe sous le decile 9. Le signal du modele n'est pas modifie, seule la
    discipline de detention l'est. C'est le `buffering` de Novy-Marx & Velikov (2016) : forte
    reduction du turnover pour une perte d'alpha generalement faible.

    ⚠️ Ce que ce module ne fait PAS : optimiser le portefeuille sous contrainte de cout
    (region de non-trade a la Garleanu & Pedersen 2013). Ce n'est pas une question de
    difficulte technique mais d'objet d'etude -- on n'evaluerait plus « le spread de deciles
    produit par le modele » mais « un portefeuille optimise qui utilise le signal du
    modele ». L'optimiseur apporte une aversion au risque, une matrice de covariance sur
    plusieurs milliers de titres et ses propres choix d'estimation, qui affectent les modeles
    inegalement et brouillent precisement la comparaison qui est l'objet du memoire.

    `sens = 'haut'` pour la jambe acheteuse (deciles eleves), `'bas'` pour la vendeuse.

    Renvoie une Series booleenne alignee sur `donnees` : True = titre detenu ce mois-la.
    """
    if sens not in ('haut', 'bas'):
        raise ValueError(f"sens doit valoir 'haut' ou 'bas', pas {sens!r}")

    table = donnees[[colonne_mois, colonne_titre, colonne_decile]].copy()
    table[colonne_mois] = table[colonne_mois].astype(str)

    if sens == 'haut':
        entre = table[colonne_decile] >= decile_entree
        reste = table[colonne_decile] >= decile_sortie
    else:
        entre = table[colonne_decile] <= decile_entree
        reste = table[colonne_decile] <= decile_sortie

    table['_entre'] = entre.fillna(False).to_numpy()
    table['_reste'] = reste.fillna(False).to_numpy()

    detenu = pd.Series(False, index=table.index)
    portefeuille = set()

    for _, bloc in table.groupby(colonne_mois, sort=True, observed=True):
        titres_entrants = set(bloc.loc[bloc['_entre'], colonne_titre])
        titres_maintenus = set(bloc.loc[bloc['_reste'], colonne_titre])

        # Un titre absent du mois (radiation, donnee manquante) sort mecaniquement : il ne
        # peut pas etre maintenu s'il n'est plus cote.
        portefeuille = (portefeuille & titres_maintenus) | titres_entrants
        detenu.loc[bloc.index] = bloc[colonne_titre].isin(portefeuille).to_numpy()

    return detenu


def poids_depuis_appartenance(donnees, colonne_detenu, colonne_mois='annee_mois',
                              colonne_titre='permno', colonne_poids=None):
    """Meme sortie que `poids_par_mois`, mais a partir d'un masque booleen de detention.

    Sert aux jambes construites par zone tampon, dont l'appartenance ne se lit plus
    directement dans une colonne de decile.
    """
    colonnes = [colonne_mois, colonne_titre]
    if colonne_poids is not None:
        colonnes.append(colonne_poids)

    jambe = donnees.loc[donnees[colonne_detenu].astype(bool), colonnes].copy()
    jambe[colonne_mois] = jambe[colonne_mois].astype(str)

    if colonne_poids is None:
        jambe['_poids'] = 1.0
    else:
        jambe = jambe.dropna(subset=[colonne_poids])
        jambe = jambe[jambe[colonne_poids] > 0]
        jambe['_poids'] = jambe[colonne_poids].astype(float)

    if jambe.empty:
        return pd.DataFrame()

    tableau = jambe.pivot_table(index=colonne_mois, columns=colonne_titre,
                                values='_poids', aggfunc='sum', fill_value=0.0)
    totaux = tableau.sum(axis=1)
    return tableau.div(totaux.where(totaux > 0), axis=0).fillna(0.0).sort_index()


def rendement_depuis_poids(poids, rendements_titres):
    """Rendement mensuel d'une jambe a partir de ses poids de formation.

    Le poids est celui du mois t, le rendement celui du mois t : c'est bien le rendement
    obtenu en detenant pendant t le portefeuille forme a l'ouverture de t.
    """
    if poids.empty:
        return pd.Series(dtype=float)
    aligne = rendements_titres.reindex(index=poids.index, columns=poids.columns).fillna(0.0)
    return (poids * aligne).sum(axis=1)


# ============================================================
# Application des couts
# ============================================================

def couts_mensuels(turnover_long, turnover_court, cout_transaction_bps,
                   cout_emprunt_annuel_bps=0.0, exposition_courte=1.0,
                   ignorer_premier_mois=True, nb_periodes_par_an=12):
    """Cout total de chaque mois, decompose entre transaction et emprunt.

    Transaction : `(TO_long + TO_court) x c`, ou c est le cout UNIDIRECTIONNEL proportionnel.
    Emprunt     : taux annuel / 12, applique au notionnel vendu, chaque mois, qu'on ait
                  trade ou non.

    `ignorer_premier_mois` : le premier mois construit les deux jambes a partir de rien, donc
    un turnover mecanique de 2.0 qui ne dit rien de la strategie. La litterature l'ecarte
    generalement. Le mettre a False repond a une autre question -- celle du cout de mise en
    place -- qui est legitime mais distincte.

    Renvoie un DataFrame indexe par mois : turnover_total, cout_transaction, cout_emprunt,
    cout_total.
    """
    turnover = (turnover_long.reindex(turnover_long.index.union(turnover_court.index))
                .fillna(0.0)
                + turnover_court.reindex(turnover_long.index.union(turnover_court.index))
                .fillna(0.0)).sort_index()

    if ignorer_premier_mois and len(turnover) > 0:
        turnover = turnover.iloc[1:]

    cout_transaction = turnover * (cout_transaction_bps / 10_000.0)
    cout_emprunt = pd.Series(
        (cout_emprunt_annuel_bps / 10_000.0) / nb_periodes_par_an * exposition_courte,
        index=turnover.index)

    return pd.DataFrame({
        'turnover_total': turnover,
        'cout_transaction': cout_transaction,
        'cout_emprunt': cout_emprunt,
        'cout_total': cout_transaction + cout_emprunt,
    })


def rendement_net(rendement_brut, couts):
    """Rendement brut moins cout total, aligne sur les mois communs.

    ⚠️ Le cout est preleve sur le mois ou la transaction a lieu, c'est-a-dire le mois dont le
    portefeuille vient d'etre reforme. Le decaler d'un mois deplacerait la perte sans la
    supprimer, mais fausserait la correlation entre couts et rendements -- or les mois de
    forte rotation ne sont pas des mois quelconques.
    """
    brut = pd.Series(rendement_brut).dropna()
    brut.index = brut.index.astype(str)
    total = couts['cout_total'].reindex(brut.index).fillna(0.0)
    return (brut - total).rename('rendement_net')


# ============================================================
# Seuil de rentabilite
# ============================================================

def cout_seuil_rendement_nul(rendement_brut, turnover, cout_emprunt_annuel_bps=0.0,
                             exposition_courte=1.0, nb_periodes_par_an=12):
    """Cout unidirectionnel, en points de base, qui annule le rendement moyen.

        c* = (rendement mensuel moyen - cout d'emprunt mensuel) / turnover moyen

    C'est LE chiffre a mettre dans le memoire. Il ne depend d'aucune hypothese contestable
    sur le niveau des couts, et il retourne la charge de la preuve : si la strategie tient
    jusqu'a 60 points de base, personne ne peut serieusement pretendre qu'elle est un
    artefact de frais. Si elle casse a 8, c'est le resultat, et il s'ecrit tel quel.
    """
    brut = pd.Series(rendement_brut).dropna()
    brut.index = brut.index.astype(str)
    turnover = pd.Series(turnover).reindex(brut.index).dropna()
    brut = brut.reindex(turnover.index)

    if len(brut) < 2 or turnover.mean() <= 0:
        return np.nan

    emprunt_mensuel = (cout_emprunt_annuel_bps / 10_000.0) / nb_periodes_par_an * exposition_courte
    return float((brut.mean() - emprunt_mensuel) / turnover.mean() * 10_000.0)


def cout_seuil_significativite(rendement_brut, turnover, seuil_t=1.96,
                               cout_emprunt_annuel_bps=0.0, exposition_courte=1.0,
                               nb_periodes_par_an=12, cout_maximum_bps=500.0,
                               tolerance=0.01):
    """Cout unidirectionnel qui ramene la t-stat du rendement moyen a `seuil_t`.

    Retranche `c x TO_t` mois par mois puis recalcule la t-stat, et cherche le c qui atteint
    le seuil par dichotomie.

    ⚠️ Ce seuil est plus bas que celui du rendement nul, souvent nettement. Retrancher un
    cout proportionnel au turnover ne fait pas que baisser la moyenne : il ajoute de la
    variance, puisque le turnover varie d'un mois sur l'autre. Une strategie peut donc perdre
    sa significativite bien avant de perdre son rendement, et c'est ce chiffre-la qu'un jury
    exigeant regardera.

    ⚠️ t-stat SIMPLE ici, pas Newey-West : la dichotomie appelle la fonction des dizaines de
    fois, et le classement des modeles ne change pas. La t-stat Newey-West de l'alpha reste
    calculee par `facteurs.alpha_facteurs` sur la serie nette finale.
    """
    brut = pd.Series(rendement_brut).dropna()
    brut.index = brut.index.astype(str)
    turnover = pd.Series(turnover).reindex(brut.index).dropna()
    brut = brut.reindex(turnover.index)

    if len(brut) < 3:
        return np.nan

    emprunt_mensuel = (cout_emprunt_annuel_bps / 10_000.0) / nb_periodes_par_an * exposition_courte

    def t_stat(cout_bps):
        net = brut - turnover * (cout_bps / 10_000.0) - emprunt_mensuel
        ecart_type = net.std(ddof=1)
        if ecart_type <= 0:
            return np.nan
        return net.mean() / (ecart_type / np.sqrt(len(net)))

    if not np.isfinite(t_stat(0.0)) or t_stat(0.0) < seuil_t:
        return 0.0            # deja non significatif sans le moindre cout
    if t_stat(cout_maximum_bps) > seuil_t:
        return np.inf         # tient au-dela de la borne exploree

    bas, haut = 0.0, cout_maximum_bps
    while haut - bas > tolerance:
        milieu = (bas + haut) / 2
        if t_stat(milieu) > seuil_t:
            bas = milieu
        else:
            haut = milieu
    return float((bas + haut) / 2)


def sensibilite_aux_couts(rendement_brut, turnover, grille_bps,
                          cout_emprunt_annuel_bps=0.0, exposition_courte=1.0,
                          nb_periodes_par_an=12, calculer_metriques=None):
    """Rendement et Sharpe nets pour chaque niveau de cout de `grille_bps`.

    Sert a tracer la courbe de sensibilite -- la reponse honnete a « quel cout retenir ? »
    etant « voici la reponse pour tous les niveaux plausibles ».
    """
    brut = pd.Series(rendement_brut).dropna()
    brut.index = brut.index.astype(str)
    turnover = pd.Series(turnover).reindex(brut.index).fillna(0.0)
    emprunt_mensuel = (cout_emprunt_annuel_bps / 10_000.0) / nb_periodes_par_an * exposition_courte

    lignes = []
    for cout_bps in grille_bps:
        net = brut - turnover * (cout_bps / 10_000.0) - emprunt_mensuel
        ligne = {'cout_bps': float(cout_bps)}
        if calculer_metriques is not None:
            mesures = calculer_metriques(net)
            ligne.update({cle: mesures[cle] for cle in
                          ('rendement_annualise', 'volatilite_annualisee', 'sharpe_ratio',
                           't_stat')})
        else:
            ligne['rendement_annualise'] = net.mean() * nb_periodes_par_an
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# ============================================================
# Chaine complete : d'un jeu de predictions a une serie nette
# ============================================================

def evaluer_long_short(donnees, colonne_decile, rendements_titres, nb_deciles=10,
                       colonne_mois='annee_mois', colonne_titre='permno',
                       colonne_poids=None, tampon_deciles=0,
                       cout_transaction_bps=0.0, cout_emprunt_annuel_bps=0.0,
                       ignorer_premier_mois=True, nb_periodes_par_an=12):
    """Construit les deux jambes, mesure leur turnover, applique les couts.

    Renvoie un dict :
      'poids_long', 'poids_court'   -- tables mois x titre
      'rendement_brut'              -- Series mensuelle (decile haut moins decile bas)
      'couts'                       -- DataFrame decompose
      'rendement_net'               -- Series mensuelle
      'turnover'                    -- Series mensuelle, somme des deux jambes

    `tampon_deciles = 0` reproduit exactement la construction des parties B et C. Une valeur
    de 1 elargit la regle de sortie d'un decile de chaque cote.

    ⚠️ Le rendement brut est recalcule ici a partir des poids, et non repris des parties B et
    C. Sans zone tampon et en equipondere il doit coincider a la precision machine pres avec
    `rendements_portefeuilles` ; l'ecart eventuel vient des lignes ecartees faute de
    capitalisation, et c'est exactement ce qu'on veut voir. La partie D affiche cet ecart.
    """
    decile_haut, decile_bas = nb_deciles, 1

    if tampon_deciles and tampon_deciles > 0:
        donnees = donnees.copy()
        donnees['_detenu_long'] = zone_tampon(
            donnees, colonne_decile, decile_haut, decile_haut - tampon_deciles,
            colonne_mois, colonne_titre, sens='haut')
        donnees['_detenu_court'] = zone_tampon(
            donnees, colonne_decile, decile_bas, decile_bas + tampon_deciles,
            colonne_mois, colonne_titre, sens='bas')
        poids_long = poids_depuis_appartenance(donnees, '_detenu_long', colonne_mois,
                                               colonne_titre, colonne_poids)
        poids_court = poids_depuis_appartenance(donnees, '_detenu_court', colonne_mois,
                                                colonne_titre, colonne_poids)
    else:
        poids_long = poids_par_mois(donnees, colonne_decile, decile_haut, colonne_mois,
                                    colonne_titre, colonne_poids)
        poids_court = poids_par_mois(donnees, colonne_decile, decile_bas, colonne_mois,
                                     colonne_titre, colonne_poids)

    rendement_brut = (rendement_depuis_poids(poids_long, rendements_titres)
                      - rendement_depuis_poids(poids_court, rendements_titres))

    detail_long = decomposer_turnover(poids_long, rendements_titres)
    detail_court = decomposer_turnover(poids_court, rendements_titres)
    turnover_long = detail_long['turnover'] if not detail_long.empty else pd.Series(dtype=float)
    turnover_court = detail_court['turnover'] if not detail_court.empty else pd.Series(dtype=float)

    # Les deux jambes cumulees, par origine de la rotation.
    index_commun = turnover_long.index.union(turnover_court.index)
    decomposition = pd.DataFrame({
        colonne: (detail_long.reindex(index_commun)[colonne].fillna(0.0)
                  + detail_court.reindex(index_commun)[colonne].fillna(0.0))
        for colonne in ('turnover', 'entrees_sorties', 'rebalancement')
    }) if not (detail_long.empty and detail_court.empty) else pd.DataFrame()

    couts = couts_mensuels(turnover_long, turnover_court, cout_transaction_bps,
                           cout_emprunt_annuel_bps, exposition_courte=1.0,
                           ignorer_premier_mois=ignorer_premier_mois,
                           nb_periodes_par_an=nb_periodes_par_an)

    return {
        'poids_long': poids_long,
        'poids_court': poids_court,
        'rendement_brut': rendement_brut,
        'turnover': couts['turnover_total'],
        'turnover_long': turnover_long,
        'turnover_court': turnover_court,
        'decomposition_turnover': decomposition,
        'couts': couts,
        'rendement_net': rendement_net(rendement_brut, couts),
    }


def resumer_turnover(turnover, nb_periodes_par_an=12):
    """Turnover moyen, renouvellement annuel, duree de detention implicite.

    La duree de detention se lit comme l'inverse du taux de renouvellement d'une jambe : un
    turnover bidirectionnel de 1.0 par mois sur les deux jambes signifie qu'une jambe
    renouvelle la moitie de ses positions chaque mois, soit une detention moyenne de deux
    mois. C'est une lecture approximative -- elle suppose un renouvellement homogene -- mais
    elle parle bien plus a un lecteur qu'un chiffre de turnover brut.
    """
    serie = pd.Series(turnover).dropna()
    if serie.empty:
        return {'turnover_mensuel_moyen': np.nan, 'turnover_annualise': np.nan,
                'duree_detention_mois': np.nan, 'n_mois': 0}

    moyen = float(serie.mean())
    renouvellement_jambe = moyen / 4.0  # /2 pour les deux jambes, /2 pour l'aller-retour
    return {
        'turnover_mensuel_moyen': moyen,
        'turnover_annualise': moyen * nb_periodes_par_an,
        'duree_detention_mois': (1.0 / renouvellement_jambe
                                 if renouvellement_jambe > 0 else np.inf),
        'n_mois': int(len(serie)),
    }
