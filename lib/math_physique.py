import numpy as np
from lib.constantes import *

def deriver(f : np.ndarray, x : np.ndarray, k : int = 1) -> np.ndarray :
    """"
    Revoie les points de la dérivée d'ordre n d'une fonction quelconque par récurence.
    Attention la dimension du tableau final est de dimension (n-k) avec n le nombre de points d'entrés

    f : np.ndarray
        Tableau des ordonnées de la fonction d'origine
    x : np.ndarray
        Tableau des abscices de la fonction d'origne
    k : int
        Ordre de la dérivée, 1 par défaut
    """
    if (k <= 0) : 
        return f # si k == 0, on renvoie la fonction puisque

    tangente = []
    for i in range(1,len(x)) :
        if x[i] == x[i-1] :
            continue
        rapport = (f[i] - f[i-1]) / (x[i] - x[i-1])
        tangente.append(rapport)
    tangente_np = np.array(tangente)
    if (k == 1) : 
        return tangente_np
    else : 
        return deriver(tangente_np, x[1:],k-1)

def erreur(f_num: float, f_th: float, precision: int = 5) -> float:
    """
    Retourne le pourcentage d'erreur de la version numérique par rapport à la théorie.
    S'adapte automatiquement aux échelles microscopiques (femtosecondes).
    """
    norme = abs(f_num - f_th)
    
    # Sécurité anti-division par zéro si la théorie est nulle
    if abs(f_th) == 0.0:
        return 100.0 if f_num != 0.0 else 0.0
        
    # Calcul direct de l'erreur relative
    erreur_relative = (norme / abs(f_th)) * 100
    
    if erreur_relative < 1e-5:
        return 0.0
        
    return float(np.round(erreur_relative, precision))

def verifier_normalisation(x: np.ndarray, f: np.ndarray, tol: float = 1e-2) -> bool:
    """
    Vérifie que ∫|ψ|² dx ≈ 1 à chaque instant, avec une tolérance numérique.
    
    Parameters
    ----------
    x   : np.ndarray - grille spatiale
    f   : np.ndarray - matrice (nt, nx) de la fonction d'onde
    tol : float      - tolérance acceptable (défaut 1%)
    """
    normes = np.array([np.trapz(np.abs(f[j])**2, x) for j in range(len(f))])

    norme_ini   = normes[0]
    norme_fin   = normes[-1]
    derive_max  = np.max(np.abs(normes - 1.0))

    print(f"Norme initiale : {norme_ini:.6g}")
    print(f"Norme finale   : {norme_fin:.6g}")
    print(f"Dérive max     : {derive_max:.2g}  (tolérance = {tol:.0g})")

    if derive_max > tol:
        print("Norme non conservée — schéma instable ou masque trop agressif")
        return False

    print("Fonction normalisée")
    return True

def calculer_RT(x : np.array, f : np.array, R = True, T = False) -> float : 
    """Retourne les vrais coefficients de réflexion et de transmission 
    en cherchant l'instant optimal avant l'absorption des bords.
    """
    if (not R and not T) : 
        print("La fonction doit prendre au moins un paramètre en True")
        return -1.0, -1.0
    
    if (len(f) <= 0):
        print("La fonction d'onde ne peut pas être vide")
        return -1.0, -1.0

    densite = np.abs(f)**2
    indices_negatifs = np.where(x <= 0)[0]
    zone_negative = x[indices_negatifs]
    
    # 1. On cherche l'instant de départ pour savoir à quoi ressemble le paquet pur
    # Au tout début, l'onde est à 100% à gauche.
    norme_initiale_gauche = np.trapz(densite[0, indices_negatifs], zone_negative)

    # 2. On cherche le moment où le paquet a fini de taper la barrière.
    # C'est l'instant où la quantité d'onde à GAUCHE atteint son minimum local 
    # (juste après le choc) avant que le paquet réfléchi ne commence à entrer dans le masque.
    nt = f.shape[0]
    
    # On calcule l'intégrale à gauche pour chaque frame
    integrales_gauche = [np.trapz(densite[j, indices_negatifs], zone_negative) for j in range(nt)]
    
    # Le vrai coefficient de transmission T est la quantité d'onde qui a RÉUSSI à passer à droite
    # juste après le choc. On le mesure par complémentarité au point le plus bas du paquet à gauche.
    quantite_gauche_minimale = np.min(integrales_gauche[int(nt*0.2):int(nt*0.8)]) # On check le milieu de la simu
    
    transmission = 1.0 - (quantite_gauche_minimale / norme_initiale_gauche)
    
    # Sécurité physique : T et R doivent être bridés entre 0 et 1
    transmission = max(0.0, min(1.0, transmission))
    reflexion = 1.0 - transmission

    if R and T :
        return reflexion, transmission
    if R:
        return reflexion
    elif T:
        return transmission
    
    return 0.0, 0.0

def calculer_RT2(x: np.array, f: np.array, R=True, T=False):
    """
    Calcule les coefficients de réflexion et de transmission par intégration
    de la densité de probabilité |psi|² à la dernière frame.
    """

    if not R and not T:
        print("La fonction doit prendre au moins un paramètre en True")
        return -1.0, -1.0

    if f is None or len(f) <= 0:
        print("La fonction d'onde ne peut pas être vide")
        return -1.0, -1.0

    densite = np.abs(f[-1, :])**2

    masque_R = x < 0
    masque_barriere = (x >= 0) & (x <= a_barriere)
    masque_T = x > a_barriere

    reflexion = np.trapz(densite[masque_R], x[masque_R])
    transmission = np.trapz(densite[masque_T], x[masque_T])
    proba_barriere = np.trapz(densite[masque_barriere], x[masque_barriere])
    proba_totale = np.trapz(densite, x)

    print(f"Probabilité totale restante = {proba_totale * 100:.2f}%")
    print(f"Probabilité dans la barrière = {proba_barriere * 100:.2f}%")

    if R and T:
        return reflexion, transmission
    elif R:
        return reflexion
    elif T:
        return transmission
    
# -------------------------------------------------------------------------
# SUIVI DU PIC DU PAQUET D'ONDE
# -------------------------------------------------------------------------
# Objectif :
# - récupérer la position du maximum de densité |psi(x,t)|² à chaque frame
# - calculer la vitesse numérique entre deux frames
# - comparer cette vitesse à la vitesse de groupe théorique :
#       vg = H_BARRE * k_0 / M
# -------------------------------------------------------------------------

def suivre_pic(x, t, f, V=None, zone="tout", seuil_relatif=0.01):
    """
    Suit la position du pic du paquet d'onde à chaque instant.

    Paramètres
    ----------
    x : np.array
        Tableau des positions spatiales.

    t : np.array
        Tableau des temps.

    f : np.array complex
        Fonction d'onde simulée.
        Sa taille est normalement (nt, nx).
        Chaque ligne f[j, :] correspond à une frame temporelle.

    V : np.array ou None
        Potentiel V(x).
        Il sert à détecter la position de la barrière si on veut suivre
        uniquement la partie gauche ou droite de l'espace.

    zone : str
        Zone dans laquelle on cherche le pic.
        - "tout"   : cherche le pic sur tout l'espace.
        - "gauche" : cherche le pic avant la barrière.
        - "droite" : cherche le pic après la barrière.

    seuil_relatif : float
        Seuil de détection du pic.
        Exemple : 0.01 signifie qu'on ignore les pics dont la hauteur est
        inférieure à 1% du maximum global de la frame.
        Cela évite de suivre une toute petite queue numérique de l'onde.

    Retour
    ------
    positions_pic : np.array
        Tableau contenant la position x du pic pour chaque frame.
        Si le pic n'est pas fiable pour une frame, on met np.nan.
    """

    # On crée un tableau rempli de np.nan.
    # np.nan signifie : "pas de pic fiable détecté pour cette frame".
    positions_pic = np.full(len(t), np.nan)

    # ---------------------------------------------------------------------
    # Choix de la zone d'analyse
    # ---------------------------------------------------------------------
    # Si zone = "tout", on cherche le pic sur tout l'axe x.
    if zone == "tout" or V is None:
        masque_zone = np.ones_like(x, dtype=bool)

    # Si zone = "droite", on cherche seulement après la barrière.
    elif zone == "droite":
        indices_barriere = np.where(V > 0)[0]

        # Si aucune barrière n'est détectée, on considère par défaut
        # que la séparation gauche/droite est autour de x = 0.
        if len(indices_barriere) == 0:
            x_fin_barriere = 0.0
        else:
            # Dernier point où V > 0 : fin de la barrière.
            x_fin_barriere = x[indices_barriere[-1]]

        # On garde seulement les points situés après la barrière.
        masque_zone = x > x_fin_barriere

    # Si zone = "gauche", on cherche seulement avant la barrière.
    elif zone == "gauche":
        indices_barriere = np.where(V > 0)[0]

        if len(indices_barriere) == 0:
            x_debut_barriere = 0.0
        else:
            # Premier point où V > 0 : début de la barrière.
            x_debut_barriere = x[indices_barriere[0]]

        # On garde seulement les points situés avant la barrière.
        masque_zone = x < x_debut_barriere

    else:
        raise ValueError("zone doit être 'tout', 'gauche' ou 'droite'")

    # Axe x restreint à la zone choisie.
    x_zone = x[masque_zone]

    # ---------------------------------------------------------------------
    # Parcours de toutes les frames temporelles
    # ---------------------------------------------------------------------
    for j in range(len(t)):

        # Densité de probabilité :
        # rho(x,t) = |psi(x,t)|²
        densite = np.abs(f[j, :])**2

        # On restreint la densité à la zone choisie.
        densite_zone = densite[masque_zone]

        # Sécurité : si la zone est vide, on passe à la frame suivante.
        if len(densite_zone) == 0:
            continue

        # Maximum global sur toute la frame.
        # Il sert de référence pour le seuil relatif.
        max_global = np.max(densite)

        # Si le maximum dans la zone choisie est trop faible,
        # on considère qu'il ne s'agit pas d'un vrai paquet détectable.
        if np.max(densite_zone) < seuil_relatif * max_global:
            continue

        # np.argmax donne l'indice du maximum de densité dans la zone.
        indice_pic = np.argmax(densite_zone)

        # On convertit cet indice en position réelle x.
        positions_pic[j] = x_zone[indice_pic]

    return positions_pic


def analyser_vitesse_pic(x, t, f, V, zone="tout", afficher_graphes=True):
    """
    Analyse la vitesse du pic du paquet d'onde.

    Cette fonction :
    - récupère la position du pic à chaque frame ;
    - ajuste une droite sur la partie linéaire ;
    - compare la pente de cette droite à la vitesse théorique.
    """
    import matplotlib.pyplot as plt

    # Vitesse de groupe théorique
    vg = (H_BARRE * k_0) / M

    # Récupération de la position du pic à chaque frame
    positions_pic = suivre_pic(
        x=x,
        t=t,
        f=f,
        V=V,
        zone=zone,
        seuil_relatif=0.01,
    )

    # ---------------------------------------------------------------------
    # Calcul de la vitesse par régression linéaire
    # ---------------------------------------------------------------------
    # On évite np.diff(positions_pic) / dt car cela crée des pics artificiels.
    # La bonne méthode ici est de mesurer la pente moyenne de x_pic(t).
    # ---------------------------------------------------------------------

    if zone == "droite":
        # Pour la zone droite, on ne fait le fit que sur la deuxième moitié 
        # de la simulation, une fois que l'onde tunnel a franchi la barrière.
        t_min_fit = t[-1] * 0.5  
    else:
        # Pour la zone "tout" ou "gauche", on peut prendre tout le début
        t_min_fit = 0.0

    masque_fit = (
        ~np.isnan(positions_pic)
        & (t >= t_min_fit)
    )

    print("\nAnalyse de la vitesse du pic")
    print("-" * 40)
    print(f"Zone analysée : {zone}")
    print(f"Vitesse théorique vg : {vg:.6g} m/s")
    print(f"Régression linéaire faite pour t >= {t_min_fit} s")

    if np.sum(masque_fit) >= 2:
        coefficients = np.polyfit(t[masque_fit], positions_pic[masque_fit], 1)

        vitesse_fit = coefficients[0]
        position_initiale_fit = coefficients[1]

        erreur_fit = abs(vitesse_fit - vg) / abs(vg) * 100

        print(f"Vitesse numérique par régression : {vitesse_fit:.6g} m/s")
        print(f"Erreur relative : {erreur_fit:.4g} %")

    else:
        print("Pas assez de points pour faire une régression linéaire.")
        return positions_pic, np.array([]), vg, None

    # ---------------------------------------------------------------------
    # Affichage du graphe position + droite ajustée
    # ---------------------------------------------------------------------
    if afficher_graphes:

        plt.figure(figsize=(8, 5))

        plt.plot(
            t,
            positions_pic,
            label="Position du pic numérique",
        )

        plt.plot(
            t[masque_fit],
            vitesse_fit * t[masque_fit] + position_initiale_fit,
            "--",
            label="Ajustement linéaire",
        )

        plt.xlabel("Temps t [s]")
        plt.ylabel("Position du pic x [m]")
        plt.title(f"Mesure de la vitesse par pente moyenne -- a = {vitesse_fit:.3g} ms^-1")
        plt.grid(True)
        plt.legend()
        plt.show()

    return positions_pic, vitesse_fit, vg, position_initiale_fit

def detecter_temps_passage(x: np.ndarray, t: np.ndarray, f: np.ndarray, 
                             x_ligne_arrivee: float, seuil_relatif: float = 0.05) -> float:
    """
    Détecte l'instant où le pic du paquet d'onde franchit une ligne d'arrivée fixe.
    
    Méthode : on suit la position du maximum de |psi|^2 à chaque frame, et on
    retient le premier instant où ce maximum dépasse x_ligne_arrivee. On ignore
    les frames où le signal est trop faible pour être fiable (queue numérique).
    
    Parameters
    ----------
    x : np.ndarray
        Grille spatiale [m].
    t : np.ndarray
        Grille temporelle [s].
    f : np.ndarray
        Matrice (nt, nx) de la fonction d'onde complexe.
    x_ligne_arrivee : float
        Position de la ligne à franchir [m] (ex: fin de la barrière).
    seuil_relatif : float
        Fraction du maximum global de TOUTE la simulation, en dessous de
        laquelle on ignore la frame (évite de détecter une fraction négligeable
        de probabilité due au bruit numérique ou à la queue de la gaussienne).
    
    Returns
    -------
    float ou None
        Instant t_passage en secondes, ou None si jamais atteint.
    """
    R, T = calculer_RT(x, f, R=True, T=True)

    densite = np.abs(f)**2
    nt, nx = f.shape

    max_initial = np.max(densite[0, :])

    if T <= R:
        # --- CAS TUNNEL / RÉFLEXION DOMINANTE ---
        # Si seulement 30% passe, on réduit le seuil de détection proportionnellement à T !
        # On s'assure ainsi que le capteur cherche 5% *de la partie transmise*.
        seuil = seuil_relatif * max_initial * T
        
        # Sécurité : distance n'est pas définie dans tes arguments, 
        # utilise directement x_ligne_arrivee ou une valeur fixe cohérente (ex: 1e-9)
        masque_droite = x > (x_ligne_arrivee - 1e-9) 
        
        for j in range(nt):
            densite_droite = densite[j, masque_droite]
            x_droite = x[masque_droite]
            
            if len(densite_droite) == 0:
                continue
                
            max_local_droite = np.max(densite_droite)
            
            if max_local_droite >= seuil:
                indice_pic = np.argmax(densite_droite)
                pos_pic = x_droite[indice_pic]
                
                if pos_pic >= x_ligne_arrivee:
                    return t[j]
    else:
        # --- CAS SUR-BARRIÈRE / TRANSMISSION DOMINANTE ---
        seuil = seuil_relatif * max_initial
        for j in range(nt):
            max_instantane = np.max(densite[j, :])
            
            if max_instantane >= seuil:
                indice_pic = np.argmax(densite[j, :])
                pos_pic = x[indice_pic]
                
                if pos_pic >= x_ligne_arrivee:
                    return t[j]

    return None