import numpy as np
import matplotlib.pyplot as plt
from lib.constantes import *
from lib.PaquetOndeGaussMI4_C import GaussWP, C , pi #, H_BARRE, M
from lib.math_physique import erreur, verifier_normalisation, calculer_RT, calculer_RT2
from lib.graphique import animation
"""
#Ce code permet de simuler l'équation de Schrödinger en 1D pour une onde gaussienne.

##Fonction : 
- : permet de calculer la dérivée d'une fonction numérique par récurence
- erreur : permet de calculer le pourcentage d'erreur entre une fonction numérique et une
"""

#--Fonctions---------------------------

"""
Définition de f'(a) = lim_h->0 (f(a+h) - f(a)) / h = df/dx(a) ~ (f(b) - f(a)) / (b-a)
"""


def Onde2d(x : np.array) -> np.array:
    """Retourne une matrice de nx lignes et nt colonnes représentant l'évolution d'une onde gaussienne dans l'espace à un t donné."""
    arr = np.empty((nt ,nx),dtype=complex)

    arr[0,:] = GaussWP(k_0, lg_initiale, x, t_0)
    return arr

def masque_absorbant(x, largeur=3.0):
    # Créer par Claude
    mask = np.ones_like(x)
    # Bord gauche
    zone_g = x < (x[0] + largeur)
    mask[zone_g] = np.exp(-((x[zone_g] - (x[0] + largeur)) / (largeur/3))**2)
    # Bord droit
    zone_d = x > (x[-1] - largeur)
    mask[zone_d] = np.exp(-((x[zone_d] - (x[-1] - largeur)) / (largeur/3))**2)
    return mask

def propagation_onde(x : np.array, t : np.array, V : np.array, x_0 : float = -4) -> np.array :
    """
    le déplacement d'un paquet d'onde
    paramètre : 
    -----------
    x : abscice [M]
    t : temps [T]
    V : potentiel [J] -- Tableau a 1d
    """
    f = Onde2d(x) #initialisation de la fonction d'onde
    dx = x[1] - x[0]
    dt = t[1] - t[0] # on peut le faire car on utilise linspace pour faire les ensembles

    # Ajouter un "masque absorbant" aux bords pour éviter les réflexions

    mask = masque_absorbant(x) # ajout de Claude pour masquer les erreurs de dérivée au niveau des bords
    # on simule l'équation avec l'eq de Sch
    for j in range(len(t)-1):
        d2f_dx2 = (f[j, 2:] - 2*f[j, 1:-1] + f[j, :-2]) / dx**2
        hamiltonien = (-H_BARRE**2 / (2*M)) * d2f_dx2 + V[1:-1] * f[j, 1:-1]
        f[j+1, 1:-1] = f[j, 1:-1] + dt * (-1j / H_BARRE) * hamiltonien
        f[j+1, :] *= mask

    return f

from scipy.linalg import solve_banded

def propagation_onde_CN(x : np.array, t : np.array, V : np.array, x_0 : float = -4) -> tuple:
    """Permet de calculer la propagation du paquet d'onde via la méthode de Crank-Nicolson
    
    Paramètres : 
    ------------
    - x : np.array
        Axe spatial [L]
    - t : np.array
        Tableau des instants temporels [T]
    - V : np.array
        Tableau du potentiel V(x) à 1D [J]
    - x_0 : float
        Position initiale du centre du paquet d'ondes [L]
        
    Retourne :
    ----------
    - f : np.array (matrice complexe de taille nt x nx)
    - t_passage : float (ou None)
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    r = 1j * H_BARRE * dt / (4 * M * dx**2)  # coefficient central

    f = np.zeros((nt, nx), dtype=complex)
    f[0, :] = GaussWP(k_0, lg_initiale, x, t[0], x_0)

    # Matrices tridiagonales (format banded pour scipy)
    diag_A  =  1 + 2*r + 1j*dt/(2*H_BARRE) * V
    off_A   = -r * np.ones(nx - 1)

    diag_B  =  1 - 2*r - 1j*dt/(2*H_BARRE) * V
    off_B   =  r * np.ones(nx - 1)

    # Format banded : [diagonale supérieure, diagonale, diagonale inférieure]
    A_banded = np.zeros((3, nx), dtype=complex)
    A_banded[0, 1:]  = off_A   # sur-diagonale
    A_banded[1, :]   = diag_A  # diagonale
    A_banded[2, :-1] = off_A   # sous-diagonale

    mask = masque_absorbant(x)  # Masque absorbant aux bords aux limites du réseau

    # --- Boucle temporelle de résolution ---
    for j in range(nt - 1):
        # Calcul du membre de droite (explicite) : B @ f[j]
        rhs = diag_B * f[j]
        rhs[1:]  += off_B * f[j, :-1]
        rhs[:-1] += off_B * f[j, 1:]

        # Résolution implicite du système tridiagonal
        f[j+1] = solve_banded((1, 1), A_banded, rhs)
        f[j+1] *= mask

    # --- Détection du temps de passage à droite ---
    indices_barriere = np.where(V > 0)[0]
    
    if len(indices_barriere) == 0:
        print("Pas de barrière détectée dans V(x). Capteur calé sur x = 0.")
        x_fin_barriere = 0.0
    else:
        x_fin_barriere = x[indices_barriere[-1]] # Coordonnée X de sortie de barrière
    
    # Isolation de la zone située strictement après l'obstacle
    masque_droite = (x > x_fin_barriere)
    x_zone_droite = x[masque_droite]

    t_passage = None

    # Parcours des résultats pour suivre le pic transmis
    # Parcours des résultats pour suivre le pic transmis
    for j in range(nt):
        densite_proba_droite = np.abs(f[j, masque_droite])**2
        
        if len(densite_proba_droite) == 0:
            continue
            
        # --- CORRECTION DU SEUIL ---
        # On n'analyse la zone que si le pic à droite est supérieur à 1% 
        # de la hauteur maximale globale de l'onde au même instant.
        # Cela élimine mathématiquement la "queue" de la gaussienne de départ.
        max_global_instant_t = np.max(np.abs(f[j, :])**2)
        seuil_detection = max_global_instant_t * 0.01 
        
        if np.max(densite_proba_droite) < seuil_detection:
            continue
            
        # Repérage du sommet local à droite
        indice_pic_droite = np.argmax(densite_proba_droite)
        position_pic_droite = x_zone_droite[indice_pic_droite]
        
        # Validation du franchissement effectif
        if position_pic_droite > (x_fin_barriere + 0.1): 
            t_passage = t[j]
            break

    # --- Affichage du bilan textuel ---
    if t_passage is not None:
        atténuation_pic = np.max(np.abs(f[0,:])**2) - np.max(np.abs(f[j,:])**2)
        print(f"Le sommet du paquet d'onde passe la barrière à t = {t_passage:.4f} s")
        print(f"Diminution de la hauteur du pic : {atténuation_pic:.4f} (dispersion + réflexion)")
    else:
        print("Le paquet d'onde n'a pas franchi la barrière (Réflexion totale ou simulation trop courte).")

    return f, t_passage

def paquet_onde_théorique(x, t, a):
    coeff = (1 / (8*pi**3)) ** 1/4 * np.sqrt(4*pi*M*a / (M*a**2 + 2j*H_BARRE*t))
    exp = np.exp(M*((a**2*k_0+ 2j*x)**2) / (4*(M*a**2 + 2j*H_BARRE*t)) - (a**2*k_0**2)/4)
    return np.abs(coeff * exp)**2

def générer_T(x : np.array, t : np.array) -> None:
    """Génère un graphique du coefficient de transmission T et de réflexion R en fonction de E / V_0"""

    nb_pts = 25
    # On fait varier V_0 (la hauteur de la barrière)
    tab_V_0 = np.linspace(5, 35, nb_pts)
    
    # Calcul de l'énergie moyenne (constante) du paquet d'ondes
    energie_moyenne = (H_BARRE**2 * k_0**2) / (2 * M) + (H_BARRE**2) / (2 * M * (lg_initiale**2))
    
    tous_les_T = []

    # On lance une simulation pour chaque valeur de V_0
    for i, v0 in enumerate(tab_V_0):
        print(f"Simulation {i+1}/{nb_pts} pour V_0 = {v0:.2f} J...")
        
        V_actuel = np.zeros_like(x)

        V_actuel[(x >= 0) & (x <= a_barriere)] = v0 
        
        f, _ = propagation_onde_CN(x, t, V_actuel)
        
        # R=False, T=True pour ne récupérer que la transmission
        T = calculer_RT(x, f, R=False, T=True)
        tous_les_T.append(T)
    
    tous_les_T = np.array(tous_les_T)
    tous_les_R = 1 - tous_les_T
    # Calcul du rapport E / V_0 pour l'axe X du graphique
    rapport_E_V0 = energie_moyenne / tab_V_0

    # -------------------------------------------------------------
    # CONFIGURATION DU GRAPHIQUE
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # On trace T en fonction de E / V_0
    plt.plot(rapport_E_V0, tous_les_T, 'o-', color="purple", linewidth=2, label="Transmission numérique")
    plt.plot(rapport_E_V0, tous_les_R, 'o-', color="green", linewidth=2, label="Réflexion numérique")
    plt.plot()
    
    # Une ligne verticale à E/V_0 = 1 pour séparer l'effet tunnel du régime classique
    plt.axvline(x=1.0, color="red", linestyle="--", label="Limite classique ($E = V_0$)")
    
    plt.xlabel(r"Rapport Énergétique $\langle E \rangle / V_0$", fontsize=11)
    plt.ylabel("Coefficient de Transmission $T$", fontsize=11)
    plt.title(r"Coefficient de Transmission et de Réflexion en fonction de $\langle E \rangle / V_0$", fontsize=12, fontweight="bold")
    plt.legend(loc="best")
    
    plt.show()

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

    t_min_fit = 1.0

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
        return positions_pic, np.array([]), vg

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
        plt.title("Mesure de la vitesse par pente moyenne")
        plt.grid(True)
        plt.legend()
        plt.show()

    return positions_pic, vitesse_fit, vg


#--Fonction principales---------------------------

def main():

    # E = h*nu = h * c / lambda
    # E >> V_0 => E >> 10.0 >> lambda << c*h / 10.0 => lambda ~ 

    while True:
        print("Type de simulation : (entrez 1 ou 2)")
        print("-" * 20)
        print("1. Simulation maison")
        print("2. Simulation optimisée")
        print("3. générer pour une liste de V_0 (entre 5 et 35 J)")
        print("q pour quitter")

        reponse = input("Choix : ")

        if reponse == "q" or reponse == "" : 
            print("Arret du programme...")
            exit(0)

        if int(reponse) in [1,2,3]:
            reponse = int(reponse)
            break
        else :
            print("Erreur, veuillez réessayer...")
        
    borne = 50
    x = np.linspace(-np.abs(borne),np.abs(borne),nx)
    x_0 = -4.0
    V_0 = 35 # on test dans le cas d'une barrière nulle
    # on peut prendre V_0 = 20.0 et a = 1.0 ca marche bien
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= a_barriere)] = V_0

    if reponse == 1 : 
        t_max = 0.2
        t = np.linspace(0,t_max,nt) 
        f, t_passage = propagation_onde(x,t,V)
        tolerance = 1e-02

    elif reponse == 2 : 
        t_max = 2.5
        t = np.linspace(0,t_max,nt) 
        f, t_passage = propagation_onde_CN(x,t,V) #propagation_onde(x,t,V)
        tolerance = 1e-5
    
    elif reponse == 3:
        t_max = 5
        t = np.linspace(0,t_max,nt) 
        générer_T(x,t)
        return

    print("Début de la simuation avec les paramètres : ")
    print("-" * 20)
    print(f"t_max   = {float(t_max)} s")
    print(f"V_0     = {float(V_0)} J")
    print(f"a       = {float(a_barriere)} m")
    print(f"x_0     = {x_0} m")


    if not verifier_normalisation(x,f,tolerance):
        print("La fonction n'est pas normalisée !")
        exit(1)
    
    # ---------------------------------------------------------------------
    # Analyse de la vitesse du pic
    # ---------------------------------------------------------------------
    # Ici, on suit le maximum de densité |psi|² à chaque frame.
    # Puis on compare sa vitesse à la vitesse théorique vg.
    #
    # Choix possible pour zone :
    # - "tout"   : suit le pic dominant dans toute la simulation.
    # - "gauche" : suit surtout le paquet incident/réfléchi avant la barrière.
    # - "droite" : suit seulement le paquet transmis après la barrière.
    #
    # Pour vérifier proprement vg, commence par tester avec V_0 = 0.
    # Pour analyser le paquet transmis après la barrière, mets zone="droite".
    # ---------------------------------------------------------------------

    positions_pic, vitesses_num, vg_pic = analyser_vitesse_pic(
        x=x,
        t=t,
        f=f,
        V=V,
        zone="droite",
        afficher_graphes=True
    )
    
    animation(x, f, t, V, V_0)

    if t_passage == None or t_passage == np.inf:
        t_passage = np.nan

    vg = (H_BARRE * k_0) / M # vitesse de groupe théorique
    print(f"vg théorique    : {vg} m^s-1")
    print(f"distance totale : {np.abs(x_0) + a_barriere} m")
    print(f"---> temps théorique : {(np.abs(x_0) + a_barriere) / vg} s")
    print(f"---> temps simulé    : {t_passage:.3g} s")
    print(f"Erreur de temps : {erreur((np.abs(x_0) + a_barriere) / vg, t_passage)}%")
    #R,T = calculer_RT(x,f,True,True)
    R,T = calculer_RT(x,f,R = True, T = True)
    print(f"R = {R*100:.2g}% | T = {T*100:.2g}%")

main()

# note séance 11/06
# tester programme avec V_0 = 0 et confronter avec Gauss x
# indiquer l'énegie moyenne
# afficher E/V_0
# cherche t tel que l'onde passe la barrière -> vérifier la valeur numérique
# calculer la réfléxion en prenant la crete des deux ondes
# la largeur de la barriere n'influe pas sur la transmission ou la reflexion, mais sur la hauteur de la crete