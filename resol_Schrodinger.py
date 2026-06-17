import sys
import numpy as np
import matplotlib.pyplot as plt
from lib.constantes import *
from lib.PaquetOndeGaussMI4_C import GaussWP, C , pi #, H_BARRE, M
from lib.math_physique import *
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

tab_affichage = [] #tableau pour afficher tous les prints

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
        tab_affichage.append("Pas de barrière détectée dans V(x). Capteur calé sur x = {}m.".format(a_barriere))
        x_fin_barriere = a_barriere
    else:
        x_fin_barriere = x[indices_barriere[-1]] # Coordonnée X de sortie de barrière

    t_passage = detecter_temps_passage(x, t, f, x_ligne_arrivee=a_barriere, seuil_relatif=0.05)
    j_detecte = np.argmin(np.abs(t - t_passage))

    densite_a_detection = np.abs(f[j_detecte, :])**2
    indice_pic_global = np.argmax(densite_a_detection)
    position_pic_reel = x[indice_pic_global]

    tab_affichage.append("t_passage détecté : {:.4f}s".format(t_passage))
    tab_affichage.append("Position du PIC RÉEL (sur tout x) à cet instant : {:.3f}m".format(position_pic_reel))
    tab_affichage.append("x_ligne_arrivee : {}".format(a_barriere))

    if t_passage is not None:
        tab_affichage.append("[Capteur numérique] Ligne d'arrivée x = {:.2f} franchie à t = {:.4f}s".format(x_fin_barriere, t_passage))
    else:
        tab_affichage.append("[Capteur numérique] Ligne d'arrivée x = {:.2f} NON franchie.".format(x_fin_barriere))

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
        tab_affichage.append(f"Simulation {i+1}/{nb_pts} pour V_0 = {v0:.2f} J...")
        
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


#--Fonction principales---------------------------

def main(an=False):

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
    dx = x[1] - x[0]
    lambda_0 = 2 * np.pi / k_0

    print(f"dx = {dx}")
    print(f"{lambda_0=}")
    print(f"Nombre de points par longueur d'onde = {lambda_0 / dx}")
    print(f"k_0 * dx = {k_0 * dx}")

    V_0 = 20 # on test dans le cas d'une barrière nulle
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

    positions_pic, vitesses_num, vg_pic, position_initiale_fit = analyser_vitesse_pic(
        x=x,
        t=t,
        f=f,
        V=V,
        zone="droite",
        afficher_graphes=True
    )
    print("-"*25)
    print("Résultats : ")
    print("="*25)

    print("Vitesse numérique du pic : {:.6g} m/s".format(vitesses_num))

    if erreur(vitesses_num, vg_pic) < 5.0:
        print("La vitesse numérique est correcte (erreur < 5%)")
    else:
        print("La vitesse numérique est incorrecte (erreur > 5%)")
        exit(1)

    if an : animation(x, f, t, V, V_0)

    if t_passage == None or t_passage == np.inf:
        t_passage = np.nan

    vg = (H_BARRE * k_0) / M
    print(f"vg théorique         : {vg} m/s")
    print(f"vitesse mesurée       : {vitesses_num} m/s")
    print(f"position initiale fit : {position_initiale_fit:.3f} m (x_0 = {x_0})")

    # Distance réelle parcourue par le pic, mesurée depuis sa position extrapolée
    distance_corrigee = a_barriere - position_initiale_fit
    temps_theorique_corrige = distance_corrigee / vitesses_num   # <-- vitesse mesurée, pas vg_pic

    print(f"Temps théorique corrigé : {temps_theorique_corrige:.4f} s")
    print(f"Temps calculé (t_passage) : {t_passage:.4f} s")
    print(f"Erreur : {erreur(t_passage, temps_theorique_corrige):.2f}%")
    R,T = calculer_RT(x,f,R = True, T = True)
    print(f"R = {R*100:.2g}% | T = {T*100:.2g}%")

    print("")
    print("-"*10, " Debug de la simulation ", "-"*10)
    for msg in tab_affichage:
        print(msg)

    exit(0)

argv = sys.argv

if (len(argv) > 1):
    if bool(argv[1]):
        main(True)

main(False)

# note séance 11/06
# tester programme avec V_0 = 0 et confronter avec Gauss x
# indiquer l'énegie moyenne
# afficher E/V_0
# cherche t tel que l'onde passe la barrière -> vérifier la valeur numérique
# calculer la réfléxion en prenant la crete des deux ondes
# la largeur de la barriere n'influe pas sur la transmission ou la reflexion, mais sur la hauteur de la crete