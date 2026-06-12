import numpy as np
from lib.constantes import *
from lib.PaquetOndeGaussMI4_C import GaussWP, C , pi #, H_BARRE, M
from lib.math_physique import erreur, verifier_normalisation
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

def propagation_onde_CN(x : np.array, t : np.array, V : np.array, x_0 : float = -4) -> np.array : # fait uniquement par Claude (je n'ai pas les connaissances mathématiques pour le faire Néo)
    """Permet de calculer la propagation du paquet d'onde via la méthode de Crank-Nicolson
    Paramètres : 
    ------------
    - x : np.array
        abscice [L]
    - t : np.array
        temps [T]
    - V : np.array
        potentiel [J] -- Tableau a 1d
    - x_0 : float
        position initiale [L]
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    r = 1j * H_BARRE * dt / (4 * M * dx**2)  # coefficient central

    f = np.zeros((nt, nx), dtype=complex)
    f[0, :] = GaussWP(k_0, lg_initiale, x, t[0], x_0)

    # Matrices tridiagonales (format banded pour scipy)
    # Matrice A (côté gauche  : implicite) : A @ f[j+1] = B @ f[j]
    # Matrice B (côté droit   : explicite)

    diag_A  =  1 + 2*r + 1j*dt/(2*H_BARRE) * V
    off_A   = -r * np.ones(nx - 1)

    diag_B  =  1 - 2*r - 1j*dt/(2*H_BARRE) * V
    off_B   =  r * np.ones(nx - 1)

    # Format banded : [diagonale supérieure, diagonale, diagonale inférieure]
    A_banded = np.zeros((3, nx), dtype=complex)
    A_banded[0, 1:]  = off_A   # sur-diagonale
    A_banded[1, :]   = diag_A  # diagonale
    A_banded[2, :-1] = off_A   # sous-diagonale

    mask = masque_absorbant(x)  # ton masque absorbant aux bords

    for j in range(nt - 1):
        # Calcul du membre de droite : B @ f[j]
        rhs = diag_B * f[j]
        rhs[1:]  += off_B * f[j, :-1]
        rhs[:-1] += off_B * f[j, 1:]

        # Résolution du système tridiagonal
        f[j+1] = solve_banded((1, 1), A_banded, rhs)
        f[j+1] *= mask

    # 1. On trouve l'indice X de la fin de ta barrière de potentiel
    # On cherche le dernier endroit où V(x) n'est pas nul
    indices_barriere = np.where(V > 0)[0]
    
    if len(indices_barriere) == 0:
        print("Pas de barrière détectée dans V(x).")
        x_fin_barriere = a_barriere

    else :
        x_fin_barriere = x[indices_barriere[-1]] # La coordonnée X de sortie de barrière
        print(f"test : {x_fin_barriere}")
    
    t_passage = None

    # 2. On parcourt chaque instant t (chaque ligne j) pour suivre le sommet
    for j in range(nt):
        densite_proba = np.abs(f[j, :])**2
        
        # On trouve l'indice X où la probabilité est maximale à cet instant précis
        indice_pic = np.argmax(densite_proba)
        position_pic = x[indice_pic]
        
        # Si le sommet du paquet vient de dépasser la fin de la barrière : BINGO !
        if position_pic > x_fin_barriere:
            t_passage = t[j]
            break # On arrête la recherche dès qu'il est passé

    if t_passage is not None:
        print(f"Le sommet du paquet d'onde passe la barrière à t = {t_passage:.4f} s")
    else:
        print("Le paquet d'onde n'a pas franchi la barrière (Réflexion totale ou simulation trop courte).")

    return f, t_passage

def paquet_onde_théorique(x, t, a):
    coeff = (1 / (8*pi**3)) ** 1/4 * np.sqrt(4*pi*M*a / (M*a**2 + 2j*H_BARRE*t))
    exp = np.exp(M*((a**2*k_0+ 2j*x)**2) / (4*(M*a**2 + 2j*H_BARRE*t)) - (a**2*k_0**2)/4)
    return np.abs(coeff * exp)**2

#--Fonction principales---------------------------

def main():

    # E = h*nu = h * c / lambda
    # E >> V_0 => E >> 10.0 >> lambda << c*h / 10.0 => lambda ~ 

    while True:
        print("Type de simulation : (entrez 1 ou 2)")
        print("-" * 20)
        print("1. Simulation maison")
        print("2. Simulation optimisée")
        print("q pour quitter")

        reponse = input("Choix : ")

        if reponse == "q" or reponse == "" : 
            print("Arret du programme...")
            exit(0)

        if int(reponse) in [1,2]:
            reponse = int(reponse)
            break
        else :
            print("Erreur, veuillez réessayer...")
        
    borne = 25
    x = np.linspace(-np.abs(borne),np.abs(borne),nx)
    x_0 = -4.0
    V_0 = 20.0 # on test dans le cas d'une barrière nulle
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

    print("Début de la simuation avec les paramètres : ")
    print("-" * 20)
    print(f"t_max   = {float(t_max)} s")
    print(f"V_0     = {float(V_0)} J")
    print(f"a       = {float(a_barriere)} m")
    print(f"x_0     = {x_0} m")


    if not verifier_normalisation(x,f,tolerance):
        print("La fonction n'est pas normalisée !")
        exit(1)
    
    animation(x, f, t, V, V_0)

    if t_passage == None or t_passage == np.inf:
        t_passage = np.nan

    vg = (H_BARRE * k_0) / M # vitesse de groupe théorique
    print(f"vg théorique    : {vg} m^s-1")
    print(f"distance totale : {np.abs(x_0) + a_barriere} m")
    print(f"---> temps théorique : {(np.abs(x_0) + a_barriere) / vg} s")
    print(f"---> temps simulé    : {t_passage:.3g} s")
    print(f"Erreur de temps : {erreur((np.abs(x_0) + a_barriere) / vg, t_passage)}%")

main()

# note séance 11/06
# tester programme avec V_0 = 0 et confronter avec Gauss x
# indiquer l'énegie moyenne
# afficher E/V_0
# cherche t tel que l'onde passe la barrière -> vérifier la valeur numérique avec la valeur théorique en utilisant la vistesse de groupe