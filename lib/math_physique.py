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

def erreur(f_num : np.float64, f_th : np.float64, precision : int = 5, erreur : float = 1e-5):
    """Retourne le pourcentage d'erreur de la version numérique par rapport à la théorie."""
    norme = abs(f_num - f_th) # float
    if norme <= erreur : norme = 0.0
    return np.round(norme / f_th * 100, precision) # precision à 0.00001

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

    print(f"Norme initiale : {norme_ini:.6f}")
    print(f"Norme finale   : {norme_fin:.6f}")
    print(f"Dérive max     : {derive_max:.2e}  (tolérance = {tol:.0e})")

    if derive_max > tol:
        print("Norme non conservée — schéma instable ou masque trop agressif")
        return False

    print("Fonction normalisée")
    return True

def calculer_RT(x : np.array, f : np.array, R = True, T = False) -> float : 
    """Retourne le coefficient de réfléxion et/ou celui de transmission pour la dernière valeur de t
    --------
    x : tableau 1d [M] -- abscice
    f : tableau 2d (nt*nx) -- la fonction d'onde
    R : bool -- réflexion 
    T : bool -- transission
    """

    if (not R and not T) : 
        print("La fonction doit prendre au moin un parametre en True")
        return -1.0, -1.0
    
    if (len(f) <= 0):
        print("La fonction d'onde ne peut pas être vide")
        return -1.0, -1.0

    zone_negative = x[x <= 0] # utile pour la réflexion
    indice_negatif = np.where(x <= 0)[0]
    reflexion = np.trapz(np.abs(f[-1,indice_negatif])**2,zone_negative) # on calcul l'intégrale pour la réflexion (le + simple)
    transmission = 1 - reflexion

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