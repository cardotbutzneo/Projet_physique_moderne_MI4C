import numpy as np
import matplotlib.pyplot as plt
from PaquetOndeGaussMI4_C import GaussWP, C, H_BARRE, M

"""
#Ce code permet de simuler l'équation de Schrödinger en 1D pour une onde gaussienne.

##Fonction : 
- deriver : permet de calculer la dérivée d'une fonction numérique par récurence
- erreur : permet de calculer le pourcentage d'erreur entre une fonction numérique et une

"""

#--Variables --------------------------
precision = 5 # 5 par défaut

##--Paramètres de l'onde gaussienne----
nx = 5 # nombre de points dans l'espace
nt = 100 # nombre de points dans le temps
k_0 = 5 # nombre d'ondes dans la gaussienne
a = 1 # amplitude de la gaussienne
t = 0 # temps initial

#--Fonctions---------------------------

"""
Définition de f'(a) = lim_h->0 (f(a+h) - f(a)) / h = df/dx(a) ~ (f(b) - f(a)) / (b-a)
"""

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

def erreur(f_num : np.array, f_th : np.array):
    """Retourne le pourcentage d'erreur de la version numérique par rapport à la théorie."""
    return np.round((abs(f_num - f_th ) / f_th) * 100, precision) # precision à 0.00001


def Onde2d(x : np.array) -> np.array:
    """Retourne une matrice de nx lignes et nt colonnes représentant l'évolution d'une onde gaussienne dans le temps et l'espace."""

    ligne = GaussWP(k_0, a, x, t)
    arr = np.empty((nx,nt),dtype=complex)

    arr[0,:] = GaussWP(k_0, a, x)
    return arr

#--Fonction principales---------------------------

def main():

    n_derivee = 2
    # on père 1 pt à chaque dérivée
    nb_pts = 100
    x = np.linspace(0,10,nb_pts)
    t = np.linspace()
    f = Onde2d()
    d2f_x = deriver(f,x,k=2)
    df_t = deriver(f,t)

    taille_finale = nb_pts - n_derivee
    d2f_th = np.full(shape=taille_finale, fill_value=2.0)

    # on enlève les premiers pts pour que la dimension de x soit la meme que celle de la dérivée que l'on veut
    x = x[n_derivee:]
    f = f[n_derivee:]

    print("----------------------Affichage de l'erreur de calcul----------------------")
    print("#### Si l'erreur est de 0 c'est que la précision est trop grande, essayez en une plus petite ####")

    plt.plot(x,f, color="blue")
    plt.plot(x,d2f_x, color="red")
    plt.plot(t,df_t,color="green")
    #plt.plot(x,d2f_th,color="green")
    #plt.errorbar(x, df_2, yerr=erreur(df_2,d2f_th), fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.grid(True)
    plt.xlabel("X")

    plt.show()

main()