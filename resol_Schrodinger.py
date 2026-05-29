import numpy as np
import matplotlib.pyplot as plt
from paquetOndeGaussMI4_C.py import GaussWP

#--Variables --------------------------
precision = 5 # 5 par défaut

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


def Onde2d():
    nx = 5
    nt = 100
    k_0 = 5
    a = 1
    t = 0
    x = np.linspace(0,10,nt)

    ligne = GaussWP(k_0, a, x, t)
    arr = np.empty((nx,nt),dtype=complex)

    arr[0,:] = GaussWP(k_0, a, x, t)
    return arr

#--Fonction principales---------------------------

def main():

    n_derivee = 2
    # on père 1 pt à chaque dérivée
    nb_pts = 100
    x = np.linspace(0,10,nb_pts)
    f = np.power(x,2)
    d2f_th = np.array(2) * (nb_pts - n_derivee)
    df_2 = deriver(f,x,n_derivee)

    taille_finale = nb_pts - n_derivee
    d2f_th = np.full(shape=taille_finale, fill_value=2.0)

    # on enlève les premiers pts pour que la dimension de x soit la meme que celle de la dérivée que l'on veut
    x = x[n_derivee:]
    f = f[n_derivee:]

    print("----------------------Affichage de l'erreur de calcul----------------------")
    print("#### Si l'erreur est de 0 c'est que la précision est trop grande, essayez en une plus petite ####")
    print(erreur(df_2,d2f_th)[:5])

    plt.plot(x,f, color="blue")
    plt.plot(x,df_2, color="red")
    plt.plot(x,d2f_th,color="green")
    plt.grid(True)
    plt.xlabel("X")

    plt.show()

main()