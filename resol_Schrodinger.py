import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PaquetOndeGaussMI4_C import GaussWP, tracer_graphique, C, H_BARRE, M

"""
#Ce code permet de simuler l'équation de Schrödinger en 1D pour une onde gaussienne.

##Fonction : 
- deriver : permet de calculer la dérivée d'une fonction numérique par récurence
- erreur : permet de calculer le pourcentage d'erreur entre une fonction numérique et une
"""

#--Variables --------------------------
precision = 5 # 5 par défaut

##--Paramètres de l'onde gaussienne----
nx = 300   # nombre de points dans l'espace
nt = 500   # nombre de points dans le temps
k_0 = 5    # nombre d'ondes dans la gaussienne
a = 1      # amplitude de la gaussienne
t_0 = 0      # temps initial

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
    """Retourne une matrice de nx lignes et nt colonnes représentant l'évolution d'une onde gaussienne dans l'espace à un t donné."""
    arr = np.empty((nt ,nx),dtype=complex)

    arr[0,:] = GaussWP(k_0, a, x, t_0)
    return arr

def propagation_onde(x : np.array, t : np.array) -> np.array :
    """Simule le déplacement d'un paquet d'onde"""
    f = Onde2d(x) #initialisation de la fonction d'onde
    V = 0 # on doit meme utiliser un V(x)
    dx = x[1] - x[0]
    dt = t[1] - t[0] # on peut le faire car on utilise linspace pour faire les ensembles

    # on simule l'équation avec l'eq de Sch

    for j in range(0,len(t)-1): # avance dans le temps de 0 à nt-2
        for i in range(1,len(x)-1): # on avant dans l'espace de 1 à nx-1
            d2f_dx2 = (f[j, i+1] - 2*f[j, i] + f[j, i-1]) / dx**2
            hamiltonien = (- H_BARRE ** 2) / (2*M) * d2f_dx2 + V * f[j,i]
            df_t = (- 1j / H_BARRE) * hamiltonien  # <-- hum
            f[j+1, i] = f[j, i] + dt * df_t # on modifie la ligne pour avec à l'instant t + ndt

    return f

def animation(x : np.array, matrice_f : np.array, t : np.array) -> None:
    """Créer une animation de la fonction d'onde passée en paramètre
    
    Paramètres:  
    ------------
    x : np.array
        Tableau des abscices [m]
    matrice_f : np.array
        Matrice des valeurs prises par la fonction d'onde pour chaque valeur de t
    t : np.array
        Tableau des valeurs du temps [T]
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    zeros_init = np.zeros_like(x)

    # création de ligne vide mises à jour
    line_real, = axes[0].plot(x, zeros_init, color="blue")
    line_imag, = axes[1].plot(x, zeros_init, color="orange")
    line_sum,  = axes[2].plot(x, zeros_init, color="green")

    for ax in axes:
        ax.set_xlim(x[0], x[-1])
        # On calcule le max de la fonction d'onde absolue pour caler l'axe Y
        ymax = np.max(np.abs(matrice_f)) * 1.2
        ax.set_ylim(-ymax, ymax)
        ax.grid(True)
        
    axes[0].set_title("Partie Réelle")
    axes[1].set_title("Partie Imaginaire")
    axes[2].set_title("Somme (Réelle + Imaginaire)")

    # Fonction appelée à chaque frame (chaque instant j)
    def update(j):
        psi_actuel = matrice_f[j, :]
        
        # Mise à jour des données des courbes
        line_real.set_data(x, psi_actuel.real)
        line_imag.set_data(x, psi_actuel.imag)
        line_sum.set_data(x, psi_actuel.real + psi_actuel.imag)
        
        fig.suptitle(f"Évolution du Paquet d'ondes - t = {t[j]:.3f} s", fontsize=14, fontweight="bold")
        return line_real, line_imag, line_sum

    # Création de l'animation (interval = temps en ms entre chaque image)
    ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
    plt.tight_layout()
    plt.show()

#--Fonction principales---------------------------

def main():
    x = np.linspace(-10,10,nx)
    t = np.linspace(0,0.5,nt) # simulation sur 10s
    f = propagation_onde(x,t)
    # on trace à t=t_0, t = (t_0 + dt)/2 et t = t_0+  nt*dt
    
    animation(x,f,t)

main()