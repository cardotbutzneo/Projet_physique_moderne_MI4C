#Créateur CARDOT-BUTZ Néo
#Version 1.2
#Création le (09/05/2026)

### Ce programme a pour but de simuler un paquet d'onde sur une marche de potentiel ###

import numpy as np
import matplotlib.pyplot as plt 

from numpy import pi,exp

TWO_PI = 2*pi
M = 9.2e-31 # masse d'une e-
H_BARRE = 6.63e-34

def tracer_graphique(x, t, waves : np.ndarray, k_list : list) -> None:
    """Trace un graphique en fonction des valeurs y en entrée et de t"""
    if (len(k_list) <= 1) :
        print("Calculer plus d'une fonction pour voir la somme")
        return
    
    colors = ["purple","blue","green","yellow","red"]

    fig, axes = plt.subplots(len(k_list)+1, 1, figsize=(8, 10))
    
    for i in range(len(k_list)):
        axes[i].plot(x, waves[i].real, color=colors[i % len(colors)])
        axes[i].set_title(f"Re(Ψ_{i+1})")
        axes[i].set_ylabel(f"k={k_list[i]:.3g}m^-1\n" f"λ={(TWO_PI/k_list[i]):1g}m")
        axes[i].grid(True)
    
    axes[-1].plot(x, waves[-1].real, color='blue')
    axes[-1].set_title("Re(Ψ_tot)")
    axes[-1].set_ylabel(f"λ={(TWO_PI/k_list[-1]):1g}m")
    axes[-1].grid(True)

    plt.tight_layout()
    plt.show()

# retourne toutes les ondes planes
def PlaneWave(amp : float , k : float , omega : float, x : np.ndarray, t : float) -> complex : 
    """Return a plane wave in 1d time reliant"""
    return amp*exp(1j*(k*x - omega * t))

def SumPlanesWaves(psi : np.ndarray) -> np.ndarray:
    """Return the sum of the previous waves functions"""
    return np.sum(psi,axis=0)

def calcul_omega(k : list) :
    if (len(k) <= 0 ) : 
        print("Le nombre de k ne peux pas être négatif")
        return ValueError
    return H_BARRE/2*M*np.power(k,2)
    
# on définit une fonction "main()", qui sert de base au programme
def main():

    print("Entrer les conditions du systeme :")
    try: # permet de vérifier la bonne saisie des valeurs (comme un do while)
        # input() sert à lire la donnée d'entrée (comme scanf())
        t = abs(int(input("Entrer la valeur de t : ")))
        amp = float(input("Entrer la valeur de l'amplitude : "))
    except ValueError: # si il y a une erreur de type on rentre dans le bloque
        print("Erreur : veuillez entrer des nombres valides.")
        return
    
    nb_points = 500 # densité d'affichage (+ il y en a + la courbe est lisse)

    k_centre, dk = 5, 0.1

    x_debut, x_fin = -50, 50 # changer l'intervalle

    x = np.linspace(x_debut,x_fin,nb_points) # calcul de l'axe x

    k = [
        k_centre - 2*dk,
        k_centre,
        k_centre + 2*dk
    ]

    n_waves = 3
    if (n_waves > len(k)): n_waves = len(k)

    omega = calcul_omega(k)

    waves = [PlaneWave(amp,k[i],omega[i],x,t) for i in range(n_waves)] # calcul de l'axe y (grace à numpy on peut directement envoyer un tableau et numpy fait le calcul à notre place)
    waves.append(SumPlanesWaves(waves))

    print("\n"*10)
    print("----------------Rappel des conditions initiales----------------")
    print(f"              k = {k}")
    print(f"              w = {omega}")

    tracer_graphique(x, t, waves, k) # on trace le graphique

if __name__ == "__main__": # on appelle la fonction "main()"
    main()
