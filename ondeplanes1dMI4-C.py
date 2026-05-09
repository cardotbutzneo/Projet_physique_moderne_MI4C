import numpy as np
import matplotlib.pyplot as plt 

from numpy import pi,exp,sqrt,real,imag,zeros,linspace,cos

def tracer_graphique(x, y, t):
    """Trace un graphique en fonction des valeurs y en entrée et de t"""
    #q1
    plt.plot(x, y.real, label=f"Re(Ψ)", color="blue", linewidth=2)
    #plt.plot(x, y[0].imag, label=f"Im(Ψ)", color="red", linestyle="--", alpha=0.7)
    # q2
    psi0 = 1
    k = 5.0
    delta_k = 0.5
    # on calcule directement les trois fonctions dans le graphique pour plus de simplicité mais il faudrait le faire à part normalement.
    psi1 = psi0 * np.cos(k * x)
    psi2 = (psi0 / 2) * cos((k + delta_k) * x)
    psi3 = (psi0 / 2) * cos((k - delta_k) * x)
    psi_tot = psi1 + psi2 + psi3

    # affiche sur le meme graphique les 3 fonctions
    plt.plot(x, psi1.real, label=f"Re(Ψ_1)", color="red", linewidth=2)
    plt.plot(x, psi2.real, label=f"Re(Ψ_2)", color="yellow", linewidth=2)
    plt.plot(x, psi3.real, label=f"Re(Ψ_3)", color="green", linewidth=2)
    plt.plot(x, psi_tot.real, label=f"Re(Ψ_tot)", color="black", linewidth=2)
    
    plt.xlabel("Position x")
    plt.ylabel("Amplitude")
    plt.title(f"Fonction d'onde Ψ(x,t) à t={t}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1) # Axe horizontal
    plt.show()

# retourne l'onde plane
def PlaneWave(amp : float , k : float , omega : float, x : np.ndarray, t : float) -> complex : 
    """Return a plane wave in 1d time reliant"""
    return amp*exp(1j*(omega * t - k * x)) 

# on définit une fonction "main", qui sert de base au programme
def main():

    print("Entrer les conditions du systeme :")
    try: # permet de vérifier la bonne saisie des valeur (comme un do while)
        # input sert à lire la donnée d'entrée (comme scanf())
        t = int(input("Entrer la valeur de t : "))
        amp = float(input("Entrer la valeur de l'amplitude : "))
        omega = float(input("Entrer la valeur de la pulsation propre : "))
        k = float(input("Entrer la valeur de k : "))
        dk = float(input("Entrer la valeur de Δk : "))
    except ValueError: # si il y a une erreur de type on rentre dans le bloque
        print("Erreur : veuillez entrer des nombres valides.")
        return
    
    nb_points = 200 # densité d'affichage (+ il y en a + la courbe est lisse)

    x = np.linspace(-pi/dk, pi/dk, nb_points) # calcul de l'axe x sur [-pi/delta_k,pi/delta_k]

    y = PlaneWave(amp, k, omega, x, t) # calcul de l'axe y (grace à numpy on peut directement envoyé un tableau et numpy fait le calcul à notre place)
    
    tracer_graphique(x,y,t) # on trace le graphique

if __name__ == "__main__": # on appel la fonction "main()"
    main()