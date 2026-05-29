#Creator : CARDOT-BUTZ Néo
#Version : 0.1
#Date of creation : (11/05/20260)


import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, exp, pi

#---Constantes numériques----------------------------------------
TWO_PI : float = 2*pi
H_BARRE : float = 6.63e-34 / TWO_PI # Constante de Planck réduite [J.s]
M : float = 9.109e-31 # Masse d'un électron [kg]

#---Constantes graphique----------------------------------------
COLORS_PLOT : list[str] = ["purple","blue","green","yellow","red"]
TITLE : list[str] = ["Re(Ψ)","Im(Ψ)","Ψ"]
FIGURE_SIZE: tuple[int, int] = (8, 10)

#---Constantes grille spaciale----------------------------------------
x_debut = 0 # début du graphique
x_fin = 10 # fin du graphique
nb_pts = 200 # nombre de points affichés


#---Fonctions----------------------------------------

def tracer_graphique(x : np.ndarray, psi : np.ndarray, t, a, k_0) -> None :
    """
    Trace les parties réelle, imaginaire et leur somme du paquet d'ondes.

    Parameters
    ----------
    x : np.ndarray
        Grille de positions [m].
    psi : np.ndarray
        Fonction d'onde complexe.
    t : float
        Instant d'évaluation [s].
    a : float
        Largeur du paquet [m].
    k_0 : float
        Vecteur d'onde central [m⁻¹].
    """

    wavelength = TWO_PI / k_0
    ylabel = f"a={a} m | k₀={k_0} m⁻¹ | λ={wavelength:.3g} m"

    components = [
        (psi.real,              f"Partie réelle de Ψ(x, t={t})"),
        (psi.imag,              f"Partie imaginaire de Ψ(x, t={t})"),
        (psi.real + psi.imag,   f"Ψ(x, t={t})"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE)
    
    fig.suptitle("Paquet d'ondes gaussien", fontsize=14, fontweight="bold")

    for ax, color, (data, title) in zip(axes, COLORS_PLOT, components):
        ax.plot(x, data, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    plt.show()

def GaussWP(k_0 : float, a : float, x : np.ndarray, t : float) -> np.ndarray :
    """
    Parameters
    ----------
    k_0 : float
        vecteur d'onde [L^-1]
    a : float
        largeur initiale du paquet [L]
    x : np.ndarray
        liste des points à calculer
    t : float
        temps [T]

    Returns
    ----------
    np.ndarray
        Valeurs complexes de la fonction d'onde [1]
    """
    a_carre = a**2
    psi_0 = sqrt(a)/(TWO_PI)**(-3/4)*sqrt(pi/(a_carre/4 + 1j*H_BARRE*t/(2*M))) # [L^-1/2]
    u = (-x**2 + 1j*(a_carre*k_0*x-(a_carre*k_0**2*H_BARRE*t)/(2*M)))/(a_carre+ 1j*(2*H_BARRE*t)/(2*M)) # [1]
    return psi_0*exp(u) # [L^-1/2]

def main():
    try:
        k_0 = float(input("Entrer la valeur de k_0 : "))
        a = float(input("Entrer la valeur de a : "))
        t = float(input("Entrer la valeur de t : "))
    except ValueError:
        print("Erreur : entrer des valeurs valides")
        exit(1)

    x = np.linspace(x_debut,x_fin,nb_pts)

    waves = GaussWP(k_0,a,x,t)

    tracer_graphique(x,waves,t,a,k_0)

if __name__ == "__main__":
    main()