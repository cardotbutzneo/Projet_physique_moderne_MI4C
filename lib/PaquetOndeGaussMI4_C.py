#Creator : CARDOT-BUTZ Néo
#Version : 0.1
#Date of creation : (11/05/20260)


import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, exp, pi

#---Constantes numériques----------------------------------------
# Toutes les constantes sont données avec 3 chiffres significatifs quand cela est nécessaire
TWO_PI : float = 2*pi
H_BARRE : float = 1 #= 6.626e-34 / TWO_PI # Constante de Planck réduite [J.s]
M : float = 1 #= 9.109e-31 # Masse d'un électron [kg]
C : float = 2.99792458e8 # vitesse de la lumière dans le vide [m.s^-1]

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

def GaussWP(k_0 : float, a : float, x : np.ndarray, t : float, x_0 : float, H_BARRE=H_BARRE, M=M) -> np.ndarray :
    """
    Paquet d'ondes gaussien normalisé en dimension 1 pour l'équation de Schrödinger libre.
    
    a : Paramètre de largeur initiale du paquet
    x_0 : Position initiale du centre du paquet
    """
    
    # 1. Calcul de la dispersion temporelle (étalement du paquet)
    # On définit le terme complexe lié à l'étalement : s(t) = a^2 + 2 * j * h_barre * t / M
    dispersion = a**2 + 2j * H_BARRE * t / M
    
    # 2. Vitesse de groupe et position du centre au cours du temps
    v_g = (H_BARRE * k_0) / M
    centre_t = x_0 + v_g * t
    
    # 3. Facteur de normalisation (Garantit que l'intégrale de |Psi|^2 vaut 1)
    # L'amplitude de la crête diminue au cours du temps car le paquet s'étale
    psi_0 = (2 * np.pi * a**2)**(-1/4) * np.sqrt(a**2 / dispersion)
    
    # 4. Phase et enveloppe spatiale
    # Le premier terme gère la forme de la cloche qui s'étale autour du centre mobile.
    # Le second terme gère la phase d'oscillation de l'onde (quantité de mouvement et énergie).
    enveloppe = -(x - centre_t)**2 / (4 * dispersion)
    phase_onde = 1j * k_0 * (x - x_0) - 1j * (H_BARRE * k_0**2 * t) / (2 * M)
    
    return psi_0 * np.exp(enveloppe + phase_onde)

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