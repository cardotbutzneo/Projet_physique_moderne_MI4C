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

def masque_absorbant(x, largeur=3e-9):
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
    f[0, :] = GaussWP(k_0, lg_initiale, x, t[0], x_0,H_BARRE=H_BARRE, M=M)

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

    x_capteur = x_fin_barriere + 3e-9 # distance arbitraire
    t_passage = detecter_temps_passage(x, t, f, x_ligne_arrivee=x_capteur, seuil_relatif=0.0001)
    if t_passage is None : return f, None
    j_detecte = np.argmin(np.abs(t - t_passage))

    densite_a_detection = np.abs(f[j_detecte, :])**2
    indice_pic_global = np.argmax(densite_a_detection)
    position_pic_reel = x[indice_pic_global]

    tab_affichage.append("t_passage détecté : {:.4g}s".format(t_passage))
    tab_affichage.append("Position du PIC RÉEL (sur tout x) à cet instant : {:.3g}m".format(position_pic_reel))
    tab_affichage.append("x_ligne_arrivee : {}".format(a_barriere))

    if t_passage is not None:
        tab_affichage.append("[Capteur numérique] Ligne d'arrivée x = {:.2g} franchie à t = {:.4g}s".format(x_fin_barriere, t_passage))
    else:
        tab_affichage.append("[Capteur numérique] Ligne d'arrivée x = {:.2g} NON franchie.".format(x_fin_barriere))

    return f, t_passage

def paquet_onde_théorique(x, t, a):
    coeff = (1 / (8*pi**3)) ** 1/4 * np.sqrt(4*pi*M*a / (M*a**2 + 2j*H_BARRE*t))
    exp = np.exp(M*((a**2*k_0+ 2j*x)**2) / (4*(M*a**2 + 2j*H_BARRE*t)) - (a**2*k_0**2)/4)
    return np.abs(coeff * exp)**2

def générer_T(x : np.array, t : np.array) -> None:
    """Génère un graphique du coefficient de transmission T et de réflexion R en fonction de E / V_0"""

    nb_pts = 25
    # On fait varier V_0 (la hauteur de la barrière)
    tab_V_0_ev = np.linspace(0.3, 2.5, nb_pts)
    tab_V_0 = tab_V_0_ev * 1.602e-19 
    
    # Calcul de l'énergie moyenne (constante) du paquet d'ondes
    energie_moyenne = (H_BARRE**2 * k_0**2) / (2 * M) + (H_BARRE**2) / (2 * M * (lg_initiale**2))
    
    tous_les_T = []

    # On lance une simulation pour chaque valeur de V_0
    for i, v0 in enumerate(tab_V_0):
        V0_ev = v0 / 1.602e-19
        print(f"Simulation {i+1}/{nb_pts} pour V_0 = {V0_ev:.2f} eV...")
        
        V_actuel = np.zeros_like(x)

        V_actuel[(x >= 0) & (x <= a_barriere)] = v0 
        
        f, _ = propagation_onde_CN(x, t, V_actuel,x_0=x_0)
        
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

def etats_stationnaires_barriere(
    x: np.ndarray,
    V_0: float,
    a: float,
    E: float | None = None,
    afficher: bool = True
) -> dict:
    """
    Calcule numériquement l'état stationnaire de diffusion associé
    à une énergie E pour une barrière rectangulaire :

        V(x) = 0      si x < 0
        V(x) = V_0    si 0 <= x <= a
        V(x) = 0      si x > a

    L'amplitude de l'onde incidente est fixée à A = 1.

    Les coefficients B, C, D et F sont obtenus en résolvant
    numériquement les conditions de continuité de psi et dpsi/dx
    aux interfaces x = 0 et x = a.

    Paramètres
    ----------
    x : np.ndarray
        Maillage spatial.
    V_0 : float
        Hauteur de la barrière en joules.
    a : float
        Largeur de la barrière en mètres.
    E : float ou None
        Énergie de l'état stationnaire en joules.
        Si E=None, on utilise l'énergie moyenne du paquet gaussien.
    afficher : bool
        Affiche les graphiques si True.

    Retour
    ------
    dict
        Dictionnaire contenant E, omega, k1, k2, k3,
        les amplitudes et l'état stationnaire psi(x).
    """

    # ---------------------------------------------------------
    # Énergie choisie
    # ---------------------------------------------------------

    if E is None:
        # Énergie moyenne du paquet gaussien :
        #
        # premier terme  : énergie associée au nombre d'onde central k_0
        # second terme   : correction liée à la largeur du paquet
        E = (
            H_BARRE**2 * k_0**2 / (2 * M)
            + H_BARRE**2 / (2 * M * lg_initiale**2)
        )

    if E <= 0:
        raise ValueError("L'énergie E doit être strictement positive.")

    if a <= 0:
        raise ValueError("La largeur a de la barrière doit être positive.")

    # Pulsation temporelle commune aux trois zones
    omega = E / H_BARRE
    # ---------------------------------------------------------
    # Zone 1 et zone 3 : V = 0
    # ---------------------------------------------------------
    k1 = np.sqrt(2 * M * E) / H_BARRE
    k3 = k1
    # Amplitude de l'onde incidente
    A = 1.0 + 0.0j
    # ---------------------------------------------------------
    # Cas 1 : E > V0, onde propagative dans la barrière
    # ---------------------------------------------------------
    if E > V_0:
        regime = "propagatif"

        k2 = np.sqrt(2 * M * (E - V_0)) / H_BARRE
        kappa = None

        # Les inconnues sont :
        #
        # u = [B, C, D, F]
        #
        # Zone 1 :
        # psi1 = A exp(i k1 x) + B exp(-i k1 x)
        #
        # Zone 2 :
        # psi2 = C exp(i k2 x) + D exp(-i k2 x)
        #
        # Zone 3 :
        # psi3 = F exp(i k3 x)

        # Conditions en x = 0 :
        #
        # A + B = C + D
        # k1(A - B) = k2(C - D)
        #
        # Conditions en x = a :
        #
        # C exp(i k2 a) + D exp(-i k2 a)
        #     = F exp(i k3 a)
        #
        # k2[C exp(i k2 a) - D exp(-i k2 a)]
        #     = k3 F exp(i k3 a)

        matrice = np.array(
            [
                [
                    1,
                    -1,
                    -1,
                    0
                ],
                [
                    -k1,
                    -k2,
                    k2,
                    0
                ],
                [
                    0,
                    np.exp(1j * k2 * a),
                    np.exp(-1j * k2 * a),
                    -np.exp(1j * k3 * a)
                ],
                [
                    0,
                    k2 * np.exp(1j * k2 * a),
                    -k2 * np.exp(-1j * k2 * a),
                    -k3 * np.exp(1j * k3 * a)
                ]
            ],
            dtype=complex
        )
        second_membre = np.array(
            [
                -A,
                -k1 * A,
                0,
                0
            ],
            dtype=complex
        )
        B, C_stationnaire, D, F = np.linalg.solve(
            matrice,
            second_membre
        )

        # Construction de psi(x) sur le maillage numérique
        psi_stationnaire = np.zeros_like(x, dtype=complex)

        zone_1 = x < 0
        zone_2 = (x >= 0) & (x <= a)
        zone_3 = x > a

        psi_stationnaire[zone_1] = (
            A * np.exp(1j * k1 * x[zone_1])
            + B * np.exp(-1j * k1 * x[zone_1])
        )

        psi_stationnaire[zone_2] = (
            C_stationnaire * np.exp(1j * k2 * x[zone_2])
            + D * np.exp(-1j * k2 * x[zone_2])
        )

        psi_stationnaire[zone_3] = (
            F * np.exp(1j * k3 * x[zone_3])
        )
    # ---------------------------------------------------------
    # Cas 2 : E < V0, effet tunnel
    # ---------------------------------------------------------
    elif E < V_0:
        regime = "tunnel"

        kappa = np.sqrt(2 * M * (V_0 - E)) / H_BARRE
        k2 = 1j * kappa

        # Dans la barrière :
        #
        # psi2 = C exp(kappa x) + D exp(-kappa x)

        # Conditions en x = 0 :
        #
        # A + B = C + D
        #
        # i k1(A - B) = kappa(C - D)
        #
        # Conditions en x = a :
        #
        # C exp(kappa a) + D exp(-kappa a)
        #     = F exp(i k3 a)
        #
        # kappa[C exp(kappa a) - D exp(-kappa a)]
        #     = i k3 F exp(i k3 a)

        matrice = np.array(
            [
                [
                    1,
                    -1,
                    -1,
                    0
                ],
                [
                    -1j * k1,
                    -kappa,
                    kappa,
                    0
                ],
                [
                    0,
                    np.exp(kappa * a),
                    np.exp(-kappa * a),
                    -np.exp(1j * k3 * a)
                ],
                [
                    0,
                    kappa * np.exp(kappa * a),
                    -kappa * np.exp(-kappa * a),
                    -1j * k3 * np.exp(1j * k3 * a)
                ]
            ],
            dtype=complex
        )
        second_membre = np.array(
            [
                -A,
                -1j * k1 * A,
                0,
                0
            ],
            dtype=complex
        )
        B, C_stationnaire, D, F = np.linalg.solve(
            matrice,
            second_membre
        )

        # Construction de psi(x) sur le maillage
        psi_stationnaire = np.zeros_like(x, dtype=complex)

        zone_1 = x < 0
        zone_2 = (x >= 0) & (x <= a)
        zone_3 = x > a

        psi_stationnaire[zone_1] = (
            A * np.exp(1j * k1 * x[zone_1])
            + B * np.exp(-1j * k1 * x[zone_1])
        )

        psi_stationnaire[zone_2] = (
            C_stationnaire * np.exp(kappa * x[zone_2])
            + D * np.exp(-kappa * x[zone_2])
        )

        psi_stationnaire[zone_3] = (
            F * np.exp(1j * k3 * x[zone_3])
        )
    # ---------------------------------------------------------
    # Cas particulier : E = V0
    # ---------------------------------------------------------
    else:
        regime = "limite E = V0"

        k2 = 0.0
        kappa = 0.0

        # Pour E = V0, l'équation dans la barrière est :
        #
        # d²psi/dx² = 0
        #
        # donc :
        #
        # psi2 = C + D x
        matrice = np.array(
            [
                [
                    1,
                    -1,
                    0,
                    0
                ],
                [
                    -1j * k1,
                    0,
                    -1,
                    0
                ],
                [
                    0,
                    -1,
                    -a,
                    np.exp(1j * k3 * a)
                ],
                [
                    0,
                    0,
                    -1,
                    1j * k3 * np.exp(1j * k3 * a)
                ]
            ],
            dtype=complex
        )
        second_membre = np.array(
            [
                -A,
                -1j * k1 * A,
                0,
                0
            ],
            dtype=complex
        )
        # Ici les inconnues sont B, C, D, F
        B, C_stationnaire, D, F = np.linalg.solve(
            matrice,
            second_membre
        )
        psi_stationnaire = np.zeros_like(x, dtype=complex)

        zone_1 = x < 0
        zone_2 = (x >= 0) & (x <= a)
        zone_3 = x > a

        psi_stationnaire[zone_1] = (
            A * np.exp(1j * k1 * x[zone_1])
            + B * np.exp(-1j * k1 * x[zone_1])
        )

        psi_stationnaire[zone_2] = (
            C_stationnaire + D * x[zone_2]
        )

        psi_stationnaire[zone_3] = (
            F * np.exp(1j * k3 * x[zone_3])
        )

    # ---------------------------------------------------------
    # Coefficients de réflexion et transmission
    # ---------------------------------------------------------

    R_stationnaire = np.abs(B / A)**2

    # Formule générale faisant intervenir le rapport des courants
    T_stationnaire = (
        np.abs(F / A)**2
        * np.real(k3)
        / np.real(k1)
    )

    # ---------------------------------------------------------
    # Vérification numérique de l'équation stationnaire
    # ---------------------------------------------------------

    dx = x[1] - x[0]

    potentiel_stationnaire = np.zeros_like(x)
    potentiel_stationnaire[
        (x >= 0) & (x <= a)
    ] = V_0

    derivee_seconde = (
        psi_stationnaire[2:]
        - 2 * psi_stationnaire[1:-1]
        + psi_stationnaire[:-2]
    ) / dx**2

    hamiltonien_psi = (
        -H_BARRE**2 / (2 * M) * derivee_seconde
        + potentiel_stationnaire[1:-1]
        * psi_stationnaire[1:-1]
    )

    residu = (
        hamiltonien_psi
        - E * psi_stationnaire[1:-1]
    )

    # On retire les points proches des discontinuités du potentiel
    # car la dérivée seconde discrète y est moins précise.
    masque_residu = (
        (np.abs(x[1:-1]) > 3 * dx)
        & (np.abs(x[1:-1] - a) > 3 * dx)
    )

    norme_reference = np.linalg.norm(
        E * psi_stationnaire[1:-1][masque_residu]
    )

    residu_relatif = (
        np.linalg.norm(residu[masque_residu])
        / max(norme_reference, 1e-300)
    )

    # ---------------------------------------------------------
    # Affichage des résultats
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("ÉTATS STATIONNAIRES DANS LES TROIS ZONES")
    print("=" * 65)

    print(
        f"Énergie E       = {E:.6e} J "
        f"= {E / 1.602176634e-19:.6g} eV"
    )

    print(
        f"Hauteur V0      = {V_0:.6e} J "
        f"= {V_0 / 1.602176634e-19:.6g} eV"
    )

    print(f"Largeur a       = {a:.6e} m")
    print(f"Régime          = {regime}")

    print("\nPulsation temporelle :")
    print(f"omega           = {omega:.6e} rad/s")

    print("\nZone 1 : avant la barrière, x < 0")
    print(f"k1              = {k1:.6e} m^-1")
    print(
        "psi1(x)         = "
        "A exp(i k1 x) + B exp(-i k1 x)"
    )

    print("\nZone 2 : dans la barrière, 0 <= x <= a")

    if E > V_0:
        print(f"k2              = {k2:.6e} m^-1")
        print(
            "psi2(x)         = "
            "C exp(i k2 x) + D exp(-i k2 x)"
        )

    elif E < V_0:
        print(f"k2              = i × {kappa:.6e} m^-1")
        print(f"kappa           = {kappa:.6e} m^-1")
        print(
            "psi2(x)         = "
            "C exp(kappa x) + D exp(-kappa x)"
        )

    else:
        print("k2              = 0")
        print("psi2(x)         = C + D x")

    print("\nZone 3 : après la barrière, x > a")
    print(f"k3              = {k3:.6e} m^-1")
    print("psi3(x)         = F exp(i k3 x)")

    print("\nAmplitudes complexes :")
    print(f"A incident       = {A:.6g}")
    print(f"B réfléchi       = {B:.6g}")
    print(f"C barrière       = {C_stationnaire:.6g}")
    print(f"D barrière       = {D:.6g}")
    print(f"F transmis       = {F:.6g}")

    print("\nCoefficients stationnaires :")
    print(f"R                = {R_stationnaire:.8f}")
    print(f"T                = {T_stationnaire:.8f}")
    print(f"R + T            = {R_stationnaire + T_stationnaire:.8f}")

    print("\nVérification numérique :")
    print(f"Résidu relatif   = {residu_relatif:.3e}")
    print("=" * 65)

    # ---------------------------------------------------------
    # Affichage graphique
    # ---------------------------------------------------------

    if afficher:
        densite_stationnaire = np.abs(psi_stationnaire)**2

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            sharex=True
        )

        # Partie réelle et imaginaire
        axes[0].plot(
            x,
            np.real(psi_stationnaire),
            label=r"$\Re(\psi_E)$"
        )

        axes[0].plot(
            x,
            np.imag(psi_stationnaire),
            label=r"$\Im(\psi_E)$"
        )

        axes[0].axvline(
            0,
            linestyle="--",
            label="Entrée de la barrière"
        )

        axes[0].axvline(
            a,
            linestyle="--",
            label="Sortie de la barrière"
        )

        axes[0].set_ylabel(r"$\psi_E(x)$")

        axes[0].set_title(
            "État stationnaire de diffusion dans les trois zones"
        )

        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Densité de probabilité
        axes[1].plot(
            x,
            densite_stationnaire,
            label=r"$|\psi_E(x)|^2$"
        )

        axes[1].axvspan(
            0,
            a,
            alpha=0.15,
            label="Barrière"
        )

        axes[1].set_xlabel("Position x (m)")
        axes[1].set_ylabel(r"$|\psi_E(x)|^2$")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return {
        "E": E,
        "omega": omega,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "kappa": kappa,
        "A": A,
        "B": B,
        "C": C_stationnaire,
        "D": D,
        "F": F,
        "R": R_stationnaire,
        "T": T_stationnaire,
        "psi_stationnaire": psi_stationnaire,
        "residu_relatif": residu_relatif,
        "regime": regime
    }


#--Fonction principales---------------------------

def main(an=False):

    # E = h*nu = h * c / lambda
    # E >> V_0 => E >> 10.0 >> lambda << c*h / 10.0 => lambda ~ 

    while True:
        print("Type de simulation : (entrez 1 ou 2)")
        print("-" * 20)
        print("1. Simulation maison")
        print("2. Simulation optimisée")
        print("3. générer pour une liste de V_0 (entre 0.5 et 1.5 eV)")
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
        
    borne = 40e-9
    x = np.linspace(-np.abs(borne),np.abs(borne),nx)
    dx = x[1] - x[0]
    lambda_0 = 2 * np.pi / k_0

    print(f"dx = {dx}")
    print(f"{lambda_0=}")
    print(f"Nombre de points par longueur d'onde = {lambda_0 / dx}")
    print(f"k_0 * dx = {k_0 * dx}")

    V_0 = 0.8 * 1.602e-19
    # on peut prendre V_0 = 20.0 et a = 1.0 ca marche bien
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= a_barriere)] = V_0

    # -------------------------------------------------------------
    # CALCUL NUMÉRIQUE DES ÉTATS STATIONNAIRES
    # -------------------------------------------------------------

    resultat_stationnaire = etats_stationnaires_barriere(
        x=x,
        V_0=V_0,
        a=a_barriere,
        E=None,
        afficher=True
    )

    # Valeurs calculées accessibles séparément
    E_stationnaire = resultat_stationnaire["E"]
    omega_stationnaire = resultat_stationnaire["omega"]

    k_zone_1 = resultat_stationnaire["k1"]
    k_zone_2 = resultat_stationnaire["k2"]
    k_zone_3 = resultat_stationnaire["k3"]

    psi_stationnaire = resultat_stationnaire["psi_stationnaire"]

    print("\nValeurs récupérées dans main :")
    print(f"E     = {E_stationnaire:.6e} J")
    print(f"omega = {omega_stationnaire:.6e} rad/s")
    print(f"k1    = {k_zone_1:.6e} m^-1")
    print(f"k2    = {k_zone_2:.6e} m^-1")
    print(f"k3    = {k_zone_3:.6e} m^-1")

    if reponse == 1 : 
        t_max = 0.2
        t = np.linspace(0,t_max,nt) 
        f, t_passage = propagation_onde(x,t,V)
        tolerance = 1e-02

    elif reponse == 2 : 
        t_max = 60e-15
        t = np.linspace(0,t_max,nt) 
        f, t_passage = propagation_onde_CN(x,t,V,x_0=x_0) #propagation_onde(x,t,V)
        tolerance = 1e-5
    
    elif reponse == 3:
        t_max = 60e-15
        t = np.linspace(0,t_max,nt) 
        générer_T(x,t)
        exit(0)

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
        afficher_graphes=True if an else False
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

    # --- CALCUL THÉORIQUE UNIVERSEL ---
    vg = (H_BARRE * k_0) / M
    E = (H_BARRE**2 * k_0**2) / (2*M) + (H_BARRE**2) / (2*M*lg_initiale**2)  # Énergie moyenne
    
    t_avant_barriere = (0 - x_0) / vg  # Trajet de x_0 à 0
    t_apres_barriere = 3e-9 / vg        # Trajet après la barrière (si ton capteur est décalé de 2m)

    if E > V_0:
        k_barriere = np.sqrt(k_0**2 - 2*M*V_0/H_BARRE**2)
        v_barriere = (H_BARRE * k_barriere) / M
        t_dans_barriere = a_barriere / v_barriere
        print(f"Régime classique - v dans la barrière : {v_barriere:.4g} m/s")
    else:
        kappa = np.sqrt(2*M*(V_0 - E) / H_BARRE**2)
        t_dans_barriere = (2 * M) / (H_BARRE * k_0 * kappa)
        print(f"Régime Tunnel - kappa : {kappa:.4g} m^-1") #vu avec IA pour explication de t_passage None pour des valeurs de V_0 trop grande (> E)

    temps_theorique_total = t_avant_barriere + t_dans_barriere + t_apres_barriere
    
    print(f"Temps théorique total : {temps_theorique_total:.4g} s")
    print(f"Temps numérique mesuré : {t_passage:.4g} s")
    
    if t_passage is not None and not np.isnan(t_passage):
        print(f"Erreur : {erreur(t_passage, temps_theorique_total):g}%")

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