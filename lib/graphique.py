import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.constantes import *
from lib.math_physique import calculer_RT

# Note : Pense à passer k_0 et lg_initiale en paramètres pour que la fonction reste autonome
def animation(x : np.array, matrice_f : np.array, t : np.array, V : np.array, V_0 : float) -> None:
    """Créer une animation sur deux sous-graphiques :
    - Écran du haut : Évolution de |\Psi|² et barrière de potentiel
    - Écran du bas : Énergie moyenne <E> face au profil du potentiel réel V(x)
    """

    # On crée 2 sous-graphiques superposés
    fig, (ax_wave, ax_energy) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # -------------------------------------------------------------
    # GRAPHIQUE 1 : DENSITÉ DE PROBABILITÉ (Haut)
    # -------------------------------------------------------------
    zeros_init = np.zeros_like(x)
    line_prob, = ax_wave.plot(x, zeros_init, color="green", linewidth=2.5, label=r"Densité de probabilité $|\Psi(x,t)|^2$")
    
    # Configuration des axes (Haut)
    ax_wave.set_xlim(x[0], x[-1])
    ax_wave.set_ylabel("Densité de probabilité", fontsize=11)
    ax_wave.grid(True, linestyle="--", alpha=0.5)
    
    ymax_wave = np.max(np.abs(matrice_f)**2) * 1.3
    ax_wave.set_ylim(-0.05 * ymax_wave, ymax_wave)
    
    # On dessine la barrière de potentiel en fond de manière subtile (juste une ombre)
    ax_wave.fill_between(x, 0, ymax_wave, where=(V > 0), color='red', alpha=0.1, label='Zone de la barrière')
    ax_wave.legend(loc="upper right", fontsize=9)

    # -------------------------------------------------------------
    # GRAPHIQUE 2 : DIAGRAMME ÉNERGÉTIQUE (Bas)
    # -------------------------------------------------------------
    # Calcul de l'énergie moyenne exacte
    energie_moyenne = (H_BARRE**2 * k_0**2) / (2 * M) + (H_BARRE**2) / (2 * M * (lg_initiale**2))
    
    # On trace le vrai profil de potentiel V(x) à l'échelle réelle de l'énergie
    ax_energy.plot(x, V, color='red', linestyle='-', linewidth=2, label=f'Potentiel $V(x)$ ($V_0 = {V_0}$)')
    ax_energy.fill_between(x, 0, V, color='red', alpha=0.15)
    
    # On trace la ligne d'énergie moyenne
    ax_energy.axhline(y=energie_moyenne, color="blue", linestyle="--", linewidth=2, 
                       label=rf"Énergie moyenne $\langle E \rangle = {energie_moyenne:.2g}$")
    
    # Calcul du rapport E / V_0 (attention si V_0 vaut 0)
    rapport = energie_moyenne / V_0 if V_0 != 0 else np.inf
    
    # Configuration des axes (Bas)
    ax_energy.set_xlabel("Position $x$ [m]", fontsize=11)
    ax_energy.set_ylabel("Énergie [J]", fontsize=11)
    ax_energy.grid(True, linestyle="--", alpha=0.5)
    
    ymax_energy = max(V_0, energie_moyenne) * 1.3
    ax_energy.set_ylim(-0.05 * ymax_energy, ymax_energy)
    ax_energy.legend(loc="upper right", fontsize=9)

    # Petit texte d'information sur le régime quantique dans le graphique du bas
    texte_regime = r"Régime : Sur-barrière ($\langle E \rangle > V_0$)" if energie_moyenne > V_0 else r"Régime : Effet Tunnel ($\langle E \rangle < V_0$)"
    ax_energy.text(0.02, 0.08, rf"{texte_regime} \n Rapport $\langle E \rangle / V_0 = {rapport:.2f}$", 
                   transform=ax_energy.transAxes, fontsize=10, fontweight="bold", 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    R,T = calculer_RT(x,matrice_f,True,True)
    ax_energy.text(0.02,0.2, rf"Réflexion : {R*100:.2g}% | Transmission : {T*100:.2g}%",transform=ax_energy.transAxes, fontsize=10, fontweight="bold", 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # -------------------------------------------------------------
    # MISE À JOUR DE L'ANIMATION
    # -------------------------------------------------------------
    def update(j):
        psi_actuel = matrice_f[j, :]
        densite_proba = np.abs(psi_actuel)**2
        
        # On met à jour uniquement la courbe de probabilité (le reste est fixe)
        line_prob.set_data(x, densite_proba)
        
        # Le grand titre global se met à jour avec le temps
        fig.suptitle(f"Simulation Quantique - t = {t[j]:.2f} s", fontsize=13, fontweight="bold")
        return line_prob,

    # Optimisation du nombre de frames pour éviter de saturer la RAM (1 frame sur 20)
    step = max(1, len(t) // 200)
    frames_a_afficher = range(0, len(t), step)

    ani = FuncAnimation(fig, update, frames=frames_a_afficher, interval=60, blit=True)
    
    plt.tight_layout()
    plt.show()
