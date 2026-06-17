import numpy as np
#--Variables --------------------------
M = 9.109e-31
H_BARRE = 1.054e-34 # on utilise celle la pour pas faire péter l'ordi
##--Paramètres de l'onde gaussienne----
nx = 2000           # nombre de points dans l'espace
nt = 5000           # nombre de points dans le temps
E = 0.5 * 1.602e-19
k_0 = np.sqrt(2*M*E) / H_BARRE           # nombre d'ondes dans la gaussienne
lg_initiale = 2e-9     # largeur du paquet d'onde
t_0 = 0             # temps initial [T]
distance = 2        # distance que l'onde doit parcourir [L]
a_barriere = 0.5e-9 # 
x_0 = -10e-9

C : float = 2.99792458e8 # vitesse de la lumière dans le vide [m.s^-1]

# attention le maillage a une incidence sur les résultats obtenu