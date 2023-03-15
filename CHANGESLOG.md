# CHANGES FROM VERSION 0.2

Modulation des paramètres de la récompense désormais possible via la ligne de commande, pour les environnements dédiés à la gestion des périodes de non-occupation (Vacancy & al)

Module de statistiques compatible avec les environnements gym

Implémentation de D2Vacancy pour retourner des states prenant la forme de matrices 2D (axe 0 : le temps, axe 1 : les paramètres)

Ajout de supervised_rc_guess.py pour l'apprentissage supervisé des paramètres RC par un réseau LSTM

Ajout d'un générateur de modèles et d'un mode autosize_max_power pour dimensionner la puissance maximale disponible en fonction de l'isolation
