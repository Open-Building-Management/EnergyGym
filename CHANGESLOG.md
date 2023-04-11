# CHANGES FROM VERSION 0.2


Amélioration de la pertinence des récompenses pour Vacancy et StepRewardVacancy et modulation des paramètres de la récompense désormais possible via la ligne de commande

Module de statistiques compatible avec les environnements gym, capable de jouer sur un parc de bâtiments, mais aussi un couple R/C bien défini et un timestamp particulier cf [stats.py](stats.py)

Implémentation de D2Vacancy pour retourner des states prenant la forme de matrices 2D (axe 0 : le temps, axe 1 : les paramètres)

Ajout de [supervised_rc_guess.py](supervised_rc_guess.py) pour l'apprentissage supervisé des paramètres RC par un réseau LSTM

Ajout d'un générateur de modèles et d'un mode autosize_max_power pour dimensionner la puissance maximale disponible en fonction de l'isolation
Ce générateur est dans [conf.py](conf.py)

Possibilité d'ajouter un threshold min et/ou max sur la température extérieure, par exemple pour n'entrainer que sur des températures clémentes, ou au contraire que sur une sélection des plus froide **[n'a pas donné grand chose]**

Ajout de 2 algorithmes :
- l'un introduisant l'architecture dueling
- l'autre introduisant une mémoire plus intelligente (per : prioritized experience replay), capable de donner plus d'importance lors des entrainement aux expériences qui ont le plus de valeur ajoutée et grâce auxquelles le réseau peut apprendre plus efficacement **[n'a pas donné grand chose pour l'instant]** algorithme probalement à stabiliser. Utilise un sumtree, la feuille i ayant comme valeur la probabilité de la transition i - cf [shared_rl_tools.py](shared_rl_tools.py)

Ajout de 2 méthodes show_episode_stats et add_scalars_to_tensorboard dans [standalone_d_dqn.py](standalone_d_dqn.py) : A UTILISER DANS TOUS LES ALGORITHMES!

Ajout de [view_tensorboard_graph.py](view_tensorboard_graph.py) pour visualiser l'architecture dueling sous tensorboard

Ajout dans la bibliothèque des méthodes :
- set_extra_params, pour ajouter des paramètres au dictionnaire du modèle d'environnement
- load, pour charger un réseau, cette méthode étant auparavant dans basicplay.py
- freeze, pour générer des jours fériés à intégrer à un agenda


