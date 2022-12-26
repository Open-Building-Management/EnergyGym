# CHANGES FROM VERSION 0.1

## paramètres tc et halfrange de basicplay.py & standalone_d_dqn.py

possibilité d'entraîner à consigne variable, autour d'une température de consigne moyenne

tc est la valeur de la consigne moyenne en °C

si halfrange= 0, la consigne est fixe d'un épisode sur l'autre

si halfrange=2 et tc=20, on pourra avoir comme consignes possibles 18, 19, 20, 21 et 22 °C

## environnement Hyst

permet d'entraîner un réseau à reproduire un comportement hystérésis

## environnement Reduce

permet de faire jouer à un hystérésis un réduit d'inoccupation

cf basicplay.py

constante REDUCE = hauteur du réduit en °C à soustraire à la consigne moyenne

## précision de l'espace d'action

espace d'action toujours discret mais plus limité à 2 valeurs

dans la version précédente, on était limité à des actions binaires :
- soit on chauffait à fond
- soit on coupait totalement le chauffage

on peut désormais fixer la taille de l'espace d'actions par exemple à 11,
ce qui permet de mobiliser 0%, 10%, 20%...90% ou 100% de la puissance maxi dispo

```
python3 standalone_d_dqn.py --action_space=11
python3 basicplay.py --action_space=11
```

# fonctions sim et play_hystnvacancy de evaluation_toolbox.py

sim permet de calculer la température intérieure dans nbh heures :
- soit en chauffant en continu
- soit sans chauffer

si on veut prévoir le point suivant seulement,
on doit donner à nbh la valeur env.text.step/3600

play_hystnvacancy joue la politique optimale sur un scénario d'intermittence


# introduction des paramètres nbh et nbh_forecast dans l'environnement

l'espace d'observation peut désormais intégrer :
- nbh heures d'histoire passée
- nbh_forecast de prévisions météo

lorsqu'on entraîne avec nbh > 0, on donne au réseau neurones des informations sur le modèle,
ce qui permet d'envisager d'entraîner à modèle variable d'un épisode sur l'autre

```
python3 standalone_d_dqn.py --nbh_forecast=24 --action_space=11
python3 basicplay.py --nbh_forecast=24 --action_space=11
```
