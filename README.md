# EnergyGym
environnement gym [openAI](https://github.com/openai/gym) pour simuler le comportement énergétique d'un bâtiment tertiaire

[le_jeu_du_chauffage_en_mode_random.webm](images/le_jeu_du_chauffage_en_mode_random.webm)

## requirements

```
sudo apt-get install -y python3-numpy python3-matplotlib
pip3 install click
pip3 install gym
pip3 install PyFina
pip3 install --upgrade tensorflow
```

play et basicplay utilisent l'autocomplétion en ligne de commande pour choisir le nom de l'agent

**Seul basicplay utilise l'environnement gym.**

## basicplay

```
python3 basicplay.py
```

paramètre |  description
--|--
agent_type | random = décision aléatoire<br>deterministic = argmax<br>stochastic = softmax
random_ts | True = joue jusqu'à 200 épisodes<br>False = joue un seul épisode sur le timestamp 1609104740
mode | vacancy = joue des périodes de non-occupation<br>week = joue une semaine type
model | le nom d'une des configurations de [conf.py](conf.py)
stepbystep | True = joue en mode pas à pas

## play

possibilité :
* de faire jouer simultanément l'agent et la politique optimale de l'environnement,
* de produire des statistiques

```
./play.py play
```
paramètre |  description
--|--
t_ext | numéro du flux de température extérieure = 1
model | le nom d'une des configurations de [conf.py](conf.py)
powerlimit | coefficient multiplicatif de la puissance max.
tc | température de consigne
n | **nombre d'épisodes à jouer**<br>0 = joue une série d'épisodes prédéfinis, on parle de snapshots
optimalpolicy | **politique optimale que l'environnement déterministe va jouer**<br>intermittence = succession de périodes d'occupation et de non-occupation<br>occupation_permanente = bâtiment occupé en permanence - cf hopital
hystpath | nom d'un agent de type hystérésis, à fournir si on veut utiliser un agent pour gérer les périodes de non-occupation et un hystéréris pour gérer les périodes de présence du personnel : `./play.py --hystpath=agents/hys20.h5 play`
holiday | nombre de jours fériés à intégrer dans les replay
silent | True = production de statistiques ou de snapshots<br>False = affiche les épisodes à l'écran 
k | coefficient énergétique, utilisé dans le calcul de la récompense

<details id=1>
  <summary><h2>A propos du modèle d'environnement</h2></summary>
  
  L'environnement est représenté sous la forme d'un modèle électrique équivalent simple à deux paramètres : 
  * une résistance R en K/W qui représente l'isolation du bâtiment
  * une capacité C en J/K qui représente l'inertie du bâtiment 
  
  [Pour en savoir plus](https://github.com/Open-Building-Management/RCmodel/blob/main/RCmodel.ipynb)
  
  Pour une résistance de 1e-4 K/W, et quelle que soit l’inertie entre 4e8 et 4e9 J/K, le système de chauffage, même utilisé à fond en permanence, ne 
  parvient pas à maintenir la température. 
  
  Pour pouvoir gérer des épisodes de froid sur des bâtiments présentant majoritairement des résistances inférieures à 2e-4 K/W, la seule solution est 
  d’augmenter la puissance disponible. 
  
  On ne devrait toutefois pas rencontrer ce cas de figure sur le terrain si les équipements de production et les pompes sont correctement dimensionnés. 
  
  Le couple R=2e-4 K/W et C=2e8 J/K semble donc être une configuration extrême, peu probable en pratique, mais susceptible de nous donner de la matière 
  pour bien cerner le fonctionnement de notre modèle.
  
  ### comportement sous météo hivernale froide
  ![](images/RC_sim2_48h.png)

</details>
