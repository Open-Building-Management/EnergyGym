# EnergyGym
gym environnement to simulate the energetic behaviour of a tertiairy building

## requirements

```
sudo apt-get install -y python3-numpy python3-matplotlib
pip3 install click
pip3 install gym
pip3 install PyFina
pip3 install --upgrade tensorflow
```
pour utiliser tensorboard :

```
tensorboard --logdir=TensorBoard

```
<details id=1>
  <summary><h2>A propos du modèle d'environnement</h2></summary>
  
  L'environnement est représenté sous la forme d'un modèle électrique équivalent simple à deux paramètres : 
  - une résistance R en K/W qui représente l'isolation du bâtiment
  - une capacité C en J/K qui représente l'inertie du bâtiment 
  
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
  
