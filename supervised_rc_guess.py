"""guess the electrical parameters from the history

Apprentissage supervisé utilisant comme étiquettes
les paramètres électriques du modèle de l'épisode

On utilise la banque de modèles pour générer des épisodes.

Pour chaque épisode, on tire aléatoirement un modèle et on joue
un certain nombre d'heures en chauffant aléatoirement,
ce qu'on ne peut pas faire sur un vrai bâtiment sauf s'il est vide.

On enregitre alors (text, tint, qc) dans une matrice 2D qu'on peut envoyer
à un réseau LSTM
"""
import sys
import argparse
#import random
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import energy_gym
from energy_gym import get_feed, set_extra_params, load, pick_name
from conf import MODELS
from conf import PATH, MAX_POWER, TEXT_FEED
import conf


# ne pas changer ce pas de temps
INTERVAL = 3600
# nombre d'heures d'histoire passée pour le réseau LSTM
NBH = 48
# caractéristiques des entrainements
BATCH_SIZE = 50
MAX_EPOCHS = 50
STEPS_PER_EPOCH = 50
# poids pour normaliser les paramètres électriques qui servent de labels
P_R = 1e4
P_C = 1e-8


class BatchGenerator:  # pylint: disable=R0903
    """generateur de batches"""
    def __init__(self, env, size):
        """
        env : environnement gym

        size : nombre d'heures de chaque épisode
        """
        self.env = env
        self.size = size

    def generate(self):
        """generate episodes/labels"""
        while True:
            x = np.zeros((BATCH_SIZE, self.size, 3))
            y = np.zeros((BATCH_SIZE, 2))
            for i in range(BATCH_SIZE):
                newmodel = conf.generate(
                    verbose=False,
                    bank_name="synth",
                    rc_min=50,
                    rc_max=100
                )
                self.env.update_model(newmodel)
                self.env.reset()
                for _ in range(self.size):
                    action = self.env.action_space.sample()
                    self.env.step(action)
                x[i, :] = np.array([
                    self.env.text[self.env.pos:self.env.pos+self.size],
                    self.env.tint[0:self.size],
                    self.env.action[0:self.size] * self.env.max_power
                ]).transpose()
                y[i, :] = np.array([
                    self.env.model["R"] * self.env.model["C"] / 3600
                ])
            #print(x.shape)
            #print(x,y)
            yield x, y


def train(env):
    """train the lstm agent"""
    agent = keras.models.Sequential()
    agent.add(keras.layers.LSTM(512))
    agent.add(keras.layers.Dense(1))
    agent.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    train_data_generator = BatchGenerator(env, NBH)
    agent.fit(train_data_generator.generate(), steps_per_epoch=STEPS_PER_EPOCH, epochs=MAX_EPOCHS)

    save = input("save ? Y=yes")
    if save == "Y":
        agent.save("LSTM")


def play(env):
    """play"""
    agent_path, saved = pick_name()
    if not saved :
        sys.exit(0)
    agent = load(agent_path)
    models = {}
    models = MODELS
    models["never_met"] = {"R": 2.5e-3, "C": 8.7e8}
    test_data_generator = BatchGenerator(env, NBH)
    data = next(test_data_generator.generate())
    prediction = agent(data[0]).numpy()
    for j in range(data[0].shape[0]):
        plt.subplot(311)
        original = {"R": data[1][j][0] / P_R, "C": data[1][j][1] / P_C}
        guess = {"R": prediction[j][0] / P_R, "C": prediction[j][1] / P_C}
        label = f'R={original["R"]:.2e} C={original["C"]:.2e}'
        label = f'{label}\n R={guess["R"]:.2e} C={guess["C"]:.2e}'
        plt.title(label)
        # on joue les vrais R et C
        env.update_model(original)
        env.reset()
        while True:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                break
        plt.plot(env.tint, color="orange", label="truth")
        env.update_model(guess)
        # on enregistre les actions jouées
        actions = env.action
        # on joue les R et C devinés par le modèle
        # avec la même condition initiale en température
        tint0 = env.tint[0] # ceci fonctionne car env.nbh=0
        env.reset(ts=env.tsvrai, tint=tint0)
        while True:
            action = int(actions[env.i])
            _, _, done, _ = env.step(action)
            if done:
                break
        plt.plot(env.tint, color="blue", label="guess")
        plt.subplot(312)
        plt.plot(actions, color="orange")
        plt.plot(env.action, color="blue")
        plt.subplot(313)
        plt.plot(env.text[env.pos:env.pos+env.wsize])
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_space', type=int, default=2, help="taille de l'espace d'actions")
    parser.add_argument('--mode', type=str, default="play", help="train or play ?")
    args = parser.parse_args()
    action_space = args.action_space
    mode = args.mode

    text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
    model = MODELS["cells"]
    model = set_extra_params(model, action_space=action_space)
    model = set_extra_params(model, autosize_max_power=True)
    # on part sur un environnement Vacancy mais peu importe
    # on a fixé la température de consigne à 20 mais c'est factice et ne sert pas
    bat = getattr(energy_gym, "Vacancy")(text, MAX_POWER, 20, **model)
    if mode == "train":
        train(bat)
    else:
        play(bat)
