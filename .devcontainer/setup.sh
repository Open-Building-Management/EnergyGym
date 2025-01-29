#!/usr/bin/env bash
# setup for the devcontainer
apt-get update
apt-get install -y python3-tk
python3 -m pip install tf-keras==$TF_VERSION
python3 -m pip install matplotlib
python3 -m pip install click
python3 -m pip install gym
python3 -m pip install PyFina