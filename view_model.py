"""ouvre un réseau neurones et en affiche le détail
notamment le nombre de neurones par couches
"""
import sys
from energy_gym import pick_name
from conf import load

agent_path, saved = pick_name()
if not saved :
    sys.exit(0)
agent = load(agent_path)
agent.summary()
