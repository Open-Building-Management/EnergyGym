from energy_gym import pick_name
from basicplay import load


agent_path, saved = pick_name()
if not saved :
    sys.exit(0)
agent = load(agent_path)
agent.summary()