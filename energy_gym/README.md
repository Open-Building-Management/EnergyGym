# Basic usage
```
import energy_gym
from energy_gym import get_feed
INTERVAL = 1800
PATH = "datas"
# water calorific capacity
CW = 1162.5 #Wh/m3/K
# flow rate : 5m3/h
# temperature delta between injection and return : 15°C
MAX_POWER = 5 * CW * 15
TEXT_FEED = 1

text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
model = {"R" : 2.59460660e-04, "C" : 1.31446233e+09}
bat = getattr(energy_gym, "Vacancy")(text, MAX_POWER, 20, **model)
bat.reset()
while True:
   action = bat.action_space.sample()
   _, _, done, _ = bat.step(action)
   if done:
       break
bat.render(stepbystep=False)
```
# Advanced usage

```
import random
import energy_gym
from energy_gym import get_feed
INTERVAL = 1800
PATH = "datas"
TEXT_FEED = 1
text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
# create a random circuit with some inertia
while True:
    _R_ = random.randint(1, 9) * random.choice([1e-3, 1e-4])
    _C_ = random.randint(1, 9) * random.choice([1e+7, 1e+8, 1e+9])
    if 50 <= _R_ * _C_ / 3600 <= 100:
        break
model = {"R" : _R_, "C" : _C_,
         "k" : 1, "k_step": 1, "vote_interval": (-1, 1), "p_c": 15,
         "autosize_max_power": True}

bat = getattr(energy_gym, "Vacancy")(text, None, 20, **model)
bat.reset()
while True:
   action = bat.action_space.sample()
   _, _, done, _ = bat.step(action)
   if done:
       break
label = f'R={bat.model["R"]:.2e} C={bat.model["C"]:.2e}'
max_power = round(bat.max_power * 1e-3)
label = f'{label} MAX_POWER={max_power}kW'
bat.render(stepbystep=False, label=label)
```
