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
while True:
   action = bat.action_space.sample()
   _, _, done, _ = bat.step(action)
   if done:
       break
bat.render(stepbystep=False)
```
