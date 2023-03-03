```
import energy_gym
from energy_gym import get_feed
from conf import MODELS
from conf import PATH, MAX_POWER, TEXT_FEED
INTERVAL = 1800
text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
model = MODELS["cells"]
bat = getattr(energy_gym, "Vacancy")(text, MAX_POWER, 20, **model)
while True:
   action = bat.action_space.sample()
   _, _, done, _ = bat.step(action)
   if done:
       break
bat.render(stepbystep=False)
```
