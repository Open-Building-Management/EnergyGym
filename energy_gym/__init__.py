"""energy_gym module

.. include:: ./README.md
"""
from .heatgym import Hyst, Reduce, Vacancy, StepRewardVacancy, TopLimitVacancy
from .heatgym import D2Vacancy, Building
from .heatgym import custom_gym_envs, vars_to_exclude_from_pdoc
from .evaluation_toolbox import sim, play_hystnvacancy
from .evaluation_toolbox import EvaluateGym
from .tools import set_extra_params, load, freeze
from .tools import pick_name, get_feed
from .planning import biosAgenda

__pdoc__ = {}
for cge in custom_gym_envs:
    for excluded in vars_to_exclude_from_pdoc:
        __pdoc__[f'{cge}.{excluded}'] = False


__all__ = [
    *custom_gym_envs,
    "sim",
    "play_hystnvacancy",
    "EvaluateGym",
    "set_extra_params",
    "load",
    "freeze",
    "pick_name",
    "get_feed",
    "biosAgenda"
]
