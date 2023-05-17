import matplotlib.pyplot as plt
import numpy as np

from conf import PATH
from energy_gym import get_feed

def reg_lin_water_law(ts, nb_dep, nb_ret, nb_pump, **params):
    """
    régression linéaire sur température de retour et de départ
    nb_dep : numéro de flux température départ
    nb_ret : numéro de flux température retour
    nb_pump : numéro de flux pompe (O/N)
    """
    interval = params.get("interval", 3600)
    view_plot = params.get("view_plot", True)
    wsize = 8 * 24 * 3600 // interval

    def window_feed(nb_feed, ts, verbose=False):

        t_mes = get_feed(nb_feed, interval, path=f'{PATH}/mesures')
        if verbose:
            print(t_mes.start)
        tse_mes = t_mes.start + t_mes.shape[0] * t_mes.step
        if verbose:
            print(tse_mes)
        pos = (ts - t_mes.start) // interval
        return t_mes[pos:pos+wsize+1]

    t_eau_dep_mes = window_feed(nb_dep, ts)
    t_eau_ret_mes = window_feed(nb_ret, ts)
    pump = window_feed(nb_pump, ts)

    # régression linéaire
    coeffs = np.polyfit(t_eau_dep_mes[pump[:]>0], t_eau_ret_mes[pump[:]>0], 1)
    t_eau_ret_mes_affine = coeffs[0] * t_eau_dep_mes + coeffs[1]

    if view_plot:
        label = f'T_eau_retour={coeffs[0]:.2f} * T_eau_départ + {coeffs[1]:.2f}'
        plt.figure(figsize=(20,10))
        ax1 = plt.subplot(211)
        plt.title("températures d'eau chaude mesurées")
        plt.plot(t_eau_dep_mes, color="red", label="T_départ_eau °C")
        plt.plot(t_eau_ret_mes, color="purple", label="T_retour_eau °C")
        plt.legend()
        ax1.twinx()
        plt.plot(pump, label="pompe ON/OFF")
        plt.legend()
        ax3 = plt.subplot(212)
        ax3.set_ylabel("T eau retour °C")
        ax3.set_xlabel("T eau départ °C")
        plt.plot(t_eau_dep_mes[pump>0], t_eau_ret_mes[pump>0], '*')
        plt.plot(t_eau_dep_mes, t_eau_ret_mes_affine, label=label)
        plt.legend()
        plt.show()

    return coeffs[0], coeffs[1]
