import matplotlib.pyplot as plt
import numpy as np
import math

from energy_gym.planning import get_random_start

def play_baseline(text, debit, t_c, **params):
    """joue un scénario de type loi d'eau"""
    ts = params.get("ts", None)
    _r_ = params.get("r", 1e-3)
    _c_ = params.get("c", 2e+8)
    interval = params.get("interval", 3600)
    wsize = 8 * 24 * 3600 // interval
    # Capacité calorifique de l'eau en Wh/m3/K
    CW = 1162.5

    if ts is None:
        start = text.start
        tse = text.start + text.step * text.shape[0]
        end = tse - wsize * interval - 4*24*3600
        ts = get_random_start(start, end, 10, 5)

    pos = (ts - text.start) // interval
    tsvrai = text.start + pos * interval
    # time for human !
    xrs = np.arange(tsvrai, tsvrai + (wsize+1)*interval, interval)
    xr = np.array(xrs, dtype='datetime64[s]')

    T_ext_week = text[pos:pos+wsize+1]

    tcte = _r_ * _c_
    cte = math.exp(-text.step/tcte)

    # puissance pour le maintien du confort intérieur à tc
    Qc = (t_c - T_ext_week) / _r_
    for i in range(Qc.shape[0]):
        if Qc[i] < 0:
            Qc[i] = 0

    # simulation de la température intérieure avec le modèle
    T_int = np.zeros(T_ext_week.shape[0])
    T_int[0] = t_c
    for j in range(1, T_int.shape[0]):
        #delta = cte * (pmtc[j-1] / _c_ + T_ext_week[j-1] / tcte)
        #delta += pmtc[j] / _c_ + T_ext_week[j] / tcte
        #T_int[j] = T_int[j-1] * cte + text.step * 0.5 * delta
        T_int[j] = T_int[j-1] * cte + text.step * 0.5 * t_c * (1 + cte) / tcte

    # températures de départ et de retour
    T_eau_dep = 1.5 * (t_c - T_ext_week) + t_c
    T_eau_ret = T_eau_dep - Qc / (CW * debit)

    # régression linéaire
    coeffs = np.polyfit(T_eau_dep[Qc[:]>0], T_eau_ret[Qc[:]>0], 1)
    T_eau_ret_affine = coeffs[0] * T_eau_dep + coeffs[1]

    label_eau = f'T_eau_retour={coeffs[0]:.2f} * T_eau_départ + {coeffs[1]:.2f}'

    title = f'BASELINE ENERGETIQUE pour un débit de {debit} m3/h'
    title = f'{title}et pour R = {_r_:.2e} K/W'

    label_power = "puissance en W pour maintien à 20 °C"

    print(ts)
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(311)
    plt.title(title)
    plt.plot(xr, T_ext_week, label="Text °C")
    plt.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.set_ylim([t_c-1, t_c+1])
    plt.plot(xr, T_int, label="Tint °C", color="green")
    plt.legend()

    ax3 = plt.subplot(312)
    plt.fill_between(xr, 0, Qc, color="#fff2c1", label=label_power)
    plt.legend(loc="upper left")
    ax4 = ax3.twinx()
    plt.plot(xr, T_eau_dep, label="T_départ_eau °C", color="red")
    plt.plot(xr, T_eau_ret, label="T_retour_eau °C", color="purple")
    plt.legend()
    ax = plt.subplot(313)
    ax.set_ylabel("T eau retour °C")
    ax.set_xlabel("T eau départ °C")
    plt.plot(T_eau_dep, T_eau_ret, '*', label=label_eau)
    plt.plot(T_eau_dep, T_eau_ret_affine)
    plt.legend()
    plt.show()
