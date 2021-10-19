#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:59:35 2021

@author: cghiaus

Figures for the paper
C. Ghiaus (2021) Computational psychrometric analysis of cooling systems as
a control problem: case of cooling and dehumidification systems,
International Journal of Building Performance Simulation,
DOI: 10.1080/19401493.2021.1995498
"""
import ipywidgets as wd
import matplotlib.pyplot as plt
import cool as cc

# %matplotlib inline  # uncomment for inline figure
# %matplotlib qt      # uncomment for figure in separate window
plt.rcParams["figure.figsize"] = (10, 7.7)
font = {'size': 16}
plt.rc('font', **font)

Kθ, Kw = 1e10, 0        # Gain factors of the P-controllers
β = 0.16                # By-pass factor of the cooling coil

m, mo = 3.1, 1.         # kg/s, mass flow rate, supply and outdoor air
θo, φo = 32., 0.5       # °C, -, outdoor air temperature and relative humidity
θ5sp, φ5sp = 26., 0.5   # °C, -, indoor air set points

mi = 1.35               # kg/s, mass flow rate of infiltration air
UA = 675.               # W/K, overall heat transfet coeffcient
QsBL, QlBL = 34000., 4000.    # W, sensible & latent auxiliar heat

parameters = m, mo, β, Kθ, Kw
inputs = θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL


def no_06():
    cool0 = cc.MxCcRhTzBl(parameters, inputs)
    Kw = 1e10
    cool0.actual[4] = Kw
    wd.interact(cool0.CAV_wd, θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φ5sp=(0.30, 1, 0.01),
                mi=(0.1, 3, 0.1), UA=(500, 800, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("CAV systems with reheating")
    print("Control of indoor air temperature & humidity (θ5, φ5)")


def no_08():
    cool1 = cc.MxCcRhTzBl(parameters, inputs)
    wd.interact(cool1.CAV_wd, θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φI5sp=(0.4, 1, 0.01),
                mi=(0.5, 3, 0.1), UA=(500, 8000, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("CAV systems without reheating")
    print("Control of indoor air temperature (θ5)")


def no_10():
    cool4 = cc.MxCcRhTzBl(parameters, inputs)
    wd.interact(cool4.VBP_wd, value='φ5', sp=(0.3, 0.5, 0.01),
                θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φ5sp=(0.4, 0.8, 0.01),
                mi=(0.5, 3, 0.1), UA=(500, 800, 10),
                Qsa=(0, 60_000, 500), Qla=(0, 20_000, 500))
    print("CAV systems")
    print("Control of indoor air temperature & humidity (θ5, φ5)")


def no_12():
    cool3 = cc.MxCcRhTzBl(parameters, inputs)
    wd.interact(cool3.VAV_wd, value='φ5', sp=(0.4, 0.5, 0.05),
                θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φ5sp=(0.4, 0.8),
                mi=(0.7, 3, 0.1), UA=(500, 800, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("VAV systems without reheating")
    print("Control of indoor air temperature & humidity (θ5, φ5)")


def no_14():
    cool6 = cc.MxCcRhTzBl(parameters, inputs)
    Kw = 0
    cool6.actual[4] = Kw
    wd.interact(cool6.VAV_wd, value='θ4', sp=(14, 21),
                θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φ5sp=(0.4, 0.8),
                mi=(0.5, 3, 0.1), UA=(500, 800, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("VAV systems without reheating")
    print("Control of indoor and supply air temperatures (θ4, θ5)")


def no_16():
    cool6 = cc.MxCcRhTzBl(parameters, inputs)
    Kw = 1e10
    cool6.actual[4] = Kw
    wd.interact(cool6.VAV_wd, value='θ4', sp=(14, 21),
                θo=(28, 36), φo=(0.4, 1),
                θ5sp=(22, 26), φ5sp=(0.4, 0.8),
                mi=(0.5, 3, 0.1), UA=(500, 800, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("VAV systems with reheating")
    print("Control of indoor air temperature & humidity and\
          of supply air temperature (θ4, θ5, φ5)")
