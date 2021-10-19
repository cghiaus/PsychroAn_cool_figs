#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:50:35 2021

@author: cghiaus
Created on Thu Mar 25 07:14:57 2021

mo  outdoor mass flow rate as an input.


Cooling as a control & parameter optimizaton problem
OOP implementation
Recycling, Cooling & desumidification (with by-pass), reheating

GENERALITIES
========================================================================
Nomenclature
    θ, Temperatures: °C
    w, Humidity ration: kg_vapor / kg_dry_air
    φ, Relative humidity: 0 ≤ φ ≤ 1
    q, Heat flow rate: W
    m, Mass flow rate: kg/s
    β, by-pass factor of the cooling coil, 0 ≤ β ≤ 1

Points on the psychrometric chart (θ, w):
o) out      outdoor air
0) M        mixed: outdoor + recycled air
1) s        efective coil surface or Apparatus Dew Point (ADP)
2) C        coil leaving air (saturated mixed with by-passed)
3) S        supply air
4) I, R, E  indoor, exhausted, recycled air


Constant Air Volume (CAV): linear control problem
-------------------------------------------------
  E                                     I
<=4=====================================4========================
  mo     ||                             m                      ||
       R 4 (m-mo) ========0=======                             ||
         ||   M   ||     βm     ||   C            S         I  ||
==o====>[MR]==0==>||           [MX]==2==>[HC]==F==3==>[TZ]==4==||
  mo          m   ||            ||   m    /    /  m    //   m  |
                  ==0==>[CC]==1===        s    m       sl      |
                 (1-β)m  // (1-β)m        |            ||      |
                         sl               |           [BL]<-mi |
                         V                |            //      |
                         t                |            sl      |
                         |                |                    |
                         |                |<------[Kw]---------+<-wI<-φI
                         |<-----------------------[Kθ]---------+<-θI

Inputs:
θo, φo      outdoor temperature & humidity ratio (=o=)
θ5sp, φ5sp  indoor temperature & humidity ratio set points (<-θI, <-φI)
mi          infiltration mass flow rate (<-mi)
QsBL, QlBL    auxiliary sensible and latent loads ([BL] sl)

Parameters:
m           mass flow rate of dry air
mo          mass flow rate of outdoor dry air
β           by-pass factir od cooling coil
UA          overall heat transfer coefficient of the building ([BL])

Elements (16 equations):
MR          mixing with given mass flow rate (2 equations)
CC          cooling coil (4 equations)
MX          mixing with given ratio (2 equations)
HC          heating (2 equations)
TZ          thermal zone (2 equations)
BL          building (2 equations)
Kθ          indoor temperature controller (1 equation)
Kw          indoor humidity controller (1 equation)
F           fan (m is a given parameter)

Outputs (16 unknowns):
0, ..., 4   temperature and humidity ratio (10 unknowns)
Qt, Qs, Ql  total, sensible and latent heat of CC (3 unknowns)
Qs          sensible heat of HC (1 unknown)
Qs, Ql      sensible and latent heat of TZ (2 unknowns)


VAV System with linear & least-squares controllers for  & θS
------------------------------------------------------------------

linear controller (Kθ & Kw) for θI, φI
non-linear controller (ls) for θS

<=4================================m==========================
       ||                                                   ||
       4 (m-mo) =======0=======                             ||
out    ||    M  ||  (1-β)m   ||    C            S        I  ||
mo===>[MX1]==0==||          [MX2]==2==[HC]==F===3=>[TZ]==4==||
                ||         s ||        /   /    |    //     |
                ===0=[CC]==1===       s   m     |   sl      |
                     /\\   βm         |   |     |   ||      |
                    t  sl             |   |<-ls-|  [BL]<-mi |
                    |                 |            //       |
                    |                 |           sl        |
                    |                 |                     |
                    |                 |<------[K]-----------|<-wI
                    |<------------------------[K]-----------|<-θI

Inputs:
θo, φo      outdoor temperature & relative humidity
θI, φI      indoor air temperature and humidity
θS          supply air temperature
QsTZ        sensible heat load of TZ
QlTZ        latent heat load of TZ

Elements (11 equations):
CC          cooling coil (4 equations)
HC          heating coil (2 equations)
TZ          thermal zone (2 equations)
F           fan (m is given)
KθI         indoor temperature controller (1 equation)
KwI         indoor humidity controller (1 equation)
lsθS        mass flow rate of dry air controller (1 non-linear equation)

Outputs (11 unknowns):
0, 1, 2     temperature and humidity ratio (6 unknowns)
QsCC, QlCC  sensible and latent heat of CC (2 unknowns)
QtCC        total heat load of CC (1 unknown)
QsHC        sensible heat load of HC (1 unknown)
Parameter:
m           mass flow rate of dry air (1 unknown)


CONTENTS (methods)
========================================================================
__init__    Initialization of CcTZ object.
lin_model   Solves the set of linear equations
            with saturation curve linearized around ts0
solve_lin   Solves iteratively the lin_model s.t. the error of
            humid. ratio between two iterrations is approx. zero
            (i.e. solves ws = f(θs) for saturation curve).
m_ls        Finds m s.t. θS = θSsp (solves θS - θSsp = 0 for m).
            Uses least-squares to find m that minimizes θS - θSsp
psy_chart   Draws psychrometric chart (imported from psychro)
CAV_wd      CAV to be used in Jupyter widgets.
            solve_lin and draws psy_chart.
VAV_wd      VAV to be used in Jupyter widgets.
            m_ls and draws psy_chart.
"""
import numpy as np
import pandas as pd
import psychro as psy
from scipy.optimize import least_squares


# constants
c = 1e3         # J/kg K, air specific heat
l = 2496e3      # J/kg, latent heat

# to be used in self.m_ls / least_squares
m_max = 100     # ks/s, max dry air mass flow rat
θs_0 = 5        # °C, initial guess for saturation temperature


class MxCcRhTzBl:
    """
    **HVAC composition**:
        mixing, cooling, reaheating, thermal zone of building, recycling
    """

    def __init__(self, parameters, inputs):
        m, mo, β, Kθ, Kw = parameters
        θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL = inputs

        self.design = np.array([m, mo, β, Kθ, Kw,       # parameters
                                θo, φo, θ5sp, φ5sp,     # inputs air out, in
                                mi, UA, QsBL, QlBL])      # --"--  building
        self.actual = np.array([m, mo, β, Kθ, Kw,
                                θo, φo, θ5sp, φ5sp,
                                mi, UA, QsBL, QlBL])

    def lin_model(self, θs0):
        """
        Linear control problem.
            Solves a set of 16 linear equations.

            Saturation curve ws = f(θs) is linearized in θs0.

        s-point (θs, ws):

        - is on a tangent to φ = 100 % in θs0;

        - is **not** on the saturation curve (Apparatus Dew Point ADP).


        Model parameter from function call
        ----------------------------
        θs0     °C, temperature for which the saturation curve is liniarized

        Model parameters and inputs from object
        ---------------------
        m, mo, θo, φo, θ5sp, φ5sp, β, mi, UA, QsBL, QlBL = self.actual


        - m           mass flow rate of dry air

        - mo          mass flow rate of outdoor dry air

        - θo, φo      outdoor temperature & humidity ratio

        - θ5sp, φ5sp  indoor temperature & humidity ratio set points

        - mi          infiltration mass flow rate

        - QsBL, QlBL    auxiliary sensible and latent loads

        - β           by-pass factir od cooling coil

        - UA          overall heat transfer coefficient of the building

        Equations (16)
        -------------
        +----------------+----+----+----+----+----+----+----+----+
        | Element        | MR | CC | MX | HC | TZ | BL | Kθ | Kw |
        +================+====+====+====+====+====+====+====+====+
        | N° of equations| 2  | 4  |  2 | 2  | 2  | 2  | 1  | 1  |
        +----------------+----+----+----+----+----+----+----+----+

        Returns (16 unknowns)
        ---------------------
        x :

        - temperatures and humidity ratios:
            θM, wM, θs, ws, θC, wC, θS, wS, θI, wI

        - heat flow rates:
            QtCC, QsCC, QlCC, QsHC, QsTZ, QlTZ
        """
        """

      E                                     I
    <=4=====================================4========================
      mo     ||                             m                      ||
           R 4 (m-mo) ========0=======                             ||
             ||   M   ||     βm     ||   C            S         I  ||
    ==o====>[MR]==0==>||           [MX]==2==>[HC]==F==3==>[TZ]==4==||
      mo          m   ||            ||   m    /    /  m    //   m  |
                      ==0==>[CC]==1===        s    m       sl      |
                     (1-β)m  // (1-β)m        |            ||      |
                             sl               |           [BL]<-mi |
                             V                |            //      |
                             t                |            sl      |
                             |                |                    |
                             |                |<------[Kw]---------+<-wI<-φI
                             |<-----------------------[Kθ]---------+<-θI
        """
        m, mo, β, Kθ, Kw, θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL = self.actual
        wo = psy.w(θo, φo)      # humidity. out

        A = np.zeros((16, 16))  # matrix of coefficents of unknowns
        b = np.zeros(16)        # vector of inputs

        # MR
        A[0, 0], A[0, 8], b[0] = m * c, -(m - mo) * c, mo * c * θo
        A[1, 1], A[1, 9], b[1] = m * l, -(m - mo) * l, mo * l * wo
        # CC
        A[2, 0], A[2, 2], A[2, 11], b[2] = (1 - β) * m * c, -(1 - β) * m * c,\
            1, 0
        A[3, 1], A[3, 3], A[3, 12], b[3] = (1 - β) * m * l, -(1 - β) * m * l,\
            1, 0
        A[4, 2], A[4, 3], b[4] = psy.wsp(θs0), -1,\
            psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        A[5, 10], A[5, 11], A[5, 12], b[5] = -1, 1, 1, 0
        # MX
        A[6, 0], A[6, 2], A[6, 4], b[6] = β * m * c, (1 - β) * m * c,\
            - m * c, 0
        A[7, 1], A[7, 3], A[7, 5], b[7] = β * m * l, (1 - β) * m * l,\
            - m * l, 0
        # HC
        A[8, 4], A[8, 6], A[8, 13], b[8] = m * c, -m * c, 1, 0
        A[9, 5], A[9, 7], b[9] = m * l, -m * l, 0
        # TZ
        A[10, 6], A[10, 8], A[10, 14], b[10] = m * c, -m * c, 1, 0
        A[11, 7], A[11, 9], A[11, 15], b[11] = m * l, -m * l, 1, 0
        # BL
        A[12, 8], A[12, 14], b[12] = (UA + mi * c), 1,\
            (UA + mi * c) * θo + QsBL
        A[13, 9], A[13, 15], b[13] = mi * l, 1, mi * l * wo + QlBL
        # Kθ indoor temperature controller
        A[14, 8], A[14, 10], b[14] = Kθ, 1, Kθ * θ5sp
        # Kw indoor humidity ratio controller
        A[15, 9], A[15, 13], b[15] = Kw, 1, Kw * psy.w(θ5sp, φ5sp)

        x = np.linalg.solve(A, b)
        return x

    def solve_lin(self, θs0):
        """
        Finds saturation point on saturation curve ws = f(θs).
            Solves iterativelly *lin_model(θs0)*:

            θs -> θs0 until ws - psy(θs, 1) < ε.

        Parameters
        ----------
        θs0     initial guess saturation temperature

        Method from object
        ---------------------
        *self.lin_model(θs0)*

        Returns (16 unknowns)
        ---------------------
        x of *self.lin_model(self, θs0)*
        """
        ε = 0.01e-3     # admisible error for ws
        # Δ_ws = 2 * ε    # kg/kg, initial difference to start the iterations
        # while Δ_ws > ε:
        #     x = self.lin_model(θs0)
        #     Δ_ws = abs(psy.w(x[2], 1) - x[3])   # psy.w(θs, 1) = ws
        #     θs0 = x[2]                          # actualize θs0
        while True:
            x = self.lin_model(θs0)
            Δ_ws = abs(psy.w(x[2], 1) - x[3])   # psy.w(θs, 1) = ws
            θs0 = x[2]                          # actualize θs0
            if Δ_ws < ε:
                break
            elif θs0 < -100 or θs0 > 100:
                print("\n================================================")
                print(f'ERROR in solve_lin: no solution, θs0 = {θs0: .2f} °C')
                print("===============================================\n ")
                break
        return x

    def m_ls(self, value, sp):
        """
        Mass flow rate *m* controls supply temperature *θS*
        or indoor humidity *wI*.
            Finds *m* which solves *value = sp*,
            i.e. minimizes *ε = value - sp*.

            Uses *scipy.optimize.least_squares* to solve the non-linear system.

        Parameters
        ----------
        value   (type string) 'θS' or 'wI': controlled variable

        sp      (type float): value of setpoint

        Calls
        -----
        *ε(m)*  which gives (value - sp) to be minimized for m

        Returns (16 unknowns)
        ---------------------
        x           given by *self.lin_model(self, θs0)*
        """
        # from scipy.optimize import least_squares

        def ε(m):
            """
            (m)

            Returns difference (value - sp) as a function of m

                ε  calculated by self.solve_lin(θs0)

                m  bounds=(0, m_max); m_max hard coded (global variable)

            Parameters
            ----------
            m : mass flow rate of dry air

            From object
                Method: self.solve.lin(θs0)

                Variables: self.actual <- m (used in self.solve.lin)

            Returns
            -------
            ε = |value - sp|: abs. difference between value and its set point
            """
            self.actual[0] = m
            x = self.solve_lin(θs_0)
            if value == 'θ4':
                θS = x[6]       # supply air
                return abs(sp - θS)
            elif value == 'φ5':
                wI = x[9]       # indoor air
                return abs(sp - wI)
            else:
                print('ERROR in ε(m): value not in {"θ5", "φ5"}')

        m0 = self.actual[0]     # initial guess
        if value == 'φ5':
            self.actual[4] = 0  # Kw = 0; no reheating
            sp = psy.w(self.actual[7], sp)  # w5 = f(θ5, φ5)
        # gives m for min(θSsp - θS); m0 is the initial guess of m
        # min ε(m) subject to 0 < m < m_max; m0 initial guess
        res = least_squares(ε, m0, bounds=(0, m_max))

        if res.cost < 0.1e-3:   # res.cost = min ε(m) s.t. 0 < m < m_max
            m = float(res.x)    # res.x = arg min ε(m) s.t. 0 < m < m_max
        else:
            print('RecAirVAV: No solution for m')

        self.actual[0] = m

        x = self.solve_lin(θs_0)
        return x

    def β_ls(self, value, sp):
        """
        (self, value, sp)

        Bypass β controls supply temperature θS or indoor humidity wI.
            Finds β which solves *value = sp*, i.e. minimizes *ε = value - sp*.

            Uses *scipy.optimize.least_squares* to solve the non-linear system.

        Parameters
        ----------
        value   (type string): 'θS' od 'wI' type of controlled variable

        sp      (type float): value of setpoint

        Calls
        -----
        *ε(m)*  gives (value - sp) to be minimized for m

        Returns (16 unknowns)
        ---------------------
        x           given by *self.lin_model(self, θs0)*
        """
        # from scipy.optimize import least_squares

        def ε(β):
            """
            Gives difference ε = (values - sp) function of β
                ε  calculated by self.solve_lin(ts0)
                β   bounds=(0, 1)

            Parameters
            ----------
            β : by-pass factor of the cooling coil

            From object
                Method: self.solve.lin(θs0)
                Variables: self.actual <- m (used in self.solve.lin)
            Returns
            -------
            ε = value - sp: difference between value and its set point
            """
            self.actual[2] = β
            x = self.solve_lin(θs_0)
            if value == 'θ5':
                θS = x[6]       # supply air
                return abs(sp - θS)
            elif value == 'φ5':
                wI = x[9]       # indoor air
                return abs(sp - wI)
            else:
                print('ERROR in ε(β): value not in {"θS", "wI"}')

        # β0 = self.actual[2]     # initial guess
        β0 = 0.1                # initial guess
        if value == 'φ5':
            self.actual[4] = 0
            sp = psy.w(self.actual[7], sp)
        # gives β for min(θSsp - θS); β0 is the initial guess of β
        # min ε(β) subject to 0 < β < 1; β0 initial guess
        res = least_squares(ε, β0, bounds=(0, 1))

        if res.cost < 1e-5:     # res.cost = min ε(β) s.t. 0 < β < 1
            β = float(res.x)    # res.x = arg min ε(β) s.t. 0 < β < 1
        else:
            print('RecAirVBP: No solution for β')

        self.actual[2] = β
        x = self.solve_lin(θs_0)
        return x

    def psy_chart(self, x, θo, φo):
        """
        Plot results on psychrometric chart.

        Parameters
        ----------
        x : θM, wM, θs, ws, θC, wC, θS, wS, θI, wI,
            QtCC, QsCC, QlCC, QsHC, QsTZ, QlTZ
                    results of self.solve_lin or self.m_ls
        θo, φo      outdoor point

        Returns
        -------
        None.

        """
        # Processes on psychrometric chart
        wo = psy.w(θo, φo)
        # Points: O, s, S, I
        θ = np.append(θo, x[0:10:2])
        w = np.append(wo, x[1:10:2])
        # Points       o   1  2  3  4  5        Elements
        A = np.array([[-1, 1, 0, 0, 0, 1],      # MR
                      [0, -1, 1, 0, 0, 0],      # CC
                      [0, -1, 1, 1, 0, 0],     # MX
                      [0, 0, 0, -1, 1, 0],      # HC
                      [0, 0, 0, 0, -1, 1]])     # TZ

        on_psy_chart = all(-10 < θ_value < 50 for θ_value in θ)
        on_psy_chart &= all(0 < w_value < 0.65 for w_value in w)
        if on_psy_chart:
            psy.chartA(θ, w, A)
        else:
            print('Values out of the phychroetric chart')

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)         # kg/kg -> g/kg
        P = pd.concat([θ, w], axis=1)   # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format})
        print()
        print(output)

        Q = pd.Series(x[10:], index=['QtCC', 'QsCC', 'QlCC', 'QsHC',
                                     'QsTZ', 'QlTZ'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')
        return None

    def CAV_wd(self, θo=32, φo=0.7, θ5sp=24, φ5sp=0.5,
               mi=0.90, UA=690, QsBL=18_000, QlBL=2_500):
        """
        Constant air volume (CAV) to be used in Jupyter with widgets

        Parameters: given in Jupyetr widget
        ----------

        Returns
        -------
        None.
        """
        # To use fewer variables in Jupyter widget:
        # select what to be updated in self.actual, e.g.:
        # self.actual[[0, 1, 2, 5, 6]] = m, θo, φo, 1000 * QsTZ, 1000 * QlTZ
        self.actual[5:] = np.array([θo, φo, θ5sp, φ5sp,
                                    mi, UA, QsBL, QlBL])
        # self.actual[5:] = θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL

        θ0 = 40
        x = self.solve_lin(θ0)
        # print(f'm = {self.actual[0]: .3f} kg/s,\
        #       mo = {self.actual[1]: .3f} kg/s')
        print('m = {m: .3f} kg/s, mo = {mo: .3f} kg/s, β = {β: .3f}'.format(
            m=self.actual[0], mo=self.actual[1], β=self.actual[2]))
        self.psy_chart(x, self.actual[5], self.actual[6])

    def VBP_wd(self, value='φ5', sp=0.5, θo=32, φo=0.5, θ5sp=24, φ5sp=0.5,
               mi=1.35, UA=675, QsBL=34_000, QlBL=4_000):
        """
        Variabl by-pass (VBP) to be used in Jupyter with widgets

        Parameters
        ----------
        value       {"θS", "wI"}' type of value controlled
        sp          set point for the controlled value
        θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL
                    given by widgets in Jupyter Lab

        Returns
        -------
        None.
        """
        """
        value='θS'

        <=4================================4============================
                ||                         m                          ||
                4 (m-mo) =======0=======                              ||
                ||    M  ||  (1-β)m   ||    C            S         I  ||
        θo,φo=>[MX1]==0==||          [MX2]==2==[HC]==F===3==>[TZ]==4==||
         mo              ||         s ||        /   /    |    //      |
                         ===0=[CC]==1===       s   m     |   sl       |
                              /\\   βm         |         |   ||       |
                             t  sl  |          |         |  [BL]<-mi  |
                             |      |          |         |  //        |
                             |      |          |         | sl         |
                             |      |          |         |            |
                             |      |<-------- | -----ls-|<-θSsp      |
                             |                 |<-----[K]-------------|<-wI
                             |<-----------------------[K]-------------|<-θI


        value='φ5' (KwI = 0)

        <=4================================4============================
                ||                         m                          ||
                4 (m-mo) =======0=======                              ||
                ||    M  ||  (1-β)m   ||    C            S         I  ||
        θo,φo=>[MX1]==0==||          [MX2]==2==[HC]==F===3==>[TZ]==4==||
         mo              ||         s ||        /   /         //      |
                         ===0=[CC]==1===       s   m         sl       |
                              /\\   βm         |             ||       |
                             t  sl  |          |            [BL]<-mi  |
                             |      |          |            //        |
                             |      |          |           sl         |
                             |      |          |                      |
                             |      |<-------- | ------ls-------------|<-φI
                             |                 |<-----[K]-------------|<-wI
                             |<-----------------------[K]-------------|<-θI
        """
        # Design values
        self.actual[5:] = θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL

        x = self.β_ls(value, sp)
        print('m = {m: .3f} kg/s, mo = {mo: .3f} kg/s, β = {β: .3f}'.format(
            m=self.actual[0], mo=self.actual[1], β=self.actual[2]))
        self.psy_chart(x, θo, φo)

    def VAV_wd(self, value='θ4', sp=14, θo=32, φo=0.7, θ5sp=24, φ5sp=0.5,
               mi=0.90, UA=690, QsBL=18_000, QlBL=2_500):
        """
        Variable air volume (VAV) to be used in Jupyter with widgets

        Parameters
        ----------
        value       {"θS", "wI"}' type of value controlled
        sp          set point for the controlled value
        θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL
                    given by widgets in Jupyter Lab

        Returns
        -------
        None.
        """
        """
        value='θS' (KwI = 0)

        <=4================================4===========================
                ||                         m                          ||
                4 (m-mo) =======0=======                              ||
                ||    M  ||  (1-β)m   ||    C            S         I  ||
        θo,φo=>[MX1]==0==||          [MX2]==2==[HC]==F===3==>[TZ]==4==||
         mo              ||         s ||        /   /    |    //      |
                         ===0=[CC]==1===       s   m     |   sl       |
                              /\\   βm         |   |     |   ||       |
                             t  sl             |   |     |  [BL]<-mi  |
                             |                 |   |     |   //       |
                             |                 |   |     |  sl        |
                             |                 |   |     |            |
                             |                 |   |--ls-|<-θS        |
                             |                 |<-----[K]-------------|<-wI
                             |<-----------------------[K]-------------|<-θI

        value='wI' (KwI = 0)

        <=4================================4===========================
                ||                         m                          ||
                4 (m-mo) =======0=======                              ||
                ||    M  ||  (1-β)m   ||    C            S         I  ||
        θo,φo=>[MX1]==0==||          [MX2]==2==[HC]==F===3==>[TZ]==4==||
         mo              ||         s ||        /   /         //      |
                         ===0=[CC]==1===       s   m         sl       |
                              /\\   βm         |   |         ||       |
                             t  sl             |   |        [BL]<-mi  |
                             |                 |   |         //       |
                             |                 |   |        sl        |
                             |                 |   |                  |
                             |                 |   |--ls--------------|<-wI
                             |                 |<-----[K]-------------|<-wI
                             |<-----------------------[K]-------------|<-θI

        """
        # Design values
        self.actual[5:] = θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL

        x = self.m_ls(value, sp)
        print('m = {m: .3f} kg/s, mo = {mo: .3f} kg/s, β = {β: .3f}'.format(
            m=self.actual[0], mo=self.actual[1], β=self.actual[2]))
        self.psy_chart(x, θo, φo)


# TESTS: uncomment
#
# Kθ, Kw = 1e10, 0     # Kw can be 0
# β = 0.16

# m, mo = 3.093, 0.94179
# θo, φo = 32, 0.5
# θ5sp, φ5sp = 26, 0.5

# mi = 15e3 / (l * (psy.w(θo, φo) - psy.w(θ5sp, φ5sp)))   # kg/s
# UA = 45e3 / (θo - θ5sp) - mi * c                        # W/K
# QsBL, QlBL = 0, 0     # W

# print(f'QsTZ = {(UA + mi * c) * (θo - θ5sp): ,.1f} W')
# print(f'QlTZ = {mi * l * (psy.w(θo, φo) - psy.w(θ5sp, φ5sp)): ,.1f} W')

# parameters = m, mo, β, Kθ, Kw
# inputs = θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL
# cool = MxCcRhTzBl(parameters, inputs)

# # 1. CAV
# print('\nCAV: m given')
# cool.CAV_wd()
# cool.CAV_wd(θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)
# # 2.
# print('\nVAV: control θS')
# θSsp = 11.45
# # cool.VAV_wd('θS', θSsp)
# cool.VAV_wd('θS', θSsp, θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)
# # 3.
# print('\nVAV: control φI')
# cool.VAV_wd('φ5', 0.5, θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)
# 4.
# print('\nVBP: control φI')
# wIsp = psy.w(26, 0.5)
# m = 3.5
# cool.actual[0] = m
# # cool.VBP_wd('wI', wIsp)
# φ5 = 0.5
# x = cool.VBP_wd('φ5', φ5, θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)
# cool.VBP_wd('wI', wIsp, θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)

# """
# Two solutions as a function of β0:
#     β0 = 0.1 ... 0.4 -> β = 0.101
#     β0 = 0.5 ... 0.9 -> β = 0.743
# Explanation
# The thermal zone characteristics cuts the saturation curve in two points.
# """

# """
# Controlling θS with β: NO SOLUTION
# Explanation
# Cannot have imposed m, θS, θI and QsTZ.
# Given θI and QsTZ:
#     either m --> θS
#     or θS --> m
# """
# print('\nVBP: control θS')
# θSsp = 11.77
# m = 3.162
# cool.actual[0] = m
# cool.VBP_wd('θS', θSsp, θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)


# Kw = 1e10
# cool.actual[4] = Kw
# cool.CAV_wd(θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL)
