# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:56:42 2022
@author: cgrinde

Modified by Spyros_Acheimastos
"""
import numpy as np

FILES = [
    'data/' + 'FFA-W3-241.txt',
    'data/' + 'FFA-W3-301.txt',
    'data/' + 'FFA-W3-360.txt',
    'data/' + 'FFA-W3-480.txt',
    'data/' + 'FFA-W3-600.txt',
    'data/' + 'cylinder.txt'
]

# Initializing tables
CL_TAB = np.zeros([105, 6])
CD_TAB = np.zeros([105, 6])
CM_TAB = np.zeros([105, 6])
AOA_TAB = np.zeros([105,])

# Reading of tables. Only do this once at startup of simulation
for i in range(np.size(FILES)):
    AOA_TAB[:], CL_TAB[:,i], CD_TAB[:,i], CM_TAB[:,i] = np.loadtxt(FILES[i], skiprows=0).T

# Thickness of the airfoils considered
THICK_PROF = np.array([24.1, 30.1, 36, 48, 60, 100])


def force_coeffs_10MW(aoa: float, thick: float) -> (float, float, float):
    """
    Calculate the aerodynamic coefficients for the blade of the DTU 10 MW
    wind turbine given an angle of attack and a thickness percentage.
    
    Args:
    ----------
        aoa:    Angle of attack in degrees.
        thick:  Thickness in t/c percentage.

    Returns:
    ----------
        cl:     Lift coefficient
        cd:     Drag coefficient
        cm:     Pitching moment coefficient
    """
    cl_aoa = np.zeros([1, 6])
    cd_aoa = np.zeros([1, 6])
    cm_aoa = np.zeros([1, 6])

    # Interpolate to current angle of attack:
    for i in range(np.size(FILES)):
        cl_aoa[0, i] = np.interp(aoa, AOA_TAB, CL_TAB[:,i])
        cd_aoa[0, i] = np.interp(aoa, AOA_TAB, CD_TAB[:,i])
        cm_aoa[0, i] = np.interp(aoa, AOA_TAB, CM_TAB[:,i])

    # Interpolate to current thickness:
    cl = np.interp(thick, THICK_PROF, cl_aoa[0,:])
    cd = np.interp(thick, THICK_PROF, cd_aoa[0,:])
    cm = np.interp(thick, THICK_PROF, cm_aoa[0,:])

    return cl, cd, cm


if __name__ == '__main__':
    # Lets test it:
    angle_of_attack = -10 # in degrees
    thickness = 27 # in percent!
    clift, cdrag, cmom = force_coeffs_10MW(angle_of_attack, thickness)

    print('cl:', clift)
    print('cd:', cdrag)
    print('cm:', cmom)
