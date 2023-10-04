import numpy as np
import pandas as pd
from interpolation_of_coeffs import force_coeffs_10MW

B = 3           # Number of blades [-]
RHO = 1.225     # Air density [kg/m^3]
MU = 1.81e-5    # Dynamic viscosity of air [Pa*s]

RELAX_FACTOR = 0.1
MAX_ITERATIONS = 1000


def steady_bem_for_each_airfoil(
    R: float,
    V_0: float,
    omega: float,
    theta_p: float,
    lamda: float,
    airfoil: dict,
    model: int = 'Glauret',
    error: float = 1e-6,
    aoa: float = None
    ) -> dict:
    """
    Calculate the local loads on a segment of a blade using the classical steady Blade Elemant Momentum (BEM) method.
    The Glauret or Wilson & Walker models can be used for the correction of the axial induction factor (a).

    If an 'aoa' value is given then a new 'beta' angle will be calculated based on that input.

    Args:
    ----------
        R:              Radius of wind turbine [m]
        V_0:            Wind speed [m/s]
        omega:          Angular velocity of rotor [rad/s]
        theta_p:        Pitch angle [deg]
        lamda:          Tip speed ratio [-]
        airfoil:        Dict of an airfoil with keys 'r', 'c', 'beta', 'thick'.
        model:          Choosen correction model ('Glauret' or 'Wilson & Walker').
        error:          Acceptable successive error.
        aoa:            Angle of attack [deg] (Default=None)

    Returns:
    ----------
        Dict with following keys:
            p_n:        Load normal to rotorplane [N/m]
            p_t:        Load tangential to rotorplane [N/m]
            C_P_local:  Local power coefficient [-]
            C_T_local:  Local thrust coefficient [-]
            aoa:        Angle of attack [deg]
            beta:       Pitch angle [deg]
    """

    # Check model
    if model == 'Glauret':
        a_c = 1/3
    elif model == 'Wilson & Walker':
        a_c = 0.2
    else:
        raise Exception('Invalid model selected. Select "Glauret" or "Wilson & Walker".')
    
    # Check if 'aoa' was given
    if aoa is None:
        calculate_aoa = True
    else:
        calculate_aoa = False

    # Unpack airfoil data
    r, c, beta, thick = airfoil['r'], airfoil['c'], airfoil['beta'], airfoil['thick']

    # Initialize axial and tangential induction factors
    a, a1 = 0, 0

    # Initialize errors and iteration counter
    error_a, error_a1, icount = 1, 1, 0

    # Iterative loop
    while error_a > error or error_a1 > error:
        # Compute flow angle (RAD)
        phi = np.arctan( ( (1 - a)*V_0 )/( (1 + a1)*omega*r) )

        # Compute local angle of attack (DEG)
        if calculate_aoa is True:
            aoa = phi*180/np.pi - (beta + theta_p)                          

        # Interpolate txt data to find the lift and drag coefficients
        C_l, C_d, _ = force_coeffs_10MW(aoa, thick)

        # Compute the normal and tangential load coefficients
        C_n = C_l*np.cos(phi) + C_d*np.sin(phi)
        C_t = C_l*np.sin(phi) - C_d*np.cos(phi)

        # Compute solidity
        sigma = (c*B)/(2*np.pi*r)

        # Compute thrust coefficient
        C_T_local = (1 - a)**2 * C_n * sigma /  (np.sin(phi))**2

        # Update induction factors using the simple model
        F = 2/np.pi * np.arccos( np.exp( -B/2 * (R-r)/(r*np.sin(np.abs(phi)))) )

        # Maybe I should somehow take into account the possibility of a>1 ???
        if a <= a_c:
            a_new = (sigma*C_n)/(4*F*np.sin(phi)**2)*(1-a)
        elif model == 'Glauret':
            a_new = C_T_local/(4*F*(1-1/4*(5-3*a)*a))
        else:
            K = 4*F*np.sin(phi)**2 / (sigma * C_n)
            a_new = 1 + K/2*(1-2*a_c) - 1/2*np.sqrt( (K*(1-2*a_c)+2)**2 + 4*(K*a_c**2 - 1) )
        
        # Update tangential induction factor
        a1_new = sigma*C_t*(1+a1)/(4*F*np.sin(phi)*np.cos(phi))

        # Underrelaxing induction factors
        # if model == 2 and a > a_c:    # Not sure if relaxation should be used on both models ?????????
        if a > a_c:
            a_new = RELAX_FACTOR*a_new + (1-RELAX_FACTOR)*a
            a1_new = RELAX_FACTOR*a1_new + (1-RELAX_FACTOR)*a1

        # Errors
        error_a = np.abs(a - a_new)
        error_a1 = np.abs(a1 - a1_new)

        # Set new values
        a = a_new
        a1 = a1_new

        # Make sure it does not run forever
        icount += 1
        if icount > MAX_ITERATIONS:
            print('Not converged')
            return None
    
    # Calculate relative velocity and pressures
    Vrel = np.sqrt(((1-a)*V_0)**2 + ((a1+1)*omega*r)**2)
    p_t = 1/2 * RHO * Vrel**2 * c * C_t
    p_n = 1/2 * RHO * Vrel**2 * c * C_n

    # Compute local power coefficient
    C_P_local = (B*lamda*c*(1-a)**2*C_t)/(2*np.pi*R*np.sin(phi)**2)

    # If an 'aoa' is given the compute a new beta
    if calculate_aoa is False:
        beta = phi*180/np.pi - (aoa + theta_p)

    return {
        'p_n': p_n,
        'p_t': p_t,
        'C_P_local': C_P_local,
        'C_T_local': C_T_local,
        'aoa': aoa,
        'beta': beta
    }


def steady_bem(
    R: float,
    V_0: float,
    omega: float,
    theta_p: float,
    lamda: float,
    df_blade: pd.DataFrame,
    model: int = 'Glauret',
    error: float = 1e-6
    ) -> dict:
    """
    Calculate the total loads on a blade using the classical steady Blade Elemant Momentum (BEM) method.
    The Glauret or Wilson & Walker models can be used for the correction of the axial induction factor (a).

    Args:
    ----------
        R:              Radius of wind turbine [m]
        V_0:            Wind speed [m/s]
        omega:          Angular velocity of rotor [rad/s]
        theta_p:        Pitch angle [deg]
        lamda:          Tip speed ratio [-]
        df_blade:       DataFrame of a blade with columns 'r', 'c', 'beta', 'thick'.
        model:          Choosen correction model ('Glauret' or 'Wilson & Walker').
        error:          Acceptable successive error.

    Returns:
    ----------
        Dict with following keys:
            P_out:          Power [kW]
            Thrust:         Thrust [kN]
            C_P:            Total power coefficient [-]
            C_T:            Total thrust coefficient [-]
            C_P_local_arr:  Array with local C_P for each blade element
            C_T_local_arr:  Array with local C_T for each blade element
    """
    # Initialize arrays
    p_n_arr = np.zeros_like(df_blade['r'])
    p_t_arr = np.zeros_like(df_blade['r'])
    C_P_local_arr = np.zeros_like(df_blade['r'])
    C_T_local_arr = np.zeros_like(df_blade['r'])

    # Loop over all blade elements except the last one
    for i, airfoil in df_blade.iterrows():
        if i == (df_blade.shape[0] - 1): break
        local_dict = steady_bem_for_each_airfoil(R, V_0, omega, theta_p, lamda, airfoil, model, error)
        p_n_arr[i] = local_dict['p_n']
        p_t_arr[i] = local_dict['p_t']
        C_P_local_arr[i] = local_dict['C_P_local']
        C_T_local_arr[i] = local_dict['C_T_local']

    # Integrate dTdr and dQdr over radius to get T and Q
    Thrust =  B*np.trapz(p_n_arr, x=df_blade['r'])                       # total axial force
    P_out = omega*B*np.trapz(df_blade['r']*p_t_arr, x=df_blade['r'])     # total torque

    # Calculate power in
    P_in = RHO * V_0 **3 / 2 * np.pi * R** 2

    # Calculate power and thrust coefficients
    C_P =  P_out / P_in
    C_T = Thrust/(RHO * V_0**2 / 2 * np.pi * R** 2)

    # [W] -> [kW],   [N] -> [kN]
    P_out = P_out/10**3
    Thrust = Thrust/10**3

    return {
        'P_out': P_out,
        'Thrust': Thrust,
        'C_P': C_P,
        'C_T': C_T,
        'C_P_local_arr': C_P_local_arr,
        'C_T_local_arr': C_T_local_arr,
    }


def find_maxes_of_matrix(matrix, row_list, column_list):
    """
    If for example C_P matrix is build like this:
    
    for i, theta_p in enumerate(theta_p_list):
        for j, lamda in enumerate(lamda_list):
            C_P_matrix[i, j] = C_P
    
    +--------+--------+--------+
    | i0, j0 | i0, j1 | i0, j2 |
    +--------+--------+--------+
    | i1, j0 | i1, j1 | i1, j2 |
    +--------+--------+--------+
    | i2, j0 | i2, j1 | i2, j2 |
    +--------+--------+--------+

    And let's say C_P_max is at (i2, j1) then use this function like so:

    C_P_max, theta_p_for_C_P_max, lamda_for_C_P_max = find_maxes_of_matrix(C_P_matrix, theta_p_list, lamda_list)
    """
    max_value = np.max(matrix)
    i, j = np.unravel_index(matrix.argmax(), matrix.shape)
    row = row_list[i]
    column = column_list[j]
    return max_value, row, column