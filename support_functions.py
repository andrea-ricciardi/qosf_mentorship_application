# -*- coding: utf-8 -*-
import numpy as np

def normalize_array(arr):
    """
    Output : np.array
    
    Parameters
    ----------
    arr : np.array
    """
    return arr / np.linalg.norm(arr)

def complex_exp(angle):
    """
    Returns the analytical expression of exp(i * angle).
    
    Output : complex
    
    Parameters
    ----------
    angle : float
    
    """
    return np.cos(angle) + 1j * np.sin(angle)

"""
Gates matrices.

"""

def get_x():
    return np.array([
        [0, 1],
        [1, 0]
    ])
    
def get_y():
    return np.array([
        [0, -1j],
        [1j, 0]
    ])

def get_z():
    return np.array([
        [1, 0],
        [0, -1]
    ])
    
def get_h():
    return np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)]
    ])
    
def get_p():
    return np.array([
        [1, 0],
        [0, 1j]
    ])
    
def get_t():
    return np.array([
        [1, 0],
        [0, complex_exp(np.pi / 4)]
    ])
    
def get_u3(params):
    assert 'theta' in params
    assert 'phi' in params
    assert 'lambda' in params
    
    theta, phi, lam = params['theta'], params['phi'], params['lambda']
    
    return np.array([
        [np.cos(theta / 2), -complex_exp(lam) * np.sin(theta / 2)],
        [complex_exp(phi) * np.sin(theta / 2), 
         complex_exp(theta + lam) * np.cos(theta / 2)]
    ])
    
def get_cnot():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]        
    ])
    
def get_znot():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]        
    ])
    
def get_swap():
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]        
    ])
    
def get_toffoli():
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]        
    ])
    
"""
Gates names.

"""
    
def cnot_names():
    return ['cx','cnot']
    
def toffoli_names():
    return ['cxx','ccx','toffoli']

def multiple_qgates_names():
    return cnot_names() + toffoli_names() + ['cz','swap']

def controlled_gates_names(n_qubits=None):
    """
    Controlled gates names depending on the number of qubits.
    
    Output : list of str
    
    Parameters
    ----------
    n_qubits : int or None
        If None, returns the names for 2 and 3 qubits.
    """
    if n_qubits is None:
        return controlled_gates_names(2) + controlled_gates_names(3)
    if n_qubits == 2:
        return cnot_names() + ['cz']
    if n_qubits == 3:
        return toffoli_names()
    return []
        
def adjacent_qubits(control_indices, target_indices, gate_name):
    """
    Returns True if the target and control indices are adjacent, 
    False otherwise. Single-qubit gates are always adjacent.
    
    Output : Boolean
    
    Parameters
    ----------
    control_indices : list of int
    target_indices  : list of int
    gate_name       : str
    
    """
    if len(control_indices) + len(target_indices) == 1:
        return True # single qubit gates are always adjacent
    
    diff_value = 1
    diff_target = [t - s for s, t in zip(target_indices, target_indices[1:])]
    if any(x != diff_value for x in diff_target):
        if gate_name == 'swap':
            # For SWAP, the order of target or control doesn't matter
            if any(x != -diff_value for x in diff_target):
                return False
        else:
            return False
    if len(control_indices) == 0:
        return True
    
    diff_control = [t-s for s,t in zip(control_indices, control_indices[1:])]
    if any(x != diff_value for x in diff_control):
        return False
    
    return target_indices[-1] - control_indices[0] == diff_value

def get_projection(ground_bit):
    """
    Returns the projection onto the state |ground_bit> of the control bit.
    
    Output : np.array
    
    Parameters
    ----------
    ground_bit : int in [0, 1]
    
    """
    assert ground_bit in [0, 1]
    
    if ground_bit == 0:
        return np.array([
            [1, 0],
            [0, 0]
        ])
    else:
        return np.array([
            [0, 0],
            [0, 1]
        ])
    
def is_variational_input(val):
    """
    Returns True if val is an input that should be replaced by some values
    in the context of the variational algorithm.
    
    Output : Boolean
    
    """
    return isinstance(val, str) and val.startswith('global')
    
def fill_variational_params(var_params_dict, params_arr):
    """
    Fills the variational inputs in var_params_dict with the parameters
    specified in params_arr.
    
    Output : dict
    
    Parameters
    ----------
    var_params_dict : dict
    params_arr      : np.array
    
    """
    for par, val in var_params_dict.items():
        if is_variational_input(val):
            global_index = int(val[-1]) - 1
            var_params_dict[par] = params_arr[global_index]
    return var_params_dict