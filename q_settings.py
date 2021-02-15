# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import support_functions

class VariationalAlgoSettings:
    
    """
    Class for variational algorithm settings
    ...
    
    Attributes
    ----------
    
    variational_algo : str
        Name of the variational algorithm.
    num_qubits       : int
        Total number of qubits.
    simulator        : SimulatorSettings
        Settings for the quantum circuit simulator.
        
    Methods
    -------
    
    sanity_checks(initial_params)
        Raise assertion errors if settings are not coherent with each other.
    print_settings()
    
    """
    
    def __init__(
            self, variational_algo, num_qubits, circuit_map, n_counts, 
            initial_amplitudes=[[1.0, 0.0]]
        ):
        
        """
        Parameters
        ----------
        variational_algo   : str
            Name of the variational algorithm.
        num_qubits         : int
            Total number of qubits.
        circuit_map        : list of dict
            List containing one dictionary per gate. Provided by the user.
        n_counts           : int
            Number of shots when measuring.
        initial_amplitudes : list of lists
            Initial amplitudes of the qubits. It can have length 1, in which
            case all the qubits will have the same amplitudes, or num_qubits,
            in which case each qubit will have amplitudes given by an element
            of the list.
        """
        self.variational_algo = variational_algo
        self.num_qubits = num_qubits
        self.simulator = SimulatorSettings(
            num_qubits, circuit_map, n_counts, initial_amplitudes
        )
        
    def sanity_checks(self, initial_params):
        """
        Raise assertion errors if settings are not coherent with each other.
        
        Parameters
        ----------
        initial_params : np.array
            Array with the initial parameters to provide to the variational
            algorithm.
        """
        assert self.simulator.circuit.has_variational_gate()
        for gate_settings in self.simulator.circuit.gates:
            assert gate_settings.how_many_variational_params() in \
                [0, len(initial_params)]
        if self.variational_algo == 'bell_state':
            assert len(self.simulator.circuit.gates) == 2
            assert self.simulator.circuit.gates[0].name == 'u3'
            assert self.simulator.circuit.gates[1].name in \
                support_functions.cnot_names()
            
    def print_settings(self):
        print("=> Variational Settings <=")
        print("Variational_algo: {}".format(self.variational_algo))
        print("Num_Qubits: {}".format(self.num_qubits))
        self.simulator.print_settings()

class SimulatorSettings:
    
    """
    Class for quantum circuit simulator settings
    ...
    
    Attributes
    ----------
    circuit  : CircuitSettings
        Settings for the circuit.
    n_counts : int
        Number of shots when measuring.
        
    Methods
    -------
    print_settings()
    
    """
    
    def __init__(
            self, num_qubits, circuit_map, n_counts, 
            initial_amplitudes=[[1.0, 0.0]]
        ):
        
        """
        Parameters
        ----------
        num_qubits         : int
            See VariationalAlgoSettings.__init__().
        circuit_map        : list of dict
            See VariationalAlgoSettings.__init__().
        n_counts           : int
            See VariationalAlgoSettings.__init__().
        initial_amplitudes : list of lists
            See VariationalAlgoSettings.__init__().
        """
        
        assert num_qubits > 0
        
        self.circuit = CircuitSettings(
            circuit_map, num_qubits, initial_amplitudes
        )
        self.n_counts = n_counts
        
    def print_settings(self):
        print("=> Simulator Settings <=")
        print("N_counts: {}".format(self.n_counts))
        self.circuit.print_settings()
        print()
        
class CircuitSettings:
    
    """
    Class for the quantum circuit
    ...
    
    Attributes
    ----------
    gates      : list of GateSettings
        List of gates settings, one for each gate.
    num_qubits : int
        See VariationalAlgoSettings.__init__().
    state      : StateSettings
        Settings for the qubits combined state.
        
    Methods
    -------
    has_variational_gate(): return boolean
        True if there is at least one variational gate in the circuit.
    print_settings()
    
    """
    
    def __init__(self, circuit_map, num_qubits, initial_amplitudes):
        """
        Parameters
        ----------
        circuit_map        : list of dict
            See VariationalAlgoSettings.__init__().
        num_qubits         : int
            See VariationalAlgoSettings.__init__().
        initial_amplitudes : list of lists
            See VariationalAlgoSettings.__init__().
        """
        self.gates = [GateSettings(gate_dict, num_qubits) \
                      for gate_dict in circuit_map]
        self.num_qubits = num_qubits
        self.state = StateSettings(num_qubits, initial_amplitudes)
    
    def has_variational_gate(self):
        """
        Output: Boolean
            True if the circuit has at least one variational gate.
        """
        for gate in self.gates:
            if gate.is_variational():
                return True
        return False
    
    def print_settings(self):
        print("=> Circuit Settings <=")
        print("Num_Qubits: {}".format(self.num_qubits))
        for idx, gate in enumerate(self.gates):
            gate.print_settings(idx + 1)
        self.state.print_settings()
    
class GateSettings:
    """
    Class for one single gate settings
    ...
    
    Attributes
    ----------
    gate_dict : dict
        Dictionary with at least 'target', 'control' and 'gate' as keys. It is
        one element of the circuit_map (See VariationalAlgoSettings.__init__())
    target    : list
        Target indices.
    control   : list
        Control indices.
    name      : str
        Name of the gate.
    matrix    : np.ndarray or None
        Unitary matrix representing the gate. When qubits are adjacent, it is
        the actual matrix; when qubits are not adjacent, it depends on the
        gate: for instance, for a controlled-U operation, the matrix is U.
        If the gate is variational, the matrix is None.
        
    Methods
    -------
    is_variational()               : return boolean
        True if the gate is variational, False otherwise.
    how_many_variational_params()  : return int
        Number of variational parameters in the gate.
    copy()                         : return GateSettings
        Returns a copy of the instance of GateSettings.
    fill_variational_params(params)
        Fills the variational parameters of the instance of GateSettings with
        the parameters specified in "params".
    print_settings()
    
    """
    
    def __init__(self, gate_dict, num_qubits):
        """
        Parameters
        ----------
        gate_dict  : dict
            See class description.
        num_qubits : int
            See VariationalAlgoSettings.__init__().
        """
        
        assert 'target' in gate_dict
        assert 'control' in gate_dict
        assert 'gate' in gate_dict
        
        self.gate_dict = gate_dict.copy()
        
        self.target = gate_dict['target']
        self.control = gate_dict['control']
            
        assert all(x in range(num_qubits) for x in self.target)
        assert all(x in range(num_qubits) for x in self.control)
        # intersection is empty
        assert len(list(set(self.target) & set(self.control))) == 0 
        
        self.name = gate_dict['gate'].lower()
        
        if len(self.target) + len(self.control) > 1:
            # Multiple-qubits gate
            assert self.name in support_functions.multiple_qgates_names()
            
            if self.name in support_functions.cnot_names():
                
                assert len(self.target) == 1 and len(self.control) == 1

                if support_functions.adjacent_qubits(
                    self.control, self.target, self.name
                ):
                    self.matrix = support_functions.get_cnot()
                else:
                    self.matrix = support_functions.get_x()
            elif self.name == 'cz':
                
                assert len(self.target) == 1 and len(self.control) == 1
                
                if support_functions.adjacent_qubits(
                    self.control, self.target, self.name
                ):
                    self.matrix = support_functions.get_znot()
                else:
                    self.matrix = support_functions.get_z()
            elif self.name == 'swap':
                
                assert len(self.target) == 2 and len(self.control) == 0
                
                self.matrix = support_functions.get_swap()
            elif self.name in support_functions.toffoli_names():
                
                assert len(self.target) == 1 and len(self.control) == 2
                
                if support_functions.adjacent_qubits(
                    self.control, self.target, self.name
                ):
                    self.matrix = support_functions.get_toffoli()
                else:
                    self.matrix = support_functions.get_x()
            else:
                raise ValueError(
                "Gate {} with target {} and control {} not recognized".format(
                    self.name, self.target, self.control
                ))
                
        else:
            # Single-qubit gate
            assert len(self.target) == 1 and len(self.control) == 0

            if self.name == 'h':
                self.matrix = support_functions.get_h()
            elif self.name == 'x':
                self.matrix = support_functions.get_x()
            elif self.name == 'y':
                self.matrix = support_functions.get_y()
            elif self.name == 'z':
                self.matrix = support_functions.get_z()
            elif self.name in ['s','p']:
                self.matrix = support_functions.get_p()
            elif self.name == 't':
                self.matrix = support_functions.get_t()
            elif self.name == 'u3':
                assert 'params' in gate_dict
                
                if any(support_functions.is_variational_input(v) \
                       for x, v in gate_dict['params'].items()):
                    self.matrix = None
                else:
                    self.matrix = support_functions.get_u3(gate_dict['params'])
            else:
                raise ValueError(
                "Gate {} with target {} and control {} not recognized".format(
                    gate_dict['gate'], self.target, self.control
                ))
        
    def is_variational(self):
        """
        Returns True if the gate is variational, False otherwise.
        """
        return self.how_many_variational_params() > 0
    
    def how_many_variational_params(self):
        """
        Returns the number of variational parameters in the gate.
        """
        count = 0
        if 'params' not in self.gate_dict:
            return count
        
        for key, val in self.gate_dict['params'].items():
            if support_functions.is_variational_input(val):
                count += 1
        return count
    
    def copy(self):
        return deepcopy(self)
    
    def fill_variational_params(self, params):
        """
        Fills self.gate_dict['params'] with the parameters provided by params
        and updates the matrix with the parameters.
        
        Parameters
        ----------
        params : np.array
            Array of floats.
        """
        self.gate_dict['params'] = support_functions.fill_variational_params(
            self.gate_dict['params'], params
        )
        self.matrix = support_functions.get_u3(self.gate_dict['params'])
        
    def print_settings(self, idx=None):
        if idx is None:
            print("=> Gate Settings:")
        else:
            print("=> Gate {} Settings <=".format(idx))
        print("Gate_dict: {}".format(self.gate_dict))
        print("Name: {}".format(self.name))
        print("Target: {}".format(self.target))
        print("Control: {}".format(self.control))
        print("Matrix: {}".format(self.matrix))
        
class StateSettings:
    
    """
    Class for the settings of the qubits' combined state
    ...
    
    Attributes
    ----------
    num_qubits         : int
        See VariationalAlgoSettings.__init__().
    initial_amplitudes : list of lists
        See VariationalAlgoSettings.__init__().
    
    Methods
    -------
    print_settings()
    
    """
    
    def __init__(self, num_qubits, initial_amplitudes):
        """
        Parameters
        ----------
        num_qubits : int
            See VariationalAlgoSettings.__init__().
        initial_amplitudes : list of lists
            See VariationalAlgoSettings.__init__().
        """
        
        assert num_qubits > 0
        assert len(initial_amplitudes) in [1, num_qubits]
        
        self.num_qubits = num_qubits
        if len(initial_amplitudes) == num_qubits:
            self.initial_amplitudes = initial_amplitudes
        else:
            self.initial_amplitudes = initial_amplitudes * num_qubits
        
        self.initial_amplitudes = [support_functions.normalize_array(
                np.array(ampl, dtype=complex)
            ) for ampl in self.initial_amplitudes]
        
    def print_settings(self):
        print("=> State Settings <=")
        print("Num_Qubits: {}".format(self.num_qubits))
        print("Initial amplitudes: {}".format(self.initial_amplitudes))