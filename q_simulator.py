# -*- coding: utf-8 -*-

import numpy as np
import support_functions

class QSimulator:
    """
    Class for the quantum circuit simulator
    ...
    
    Attributes
    ----------
    settings : SimulatorSettings
    circuit  : QCircuit
    
    Methods
    -------
    initialize_circuit(variational_params=None)
        Initialize the circuit in self.circuit.
    run_circuit(variational_params=None)
        Initialize and run the circuit, then measure the state.
    get_counts(with_zero_counts=True)
        Returns the counts of the state after measurement.
    print_counts(with_zero_counts=True)
    
    """
    
    def __init__(self, simulator_settings):
        """
        simulator_settings : SimulatorSettings
        """
        self.settings = simulator_settings
        
    def initialize_circuit(self, variational_params=None):
        """
        Initialize the circuit in self.circuit.
        
        Parameters
        ----------
        variational_params : np.array or None
            When running a variational algorithm, it is the list of parameters
            to feed to the circuit; when running a non-variational algorithm,
            it is None.
        """
        self.circuit = QCircuit(self.settings.circuit, variational_params)
        
    def run_circuit(self, variational_params=None):
        """
        Initialize and run the circuit, then measure the state of the system.
        
        Parameters
        ----------
        variational_params : np.array or None
            See self.initialize_circuit.
        """
        self.initialize_circuit(variational_params)
        self.circuit.run()
        self.circuit.measure_state(self.settings.n_counts)
        
    def get_counts(self, with_zero_counts=True):
        """
        Output : dict
            States as keys, number of counts as elements.
            
        Parameters
        ----------
        with_zero_counts : boolean
            True for including the states with 0 counts, False otherwise.
        """
        return self.circuit.state.get_counts(with_zero_counts)
    
    def print_counts(self, with_zero_counts=True):
        print("*** Final counts ***")
        print()
        print(self.get_counts(with_zero_counts))
        
class QCircuit:
    """
    Class for the quantum circuit
    ...
    
    Attributes
    ----------
    settings : CircuitSettings
    state    : QState
        Instance representing the combined state of the qubits.
    gates    : list of QGate
        Each element is an instance of QGate representing a gate in the circuit
    
    Methods
    -------
    run()
        For each gate, calculate the matrix operator and apply it to the state.
    measure_state(n_counts)
        Measure the state of the system.
    
    """
    
    def __init__(self, circuit_settings, variational_params=None):
        """
        Parameters
        ----------
        circuit_settings   : CircuitSettings
        variational_params : np.array or None
            See QSimulator.initialize_circuit.
        """
        self.settings = circuit_settings
        self.state = QState(self.settings.state)
        self.gates = []
        for gate_settings in self.settings.gates:
            g_settings = gate_settings.copy()
            if gate_settings.is_variational():
                g_settings.fill_variational_params(variational_params)
            self.gates.append(QGate(g_settings))
        
    def run(self):
        """
        For each gate, calculate the matrix operator and apply it to the state.
        """
        for gate in self.gates:
            operator = gate.get_operator(self.settings.num_qubits)
            self.state.apply_operator(operator)
            
    def measure_state(self, n_counts):
        """
        Measure the state of the system.
        """
        self.state.measure(n_counts)

class QState:
    
    """
    Class for the combined state of the qubits
    ...
    
    Attributes
    ----------
    settings    : StateSettings
    state_arr   : np.array
        Array of size 2 ** num_qubits representing the current state.
    index       : list
        Ordered list with the strings representing the possible final states.
    measurement : dict
        Measurement results. Keys are the final states, elements are number of 
        times they occurred after the measurement.
        
    Methods
    -------
    initialize()
        Initialize state_arr with the tensor product of the initial amplitudes.
    make_index()
        Build self.index.
    apply_operator(operator)
        Apply "operator" to the current state.
    measure(n_counts)
        Measure the current state of the system, putting the results into
        self.measurement.
    get_counts(with_zero_counts=True)
        Returns self.measurement with zero counts excluded/included depending
        on the function parameter.
    
    """
    
    def __init__(self, state_settings):
        """
        Parameters
        ----------
        state_settings : StateSettings
        """
        self.settings = state_settings
        self.state_arr = np.zeros(
            (2 ** self.settings.num_qubits,), dtype=complex
        )
        self.initialize()
        
        self.index = []
        self.make_index()
        
        self.measurement = {}
        
    def initialize(self):
        """
        Initialize self.state_arr with the tensor product of the initial 
        amplitudes.
        
        """
        
        def tensor_product():
            result = np.array([[1]])
            for ampl in self.settings.initial_amplitudes:
                result = np.kron(result, ampl)
            return support_functions.normalize_array(result)
        
        self.state_arr = np.array(tensor_product())
        
    def make_index(self):
        """
        Fill self.index.
        
        """
        self.index = []
        for n in range(2 ** self.settings.num_qubits):
            this_index = bin(n)[2:] # output is 0b...
            final_index = \
                '0' * (self.settings.num_qubits - len(this_index)) + this_index
            self.index.append(final_index)
            
    def apply_operator(self, operator):
        """
        Apply operator to self.state_arr.
        
        Parameters
        ----------
        operator : np.ndarray
            Matrix operator.
        """
        self.state_arr = np.dot(self.state_arr, operator)
        
    def measure(self, n_counts):
        """
        Measure the current state of the system.
        
        Parameters
        ----------
        n_counts : int
            Number of shots when measuring.
        """
        probs = np.squeeze(
            np.real(self.state_arr * np.conjugate(self.state_arr))
        )
        choices = np.random.choice(
            self.index, n_counts, p=probs / np.sum(probs)
        )
        unique, counts = np.unique(choices, return_counts=True)
        self.measurement = {ind: count for ind, count in zip(unique, counts)}
        
    def get_counts(self, with_zero_counts=True):
        """
        Output : dict
            Returns self.measurement if with_zero_counts is False, otherwise
            it returns self.measurement with the zero counts added.
            
        Parameters
        ----------
        with_zero_counts : boolean
        
        """
        if with_zero_counts:
            return {ind: self.measurement[ind] \
                    if ind in self.measurement else 0 for ind in self.index}
        else:
            return self.measurement
            
class QGate:
    
    """
    Class representing a quantum gate
    ...
    
    Attributes
    ----------
    settings : GateSettings
    
    Methods
    -------
    get_operator(num_qubits)
        Returns the gate's matrix operator.
    
    """
    
    def __init__(self, settings):
        """
        Parameters
        ----------
        settings : GateSettings
        """
        self.settings = settings
        
    def get_operator(self, num_qubits):
        """
        Output: np.ndarray
            Returns the gate's matrix operator.
            
        Parameters
        ----------
        num_qubits : int
            Total number of qubits.
        """
        target = self.settings.target
        control = self.settings.control
        # U is CCX/CNOT/CZ if adjacent qubits, X/X/Z if not adjacent
        U = self.settings.matrix 
        
        def operator_adjacent_qubits(target_indices, control_indices):
            adj_operator = np.array([[1]])
            I = np.identity(2)
            found_target_or_control = False
            for q_index in range(num_qubits):
                if q_index in target_indices or q_index in control_indices:
                    if not found_target_or_control:
                        adj_operator = np.kron(adj_operator, U)
                        found_target_or_control = True
                else:
                    adj_operator = np.kron(adj_operator, I)
            return adj_operator
        
        if support_functions.adjacent_qubits(
            control, target, self.settings.name
        ):
            # single-qubit gate or multiple qubit gates with adjacent qubits
            return operator_adjacent_qubits(target, control)
        
        # multi-qubits gate with non-adjacent qubits, because single-qubit 
        # gates are always adjacent.
        I = np.identity(2)
        
        assert len(control) in [0, 1, 2]
        assert len(target) in [1, 2]
        assert self.settings.name in support_functions.multiple_qgates_names()
        
        if self.settings.name in support_functions.controlled_gates_names(
                n_qubits=None
        ):
            # Get projection operator |0><0|
            p0 = support_functions.get_projection(0)
            # Get projection operator |1><1|
            p1 = support_functions.get_projection(1)
            
            if self.settings.name in support_functions.controlled_gates_names(
                    n_qubits=2
            ):
                operator_0, operator_1 = np.array([[1]]), np.array([[1]])
                for q_index in range(num_qubits):
                    if q_index in control:
                        operator_0 = np.kron(operator_0, p0)
                        operator_1 = np.kron(operator_1, p1)
                    else:
                        operator_0 = np.kron(operator_0, I)
                        operator_1 = np.kron(operator_1, U) \
                            if q_index in target else np.kron(operator_1, I)
                return operator_0 + operator_1
            
            if self.settings.name in support_functions.controlled_gates_names(
                    n_qubits=3
            ):
                op_1, op_2 = np.array([[1]]), np.array([[1]])
                op_3, op_4 = np.array([[1]]), np.array([[1]])
                for q_index in range(num_qubits):
                    if q_index == control[0]:
                        op_1 = np.kron(op_1, p0)
                        op_2 = np.kron(op_2, p1)
                        op_3 = np.kron(op_3, p0)
                        op_4 = np.kron(op_4, p1)
                    elif q_index == control[1]:
                        op_1 = np.kron(op_1, p0)
                        op_2 = np.kron(op_2, p0)
                        op_3 = np.kron(op_3, p1)
                        op_4 = np.kron(op_4, p1)
                    else:
                        op_1 = np.kron(op_1, I)
                        op_2 = np.kron(op_2, I)
                        op_3 = np.kron(op_3, I)
                        op_4 = np.kron(op_4, U) \
                            if q_index == target[0] else np.kron(op_4, I)
                return op_1 + op_2 + op_3 + op_4 
            else:
                raise ValueError("Controlled gates with more than 3 qubits"\
                                 "are not implemented")
                
        if self.settings.name == 'swap':
            # SWAP between non adjacent qubits
            
            # Swaps idx element with idx+1 element
            def swap_operator(idx):
                return operator_adjacent_qubits([idx, idx+1], [])
            
            operator = None
            for idx in range(target[0], target[1]):
                operator = swap_operator(idx) if operator is None \
                                    else np.dot(operator, swap_operator(idx))
            idx = target[1] - 2
            while idx >= target[0]:
                operator = swap_operator(idx) if operator is None \
                                    else np.dot(operator, swap_operator(idx))
                idx -= 1
            return operator
        else:
            raise ValueError(
                "Gate {} for non adjacent qubits not implemented".format(
                    self.settings.name
                )
            )
