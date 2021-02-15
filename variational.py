# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
from q_simulator import QSimulator
import scipy.optimize
import support_functions

class VariationalAlgorithm(ABC):
    """
    Abstract class for variational algorithms.
    ...
    
    Attributes
    ----------
    settings  : VariationalAlgoSettings
    simulator : QSimulator
    minimum   : scipy.optimize.OptimizeResult
        Information about the optimized minimum.
        
    Methods
    -------
    run_quantum(params): return float
        Run the quantum circuit and returns the cost.
    get_cost
        Abstract method for computing the cost.
    minimize(global_params)
        Minimize the cost by optimizing the variational parameters.
    filled_circuit_map(params): return list of dicts
        Return a circuit_map with the variational parameters filled by params.
    
    """
    
    def __init__(self, settings):
        """
        settings : VariationalAlgoSettings
        
        """
        self.settings = settings
        self.simulator = QSimulator(self.settings.simulator)
        
    def run_quantum(self, params):
        """
        Output: float
            Run the quantum circuit and returns the cost.
        
        Parameters
        ----------
        params : np.array
            Array with the parameters to pass to the circuit as variational
            parameters.
        """
        self.simulator.run_circuit(params)
        return self.get_cost(self.simulator.get_counts())
        
    @abstractmethod
    def get_cost(self):
        pass
    
    def minimize(self, global_params):
        """
        Fills self.minimum with information on the optimization.
        
        Parameters
        ----------
        global_params : np.array
            Initial parameters for the variational algorithm.
        """
        self.minimum = scipy.optimize.minimize(
            self.run_quantum, global_params, method="Powell", 
            tol=1e-7, options={'disp': True}
        )
    
    def filled_circuit_map(self, params):
        """
        Output : list of dict
            Returns the circuit_map after filling the variational strings with
            the parameters in params.
            
        Parameters
        ----------
        params : np.array
        
        """
        circuit_map = []
        for gate in self.simulator.circuit.gates:
            gate_dict = gate.settings.gate_dict.copy()
            if gate.settings.is_variational():
                gate_dict['params']=support_functions.fill_variational_params(
                    gate_dict['params'], params
                )
            circuit_map.append(gate_dict)
        return circuit_map
    
class BellState(VariationalAlgorithm):
    """
    Child of VariationalAlgorithm.
    Class representing the BellState variational algorithm, defined as the
    optimization of the U3 gate parameters in such a way that the circuit with
    the U3 gate the CNOT gate generates the Bell State
    ...
    
    Methods
    -------
    get_cost(counts): return float
        Returns the cost of the algorithm.
    
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        
        assert self.settings.num_qubits == 2
        
    def get_cost(self, counts):    
        """
        Output : float
            Returns the cost of the algorithm.
            
        Parameters
        ----------
        counts : dict
            Dictionary containing the measurements.
        """        
        cost = 0.0
        for ind in ['01','10']:
            cost += counts[ind] ** 2
        
        cost += (counts['00'] - counts['11']) ** 2
        return np.sqrt(cost)