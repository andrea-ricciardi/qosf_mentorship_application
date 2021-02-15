# -*- coding: utf-8 -*-
import numpy as np
import q_settings
from q_simulator import QSimulator
from variational import BellState

def main():
    # Variational algorithm inputs:
    # - variational_algo can be 'bell_state', in which case the program will
    #   run the "Bell State" variational algorithm, or None, in which case
    #   no variational algorithm is run
    # - initial_variational_params can be an array with the initial guess
    #   for the variational parameters, or None, in case no variational
    #   algorithm is run
    variational_algo = 'bell_state'
    initial_variational_params = np.array([0.0, 0.0, 1.0]) # np.array or None
    # End variational algorithm inputs
    
    num_qubits = 2
    # initial_amplitudes is a list of lists. The length of the list can be
    # either 1 or num_qubits:
    # - if 1 => all the qubits will have the same amplitudes
    # - if num_qubits => nth qubit will have amplitudes given by the nth
    #   element on the list.
    initial_amplitudes = [[1.0, 0.0]]
    n_counts = 1000
    
    # Circuit map. Accepted keys of the dictionaries are 'gate', 'target',
    # 'control' and 'params' if needed.
    circuit = [
        {'gate': 'u3', 'target': [0], 'control': [], 
         'params': {'theta':'global_1', 'phi':'global_2', 'lambda':'global_3'}
        },
        {'gate': 'cx', 'target': [1], 'control': [0]}
    ]
    
    if variational_algo is None:
        run_nonvariational_algo(
            num_qubits, initial_amplitudes, circuit, n_counts
        )
    else:
        run_variational_algo(
            variational_algo, initial_variational_params, num_qubits, 
            initial_amplitudes, circuit, n_counts
        )
    
    
def run_variational_algo(algo, initial_params, num_qubits, initial_amplitudes, 
                         circuit, n_counts):
    print(">>> Variational algorithm <<<")
    print()
    
    assert initial_params is not None and len(initial_params) > 0
    
    # Make and print settings
    variational_settings = q_settings.VariationalAlgoSettings(
        algo, num_qubits, circuit, n_counts, initial_amplitudes
    )
    variational_settings.print_settings()
    # Sanity check the settings
    variational_settings.sanity_checks(initial_params)
    
    if algo == 'bell_state':
        # Algorithm: optimize U3 parameters so that the circuit given by the
        # U3 gate and by a CX gate generates a Bell state on two qubits.
        
        # 1) Initialize algorithm
        bell_algo = BellState(variational_settings)
        # 2) Optimize parameters
        bell_algo.minimize(initial_params)
        # 3) Make circuit map with the optimized parameters filled into U3.
        nonvariational_circuit = bell_algo.filled_circuit_map(
            bell_algo.minimum.x
        )
        # 4) Run circuit with the optimized parameters
        run_nonvariational_algo(
            num_qubits, initial_amplitudes, nonvariational_circuit, n_counts
        )
    else:
        raise ValueError(
            "Variational algorithm {} not recognized".format(algo)
        )
    
def run_nonvariational_algo(num_qubits, initial_amplitudes, circuit, n_counts):
    print(">>> Non-variational algorithm <<<\n")
    # Make and print settings
    simulator_settings = q_settings.SimulatorSettings(
        num_qubits, circuit, n_counts, initial_amplitudes
    )
    simulator_settings.print_settings()
    
    assert not simulator_settings.circuit.has_variational_gate()
    
    # Initialize simulator
    simulator = QSimulator(simulator_settings)
    simulator.run_circuit()
    # Print results
    simulator.print_counts(with_zero_counts=False)

if __name__ == '__main__':
    main()
    