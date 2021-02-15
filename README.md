# qosf_mentorship_application

Hey,

This repository contains the screening task submission for the QOSF mentorship program.

# Task 3
I wrote a simple quantum circuit simulator from scratch. Quantum gates accepted by the program are: 
```
X, Y, Z, H, P, T, U3, CNOT, ZNOT, SWAP, TOFFOLI
```
and they work on adjacent or non-adjacent qubits.

As well as running any combination of these gates in the simulator, it is also possible to run a simple variational algorithm, the "Bell-State variational algorithm". This algorithm is designed for two qubits only, and optimizes the parameters of a U3 gate so that the following circuit produces a Bell state:

```
q_0 ------ U3 -------*
                     | cx
q_1 -----------------O
```

On my computer, the circuit runs smoothly up to TODO qubits, then I start to see some performance issues.

I will now go through the structure of the program, then I will explain each file one by one.

## Structure
The program is organized in five files:
- `main.py`: main program in which the user initialize the inputs.
- `q_settings.py`: package defining all the settings for the variational algorithm and for the circuit simulator.
- `q_simulator.py`: main package for the circuit simulator.
- `variational.py`: main package for the variational algorithm.
- `support_functions.py`: helper package with some functions used by the program. All the matrices are defined in here.

## First run
By running the program as it is, the user will run the Bell-State variational algorithm to find the optimal parameters, then such parameters will be used to run the optimized circuit on the simulator.

Let's now dive into all the packages. I will not go deep into them as the comments in the code should be able to explain everything that is not mentioned here.

## main.py: define the inputs and run the program
Let's start from the `main()` procedure. The user should first decide whether he/she wants to run the Bell-State variational algorithm or just run a quantum circuit on the simulator. This decision is made with the `variational_algo` and `initial_variational_params`. 
After defining the variational algorithm inputs, the circuit settings simulator are defined within `num_qubits`, `initial_amplitudes`, `n_counts` and `circuit`.

Please notice that if your inputs are not coherent with each other, you will get an `AssertionError` or a `ValueError` sooner or later - if you see any of the two, it means there is probably something wrong with the inputs.

Depending on whether you are running the variational algorithm or not, `main()` will call one of the following two procedures:
- `run_variational_algo` runs the variational algorithm in the following steps:
  1. standardize the inputs into settings that are passed to the `BellState` class to initialize the algorithm; 
  2. optimize the parameters;
  3. create a new circuit map with the optimized parameters for the U3 gate;
  4. run the optimized circuit by calling `run_nonvariational_algo`.
- `run_nonvariational_algo` runs the circuit defined in its input `circuit` in the following steps:
  1. standardize the inputs into settings that are passed to the `QSimulator` class to initialize the simulator;
  2. run the circuit;
  3. print measurements results.
  
## q_simulator.py: simulator implementation
The package can be read from top to bottom, where the top classes are the most general classes and include instances of the other "lower" classes as attributes.

They are organized as follows:
- `QSimulator` is the main class of the package and allows to initialize and run the circuit, and to trigger the measurement.
  - `QCircuit` is a class for the quantum circuit. It runs the quantum circuit by applying the gates matrices operators to the state of the system.
    - `QState` represents the combined state of the system. The actual state array is stored into `self.state_arr`. The class comes with some functions that are naturally associated with the state of the system, such as `apply_operator`, `measure` and `get_counts`.
    - `QGate` is designed to represent quantum gates. The main result for this class is the `get_operator(self, num_qubits)` function, returning the matrix operator of the gate. I implemented single-qubit gates, adjacent multi-qubit gates and non-adjacent multi-qubit gates.
    
## variational.py: variational algorithm
The `VariationalAlgorithm` is an abstract class as it is not possible to run a universal variational algorithm. Any variational algorithm can be implemented by inhereting from this class, but `VariationalAlgorithm` cannot be instantiated. The abstract class contains only one abstract method `get_cost`, and it implements all the other methods such as `run_quantum`, `minimize` and `filled_circuit_map`.

The `BellState` class derives from `VariationalAlgorithm` and implements the `get_cost` function by minimizing the counts for `|01>` and `|10>` and equalizing the counts of `|00>` and `|11>`. This is done by the formula:
```
cost = sqrt[counts_01 ** 2 + counts_10 ** 2 + (counts_00 - counts_11) ** 2]
```

## q_settings.py: standardized settings
This package contains all the settings classes for variational algorithm, simulator, circuit, state and gates. It is structured similarly to `q_simulator.py`, meaning that settings are boxed into each other.

## support_functions.py: some useful functions
It contains all the matrices supported by the program, as well as normalization functions, the important `adjacent_qubits` and `get_projection` functions.

