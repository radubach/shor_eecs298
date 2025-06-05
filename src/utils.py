import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate, QFT
from typing import List, Union, Tuple
from math import floor, log
from fractions import Fraction
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error



def modmul_matrix(a: int, power: int, N: int = 15) -> np.ndarray:
    """
    Creates a matrix representation of the modular multiplication operation a^power mod N.
    
    Args:
        a (int): The base number for modular multiplication
        power (int): The power to raise a to
        N (int, optional): The modulus. Defaults to 15.
    
    Returns:
        np.ndarray: A 2D numpy array representing the unitary matrix for modular multiplication
    """
    dim = 16  # 4 qubits
    U = np.zeros((dim, dim))
    for y in range(dim):
        if y >= N:
            # leave y unchanged outside mod-N range
            U[y][y] = 1
        else:
            target = pow(a, power, N) * y % N
            U[target][y] = 1
    return U

def create_unitary_gate(a: int, power: int, N: int = 15) -> UnitaryGate:
    """
    Creates a Qiskit UnitaryGate from the modular multiplication matrix.
    
    Args:
        a (int): The base number for modular multiplication
        power (int): The power to raise a to
        N (int, optional): The modulus. Defaults to 15.
    
    Returns:
        UnitaryGate: A Qiskit UnitaryGate representing the modular multiplication operation
    """
    U = modmul_matrix(a, power, N)
    return UnitaryGate(U)


#########################################################
# Note that this is not used in the final implementation
# function uses non-standard convention for qubit ordering
# standard convention is to have least significant qubit as the first one
# #########################################################
def c_amod15(a: int, power: int) -> UnitaryGate:
    """
    Creates a controlled modular multiplication gate for N=15.
    This is a specialized implementation for N=15 that uses SWAP and X gates.
    
    Args:
        a (int): The base number for modular multiplication (must be in [2,7,8,11,13])
        power (int): The power to raise a to
    
    Returns:
        UnitaryGate: A controlled unitary gate implementing the modular multiplication
    
    Raises:
        ValueError: If a is not in the allowed set [2,7,8,11,13]
    """
    if a not in [2,7,8,11,13]:
        raise ValueError("a must be one of [2,7,8,11,13]")
    
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        
        if a == 11:
            U.swap(2,3)
            U.swap(0,2)

        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U

#########################################################
# Note that this is not used in the final implementation
# same as c_amod15 but uses standard qubit ordering
# and only applies to a = 7
# #########################################################
def c_7mod15(power: int) -> UnitaryGate:
    """
    Creates a controlled modular multiplication gate for a = 7, N=15.
    This is a specialized implementation for N=15 that uses SWAP and X gates.
    
    Args:
        power (int): The power to raise 7 to
    
    Returns:
        UnitaryGate: A controlled unitary gate implementing the modular multiplication
    
    """
    
    U = QuantumCircuit(4)
    for iteration in range(power):
        U.swap(1,0)
        U.swap(2,1)
        U.swap(3,2)
        for q in range(4):
            U.x(q)
    U = U.to_gate()
    U.name = f"7^{power} mod 15"
    c_U = U.control()
    return c_U


def qft_dagger(n: int) -> QuantumCircuit:
    """
    Creates the inverse Quantum Fourier Transform (QFT) circuit.
    
    Args:
        n (int): Number of qubits in the circuit
    
    Returns:
        QuantumCircuit: A quantum circuit implementing the inverse QFT
    """
    qc = QuantumCircuit(n)

    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def qpe_mod15(a: int) -> float:
    """
    Performs Quantum Phase Estimation (QPE) for modular multiplication by a mod 15.
    
    Args:
        a (int): The base number for modular multiplication
    
    Returns:
        float: The estimated phase
    """
    n_count = 8
    qc = QuantumCircuit(4+n_count, n_count)
    
    # Initialize counting qubits
    for q in range(n_count):
        qc.h(q)
    
    # Initialize target qubit
    qc.x(3+n_count)
    
    # Apply controlled modular multiplication
    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), [q]+[i+n_count for i in range(4)])
    
    # Apply inverse QFT
    qc.append(qft_dagger(n_count), range(n_count))
    
    # Measure
    qc.measure(range(n_count), range(n_count))

    # Run the circuit
    # Run the circuit
    aer_sim = AerSimulator()
    t_qc = transpile(qc, aer_sim)
    q_obj = aer_sim.run(t_qc, shots=1, memory=True)
    result = q_obj.result()

    # Get and process results
    readings = result.get_memory()
    print(f"Register Reading: {readings[0]}")
    
    phase = int(readings[0], 2)/(2**n_count)
    print(f"Corresponding phase: {phase}")
    return phase

def find_period(a: int, N: int = 15) -> int:
    """
    Finds the period of the function f(x) = a^x mod N using quantum phase estimation.
    
    Args:
        a (int): The base number
        N (int, optional): The modulus. Defaults to 15.
    
    Returns:
        int: The period of the function
    """
    phase = qpe_mod15(a)
    # Convert phase to fraction
    from fractions import Fraction
    frac = Fraction(phase).limit_denominator(N)
    return frac.denominator

def find_factors(N: int = 15) -> Tuple[int, int]:
    """
    Attempts to find factors of N using Shor's algorithm.
    
    Args:
        N (int, optional): The number to factor. Defaults to 15.
    
    Returns:
        Tuple[int, int]: A tuple containing the two factors found
    
    Raises:
        ValueError: If N is not a composite number
    """
    if N % 2 == 0:
        return (2, N//2)
    
    # Try different values of a
    for a in [2, 3, 4, 5, 7, 8, 11, 13]:
        if a >= N:
            continue
            
        # Find period
        r = find_period(a, N)
        
        # Check if period is valid
        if r % 2 == 0:
            x = pow(a, r//2, N)
            if x != N-1:
                factor1 = np.gcd(x+1, N)
                factor2 = np.gcd(x-1, N)
                if factor1 != 1 and factor2 != 1:
                    return (factor1, factor2)
    
    raise ValueError(f"Could not find factors for {N}")




#########################################################
# second iteration
#########################################################

def mod_mult_gate(b: int, N: int) -> UnitaryGate:
    """
    Creates a unitary gate that implements modular multiplication by b modulo N.
    
    This function creates a quantum gate that performs the operation |x⟩ → |b*x mod N⟩.
    The gate is constructed as a unitary matrix where each column represents the
    transformation of a basis state.
    
    Args:
        b (int): The multiplier in the modular multiplication
        N (int): The modulus for the operation
        n (int): The number of qubits in the target register
                   needed to represent all possible numbers in 0 to N-1 
        U (np.ndarray): The unitary matrix representing the modular multiplication
    
    Returns:
        UnitaryGate: A Qiskit UnitaryGate implementing the modular multiplication
    
    Raises:
        ValueError: If the generated matrix is not unitary
    """
    n = floor(log(N - 1, 2)) + 1  # determine number of qubits in output register
    U = np.zeros((2 ** n, 2 ** n))  # initialize the unitary n x n matrix
    for x in range(N):
        U[b * x % N][x] = 1 # apply the modular multiplication to each basis state
    for x in range(N, 2 ** n):
        U[x][x] = 1 # if the target register is larger than N, set the extras to the identity
    if not np.allclose(U.conj().T @ U, np.eye(U.shape[0])):
        raise ValueError(f"Generated U matrix for b={b}, N={N} is not unitary!")
    return UnitaryGate(U, label=f"M_{b}")

def order_finding_circuit(a: int, N: int) -> QuantumCircuit:
    """
    Creates a quantum circuit for finding the order of a modulo N.
    
    This circuit implements the quantum phase estimation algorithm to find
    the order of a modulo N. It uses a control register of size 2n and a target
    register of size n, where n is the number of bits needed to represent N-1.
    
    The circuit:
    1. Initializes the target register to |1⟩
    2. Applies Hadamard gates to the control register
    3. Applies controlled modular multiplication gates
    4. Applies inverse QFT
    5. Measures the control register
    
    Args:
        a (int): The base number whose order we want to find
        N (int): The modulus
    
    Returns:
        QuantumCircuit: A quantum circuit that can be used to find the order of a modulo N
    """
    n = floor(log(N - 1, 2)) + 1
    m = 2 * n
    control = QuantumRegister(m, name="X")
    target = QuantumRegister(n, name="Y")
    output = ClassicalRegister(m, name="Z")
    qc = QuantumCircuit(control, target, output)
    qc.x(target[0])  # |1> in target register

    for k, qubit in enumerate(control):
        qc.h(qubit)
        b = pow(a, 2 ** k, N)
        qc.compose(mod_mult_gate(b, N).control(), [qubit] + list(target), inplace=True)

    qc.compose(QFT(m, inverse=True), qubits=control, inplace=True)
    qc.measure(control, output)
    return qc

def find_order_quantum(a: int, N: int) -> Tuple[int, QuantumCircuit]:
    """
    Finds the order of a modulo N using quantum computation.
    
    This function runs the order finding circuit multiple times until it finds
    a valid order. The order is the smallest positive integer r such that
    a^r ≡ 1 (mod N).
    
    Args:
        a (int): The base number whose order we want to find
        N (int): The modulus
    
    Returns:
        Tuple[int, QuantumCircuit]: A tuple containing:
            - The order r of a modulo N
            - The quantum circuit used to find the order
    
    Note:
        This function may need to be run multiple times to find a valid order,
        as quantum measurements are probabilistic.
    """
    n = floor(log(N - 1, 2)) + 1
    m = 2 * n
    qc = order_finding_circuit(a, N)
    transpiled = transpile(qc, AerSimulator())
    while True:
        result = AerSimulator().run(transpiled, shots=1, memory=True).result()
        y = int(result.get_memory()[0], 2)
        r = Fraction(y / 2 ** m).limit_denominator(N).denominator
        if r == 0:
            continue
        if pow(a, r, N) == 1:
            return r, qc
        

#########################################################
# Custom Noise model, 
# would use built in FakeBackend, but IBM will sunset it soon
#########################################################

def make_custom_noise_model(one_qubit_error=0.001, two_qubit_error=0.01):
    """
    Create a custom noise model with configurable error rates
    
    Args:
        one_qubit_error (float): Error rate for single-qubit gates (default: 0.001 = 0.1%)
        two_qubit_error (float): Error rate for two-qubit gates (default: 0.01 = 1%)
    
    Returns:
        NoiseModel: Configured noise model for quantum simulation
    
    Example usage:
        # Conservative (default)
        noise = make_custom_noise_model()
        
        # Optimistic
        noise = make_custom_noise_model(one_qubit_error=0.0001, two_qubit_error=0.005)
        
        # Pessimistic  
        noise = make_custom_noise_model(one_qubit_error=0.002, two_qubit_error=0.02)
        
        # Near-perfect (for testing)
        noise = make_custom_noise_model(one_qubit_error=0.00001, two_qubit_error=0.0001)
    """
    noise_model = NoiseModel()

    # Depolarizing noise on 1-qubit gates
    one_q_error = depolarizing_error(one_qubit_error, 1)
    
    # Depolarizing noise on 2-qubit gates  
    two_q_error = depolarizing_error(two_qubit_error, 2)

    # Add to common gates used in quantum circuits
    noise_model.add_all_qubit_quantum_error(one_q_error, ['x', 'h', 'rz', 'sx'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    # Print the configuration for reference
    print(f"Noise Model Configuration:")
    print(f"  1-qubit gates: {one_qubit_error*100:.3f}% error rate")
    print(f"  2-qubit gates: {two_qubit_error*100:.3f}% error rate")

    return noise_model

# Convenience functions for common configurations
def conservative_noise():
    """Conservative noise model (current average hardware)"""
    return make_custom_noise_model(0.001, 0.01)

def optimistic_noise():
    """Optimistic noise model (best current hardware)"""
    return make_custom_noise_model(0.0005, 0.005)

def pessimistic_noise():
    """Pessimistic noise model (older/noisier hardware)"""
    return make_custom_noise_model(0.002, 0.02)

def near_perfect_noise():
    """Near-perfect noise model (future target)"""
    return make_custom_noise_model(0.0001, 0.001)

def your_original_noise():
    """Your original noise model settings"""
    return make_custom_noise_model(0.0001, 0.005)


def make_advanced_noise_model():
    noise_model = NoiseModel()
    
    # Relaxation errors: T1 and T2 times (in microseconds)
    t1 = 100e3  # 100 μs
    t2 = 80e3   # 80 μs
    gate_time_1q = 100  # ns
    gate_time_2q = 300  # ns

    # Thermal relaxation errors
    relax_error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    relax_error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(thermal_relaxation_error(t1, t2, gate_time_2q))

    # Combine depolarizing + relaxation
    one_q_error = depolarizing_error(0.01, 1).compose(relax_error_1q)
    two_q_error = depolarizing_error(0.03, 2).compose(relax_error_2q)

    noise_model.add_all_qubit_quantum_error(one_q_error, ['x', 'h', 'rz', 'sx'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model