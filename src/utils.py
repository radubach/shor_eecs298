import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate
from typing import List, Union, Tuple

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