#!/usr/bin/env python3
"""
f-clock2.py - 2-phase factorization-based clock
Difficulty blocks added: multiply quasi-semiprime for each digit

Encoding time notation (HH:MM:SS) to integer:
N = (2^H10 * 3^H1 * 5^M10 * 7^M1 * 11^S10 * 13^S1) * (R0 * R1 * R2 * R3 * R4 * R5)

Where each Rj is a difficulty block (quasi-semiprime)

Representation process:
1. Extract p-adic exponents to show digit draft (blurry)
2. Factorize difficulty blocks (Pollard's Rho + Trial Division)
3. Increase clarity for each digit when block is completed
4. Mark as finalized when all blocks are completed

Target time: 0.8~0.9 seconds
"""

import time
from datetime import datetime
from typing import List, Tuple, Dict
import random
import math


# =============================================================================
# Constants
# =============================================================================

# Prime basis: P = {2, 3, 5, 7, 11, 13} (6 digits)
PRIMES = [2, 3, 5, 7, 11, 13]

# Target computation time (seconds)
TARGET_TIME = 0.85  # Middle value of 0.8~0.9 seconds

# Difficulty adjustment parameter
DIFFICULTY_ALPHA = 0.15  # Adaptive difficulty adjustment coefficient

# Difficulty block bit length tiers (per digit)
# Distributed to match TARGET_TIME on average for 6 digits
DIFFICULTY_TIERS = {
    0: 24,  # Very easy (for digit 0)
    1: 26,  # Easy
    2: 28,  # Normal-
    3: 30,  # Normal
    4: 32,  # Normal+
    5: 34,  # Hard
    6: 36,  # Very hard
    7: 38,  # Extremely hard
    8: 40,  # Very extremely hard
    9: 42,  # Maximum difficulty
}


# =============================================================================
# Utility functions: Prime generation and validation
# =============================================================================

def is_prime(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin primality test
    
    Args:
        n: Number to test
        k: Number of iterations (determines accuracy)
    
    Returns:
        True if n is likely prime
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Decompose n - 1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Repeat k times
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def generate_prime(bits: int) -> int:
    """
    Generate a prime number with specified bit length
    
    Args:
        bits: Bit length
    
    Returns:
        Prime number
    """
    while True:
        # Generate random odd number
        n = random.getrandbits(bits)
        n |= (1 << bits - 1) | 1  # Set most significant and least significant bits to 1
        
        if is_prime(n):
            return n


def generate_semiprime(bits: int) -> Tuple[int, int, int]:
    """
    Generate a quasi-semiprime (product of two primes)
    
    Args:
        bits: Target bit length
    
    Returns:
        (semiprime, p, q): Semiprime and its two prime factors
    """
    # Unequally distribute bits between two primes (controls factorization difficulty)
    # Example: 30-bit -> 12-bit * 18-bit
    bits_p = bits // 2 - 2
    bits_q = bits - bits_p
    
    p = generate_prime(bits_p)
    q = generate_prime(bits_q)
    
    # Regenerate if p and q are too close (prevent Fermat attack)
    while abs(p - q) < (1 << (bits // 2 - 4)):
        q = generate_prime(bits_q)
    
    return p * q, p, q


# =============================================================================
# Factorization algorithms
# =============================================================================

def gcd(a: int, b: int) -> int:
    """
    Calculate greatest common divisor using Euclidean algorithm
    
    Args:
        a, b: Two integers
    
    Returns:
        Greatest common divisor
    """
    while b:
        a, b = b, a % b
    return a


def pollard_rho(n: int, max_iterations: int = 100000) -> int:
    """
    Find a factor using Pollard's Rho algorithm
    
    Args:
        n: Number to factorize
        max_iterations: Maximum number of iterations
    
    Returns:
        Found factor (returns n on failure)
    """
    if n == 1:
        return 1
    if n % 2 == 0:
        return 2
    
    # Random starting point
    x = random.randint(2, n - 1)
    y = x
    c = random.randint(1, n - 1)
    d = 1
    
    # f(x) = (x^2 + c) mod n
    for _ in range(max_iterations):
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        
        d = gcd(abs(x - y), n)
        
        if d != 1 and d != n:
            return d
    
    return n


def trial_division(n: int, limit: int = 10000) -> int:
    """
    Find a small factor using trial division
    
    Args:
        n: Number to factorize
        limit: Maximum prime to attempt
    
    Returns:
        Found factor (returns n if none found)
    """
    if n == 1:
        return 1
    
    # Divide by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        if p > limit:
            break
        if n % p == 0:
            return p
    
    # Continue with odd numbers
    divisor = 53
    while divisor * divisor <= n and divisor <= limit:
        if n % divisor == 0:
            return divisor
        divisor += 2
    
    return n


def factorize(n: int, timeout: float = 2.0) -> List[int]:
    """
    Prime factorize an integer
    
    Uses combination of Trial Division + Pollard's Rho
    
    Args:
        n: Number to factorize
        timeout: Maximum allowed time (seconds)
    
    Returns:
        List of prime factors
    """
    if n == 1:
        return []
    
    if is_prime(n):
        return [n]
    
    factors = []
    remaining = n
    start_time = time.time()
    
    # Remove small factors using Trial Division
    while remaining > 1:
        if time.time() - start_time > timeout:
            # On timeout, add remaining number as-is
            factors.append(remaining)
            break
        
        factor = trial_division(remaining, limit=10000)
        
        if factor == remaining:
            # Trial Division failed, try Pollard's Rho
            factor = pollard_rho(remaining, max_iterations=50000)
        
        if factor == remaining:
            # All methods failed, treat as prime
            factors.append(remaining)
            break
        
        factors.append(factor)
        remaining //= factor
    
    return sorted(factors)


# =============================================================================
# Time encoding/decoding
# =============================================================================

def encode_time_to_integer(hour: int, minute: int, second: int, 
                          difficulty_blocks: List[int]) -> int:
    """
    Encode time with difficulty blocks into an integer
    
    N = (2^H10 * 3^H1 * 5^M10 * 7^M1 * 11^S10 * 13^S1) * (R0 * R1 * R2 * R3 * R4 * R5)
    
    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)
        difficulty_blocks: List of 6 difficulty blocks
    
    Returns:
        Encoded integer N
    """
    h10 = hour // 10
    h1 = hour % 10
    m10 = minute // 10
    m1 = minute % 10
    s10 = second // 10
    s1 = second % 10
    
    digits = [h10, h1, m10, m1, s10, s1]
    
    # p-adic exponent part
    N = 1
    for i, prime in enumerate(PRIMES):
        N *= prime ** digits[i]
    
    # Multiply by difficulty blocks
    for block in difficulty_blocks:
        N *= block
    
    return N


def p_adic_valuation(n: int, p: int) -> int:
    """
    Calculate p-adic valuation
    
    v_p(n) = max{k ∈ ℕ: p^k | n}
    
    Args:
        n: Target integer
        p: Prime number
    
    Returns:
        p-adic valuation
    """
    if n == 0:
        return float('inf')
    
    count = 0
    while n % p == 0:
        n //= p
        count += 1
    
    return count


def decode_integer_to_time(N: int) -> Tuple[List[int], int]:
    """
    Decode integer to time digits (extract p-adic valuations only)
    
    Args:
        N: Encoded integer
    
    Returns:
        (digits, cofactor): List of digits and remaining cofactor (product of difficulty blocks)
    """
    digits = []
    remaining = N
    
    for prime in PRIMES:
        val = p_adic_valuation(remaining, prime)
        digits.append(val)
        remaining //= (prime ** val)
    
    return digits, remaining


# =============================================================================
# Difficulty block generation and management
# =============================================================================

class DifficultyController:
    """
    Adaptive difficulty controller
    
    Dynamically adjusts difficulty to converge to TARGET_TIME
    """
    
    def __init__(self, target_time: float = TARGET_TIME, alpha: float = DIFFICULTY_ALPHA):
        """
        Initialize
        
        Args:
            target_time: Target time (seconds)
            alpha: Adjustment coefficient (0 < alpha < 1)
        """
        self.target_time = target_time
        self.alpha = alpha
        self.theta = 1.0  # Difficulty scalar
        self.history = []  # (actual_time, theta) history
    
    def update(self, actual_time: float):
        """
        Adjust difficulty based on actual elapsed time
        
        Θ ← Θ * exp(α * (t - target) / target)
        
        Args:
            actual_time: Actual elapsed time (seconds)
        """
        error = (actual_time - self.target_time) / self.target_time
        adjustment = math.exp(self.alpha * error)
        
        self.theta *= adjustment
        
        # Limit theta range (0.1 ~ 10.0)
        self.theta = max(0.1, min(10.0, self.theta))
        
        self.history.append((actual_time, self.theta))
    
    def get_difficulty_bits(self, digit_value: int) -> int:
        """
        Determine difficulty bit length based on digit value and current theta
        
        Args:
            digit_value: Digit value (0-9)
        
        Returns:
            Bit length
        """
        base_bits = DIFFICULTY_TIERS.get(digit_value, 30)
        adjusted_bits = int(base_bits * self.theta)
        
        # Limit to minimum 20-bit, maximum 50-bit
        return max(20, min(50, adjusted_bits))


def generate_difficulty_blocks(digits: List[int], controller: DifficultyController) -> Tuple[List[int], Dict]:
    """
    Generate difficulty blocks for each digit
    
    Args:
        digits: 6 digits [H10, H1, M10, M1, S10, S1]
        controller: Difficulty controller
    
    Returns:
        (blocks, metadata): List of difficulty blocks and metadata
    """
    blocks = []
    metadata = {
        'blocks': [],
        'theta': controller.theta
    }
    
    for i, digit in enumerate(digits):
        bits = controller.get_difficulty_bits(digit)
        semiprime, p, q = generate_semiprime(bits)
        
        blocks.append(semiprime)
        metadata['blocks'].append({
            'index': i,
            'digit': digit,
            'value': semiprime,
            'bits': semiprime.bit_length(),
            'factors': [p, q]
        })
    
    return blocks, metadata


# =============================================================================
# Display and visualization
# =============================================================================

def format_time_display(digits: List[int], clarity: List[float]) -> str:
    """
    Display time digits according to clarity level
    
    Args:
        digits: [H10, H1, M10, M1, S10, S1]
        clarity: Clarity level for each digit (0.0 = blurry, 1.0 = clear)
    
    Returns:
        "HH:MM:SS" formatted string (with clarity indicators)
    """
    h10, h1, m10, m1, s10, s1 = digits
    c = clarity
    
    # Represent clarity with characters
    # 0.0-0.3: ░ (blurry)
    # 0.3-0.7: ▒ (medium)
    # 0.7-1.0: ▓ (almost clear)
    # 1.0: digit as-is (completely clear)
    
    def display_digit(digit, clear):
        if clear >= 1.0:
            return str(digit)
        elif clear >= 0.7:
            return f"▓{digit}▓"
        elif clear >= 0.3:
            return f"▒{digit}▒"
        else:
            return f"░{digit}░"
    
    time_str = (
        f"{display_digit(h10, c[0])}{display_digit(h1, c[1])}:"
        f"{display_digit(m10, c[2])}{display_digit(m1, c[3])}:"
        f"{display_digit(s10, c[4])}{display_digit(s1, c[5])}"
    )
    
    return time_str


def verify_encoding(N: int, digits: List[int], blocks: List[int]) -> bool:
    """
    Verify encoding
    
    Args:
        N: Original integer
        digits: Recovered digits
        blocks: Difficulty blocks
    
    Returns:
        Verification success or failure
    """
    reconstructed = 1
    for i, prime in enumerate(PRIMES):
        reconstructed *= prime ** digits[i]
    for block in blocks:
        reconstructed *= block
    
    return N == reconstructed


def log_factorization(N: int, digits: List[int], metadata: Dict, 
                     factorization_results: List[Tuple[int, List[int], float]]):
    """
    Print factorization log
    
    Args:
        N: Original integer
        digits: Digits
        metadata: Difficulty block metadata
        factorization_results: [(block_value, factors, time), ...]
    """
    print("\n" + "="*70)
    print("FACTORIZATION LOG")
    print("="*70)
    print(f"Original N: {N}")
    print(f"Bit length: {N.bit_length()} bits")
    print(f"\n1) p-adic exponents (digits):")
    
    for i, (prime, digit) in enumerate(zip(PRIMES, digits)):
        if digit > 0:
            print(f"   {prime}^{digit} = {prime ** digit}")
    
    print(f"\n2) Difficulty blocks:")
    for i, block_info in enumerate(metadata['blocks']):
        print(f"   R{i} (digit={block_info['digit']}): {block_info['value']}")
        print(f"      Bits: {block_info['bits']}, "
              f"Factors: {block_info['factors'][0]} × {block_info['factors'][1]}")
    
    print(f"\n3) Factorization results:")
    total_time = 0
    for i, (value, factors, elapsed) in enumerate(factorization_results):
        print(f"   R{i}: {factors} (took {elapsed*1000:.2f}ms)")
        total_time += elapsed
    
    print(f"\n4) Total factorization time: {total_time:.4f}s")
    
    # Verification
    blocks = [info['value'] for info in metadata['blocks']]
    is_valid = verify_encoding(N, digits, blocks)
    print(f"\n5) Verification: {'✓ PASS' if is_valid else '✗ FAIL'}")
    print("="*70 + "\n")


# =============================================================================
# Main clock display
# =============================================================================

def display_clock():
    """
    Main clock display loop
    """
    print("="*70)
    print("F-CLOCK 2: Factorization-based Clock (Phase 2)")
    print("With difficulty blocks (semi-primes)")
    print(f"Target time: {TARGET_TIME:.2f}s per factorization")
    print("="*70)
    print("\nPress Ctrl+C to exit\n")
    
    controller = DifficultyController()
    last_second = -1
    
    try:
        while True:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            second = now.second
            
            # Encode/decode whenever second changes
            if second != last_second:
                print(f"\n{'='*70}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] New second")
                print(f"{'='*70}")
                
                # Extract digits
                h10, h1 = hour // 10, hour % 10
                m10, m1 = minute // 10, minute % 10
                s10, s1 = second // 10, second % 10
                digits = [h10, h1, m10, m1, s10, s1]
                
                # Generate difficulty blocks
                print(f"\n[1] Generating difficulty blocks (θ={controller.theta:.3f})...")
                blocks, metadata = generate_difficulty_blocks(digits, controller)
                
                # Encode time
                print(f"[2] Encoding time...")
                N = encode_time_to_integer(hour, minute, second, blocks)
                print(f"    N = {N}")
                print(f"    Bit length: {N.bit_length()} bits")
                
                # Phase 1: Extract p-adic exponents (fast)
                print(f"\n[3] Phase 1: Extracting p-adic valuations...")
                start_phase1 = time.time()
                decoded_digits, cofactor = decode_integer_to_time(N)
                phase1_time = time.time() - start_phase1
                
                # Draft display (all blurry)
                clarity = [0.0] * 6
                draft_display = format_time_display(decoded_digits, clarity)
                print(f"    -> {draft_display}")
                print(f"    (took {phase1_time*1000:.2f}ms)")
                
                # Phase 2: Factorize difficulty blocks (slow)
                print(f"\n[4] Phase 2: Factorizing difficulty blocks...")
                print(f"    Cofactor to factorize: {cofactor}")
                print(f"    (Product of {len(blocks)} semi-primes)")
                
                start_phase2 = time.time()
                factorization_results = []
                
                # Factorize each block sequentially
                remaining_cofactor = cofactor
                for i, block_info in enumerate(metadata['blocks']):
                    block_value = block_info['value']
                    
                    print(f"\n    Factorizing R{i} ({block_value.bit_length()} bits)...")
                    block_start = time.time()
                    
                    # Factorize
                    factors = factorize(block_value, timeout=2.0)
                    block_time = time.time() - block_start
                    
                    factorization_results.append((block_value, factors, block_time))
                    
                    # Increase clarity
                    clarity[i] = 0.5
                    progress_display = format_time_display(decoded_digits, clarity)
                    print(f"       Factors: {factors}")
                    print(f"       Time: {block_time*1000:.2f}ms")
                    print(f"       -> {progress_display}")
                    
                    # Remove from remaining_cofactor
                    for factor in factors:
                        if remaining_cofactor % factor == 0:
                            remaining_cofactor //= factor
                    
                    # Fully finalize
                    clarity[i] = 1.0
                
                phase2_time = time.time() - start_phase2
                
                # Final display
                final_display = format_time_display(decoded_digits, clarity)
                print(f"\n[5] Phase 2 Complete!")
                print(f"    -> {final_display}")
                print(f"    Total factorization time: {phase2_time:.4f}s")
                print(f"    Target time: {TARGET_TIME:.2f}s")
                print(f"    Deviation: {(phase2_time - TARGET_TIME)*100/TARGET_TIME:+.1f}%")
                
                # Detailed log
                log_factorization(N, decoded_digits, metadata, factorization_results)
                
                # Adjust difficulty
                controller.update(phase2_time)
                print(f"[6] Difficulty adjusted: θ = {controller.theta:.4f}")
                
                last_second = second
            
            # Wait 0.1 seconds
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nClock stopped.")
        print(f"\nDifficulty history (last 10):")
        for i, (t, theta) in enumerate(controller.history[-10:]):
            print(f"  {i+1}. Time: {t:.4f}s, θ: {theta:.4f}")


# =============================================================================
# Tests
# =============================================================================

def test_factorization():
    """
    Test factorization algorithms
    """
    print("="*70)
    print("TESTING FACTORIZATION ALGORITHMS")
    print("="*70)
    
    test_cases = [
        (24, "24-bit semi-prime"),
        (28, "28-bit semi-prime"),
        (32, "32-bit semi-prime"),
        (36, "36-bit semi-prime"),
    ]
    
    for bits, description in test_cases:
        print(f"\n{description}:")
        semiprime, p, q = generate_semiprime(bits)
        print(f"  Value: {semiprime}")
        print(f"  Actual bits: {semiprime.bit_length()}")
        print(f"  True factors: {p} × {q}")
        
        start = time.time()
        factors = factorize(semiprime, timeout=5.0)
        elapsed = time.time() - start
        
        print(f"  Found factors: {factors}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Correct: {set(factors) == {p, q}}")
    
    print("\n" + "="*70 + "\n")


def test_encoding():
    """
    Test encoding/decoding
    """
    print("="*70)
    print("TESTING ENCODING/DECODING WITH DIFFICULTY BLOCKS")
    print("="*70)
    
    controller = DifficultyController()
    
    test_cases = [
        (12, 34, 56),
        (0, 0, 0),
        (23, 59, 59),
    ]
    
    for hour, minute, second in test_cases:
        print(f"\nTest: {hour:02d}:{minute:02d}:{second:02d}")
        
        h10, h1 = hour // 10, hour % 10
        m10, m1 = minute // 10, minute % 10
        s10, s1 = second // 10, second % 10
        digits = [h10, h1, m10, m1, s10, s1]
        
        # Generate difficulty blocks
        blocks, metadata = generate_difficulty_blocks(digits, controller)
        
        # Encode
        N = encode_time_to_integer(hour, minute, second, blocks)
        print(f"  N bits: {N.bit_length()}")
        
        # Decode
        decoded_digits, cofactor = decode_integer_to_time(N)
        print(f"  Decoded: {decoded_digits}")
        print(f"  Cofactor bits: {cofactor.bit_length()}")
        
        # Verify
        is_valid = verify_encoding(N, decoded_digits, blocks)
        print(f"  Valid: {is_valid}")
        
        assert is_valid
        assert decoded_digits == digits
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70 + "\n")


def main():
    """
    Main function
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_encoding()
        elif sys.argv[1] == '--test-factor':
            test_factorization()
        else:
            print("Usage: python f-clock2.py [--test | --test-factor]")
    else:
        display_clock()


if __name__ == "__main__":
    main()
