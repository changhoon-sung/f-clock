"""
Time Encoder: Encode HHMMSS as products of large primes
"""

import argparse
import random
import time
from datetime import datetime


SEED = 42
random.seed(SEED)

def get_first_n_primes(n):
    """Return the first n prime numbers"""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


# Generate prime pool once
PRIME_POOL = get_first_n_primes(70)


def encoder(digit, target_bits=128):
    """
    Select 'digit' random primes and adjust exponents to
    generate a number of approximately target_bits size
    
    Args:
        digit: Integer from 0-9
        target_bits: Target bit length of the result
    
    Returns:
        Large composite number with approximately target_bits length
    """

    digit = digit + 1
    
    # Select 'digit' primes from the pool
    selected_primes = random.sample(PRIME_POOL, digit)
    
    # Calculate approximate bit contribution per prime
    bits_per_prime = target_bits / digit
    
    # Set initial exponents
    exponents = []
    for prime in selected_primes:
        prime_bits = prime.bit_length()
        exponent = max(1, int(bits_per_prime / prime_bits))
        exponents.append(exponent)
    
    # Calculate current bit length
    result = 1
    for prime, exp in zip(selected_primes, exponents):
        result *= prime ** exp
    
    # Fine-tune to target bits
    adjustment_idx = 0
    
    while result.bit_length() < target_bits:
        exponents[adjustment_idx] += 1
        result = 1
        for prime, exp in zip(selected_primes, exponents):
            result *= prime ** exp
    
    # Reduce if too large
    while result.bit_length() > target_bits + 10:
        if exponents[adjustment_idx] > 1:
            exponents[adjustment_idx] -= 1
            result = 1
            for prime, exp in zip(selected_primes, exponents):
                result *= prime ** exp
        else:
            break
    
    return result


def factorize(n):
    """
    Prime factorization of a given number
    
    Args:
        n: Integer to factorize
    
    Returns:
        Dictionary {prime: exponent}
    """
    if n <= 1:
        return {}
    
    factors = {}
    
    # Divide by primes in PRIME_POOL
    for prime in PRIME_POOL:
        if prime * prime > n:
            break
        
        exponent = 0
        while n % prime == 0:
            n //= prime
            exponent += 1
        
        if exponent > 0:
            factors[prime] = exponent
    
    # If remaining number > 1, it's prime itself
    if n > 1:
        factors[n] = 1
    
    return factors


def decoder(encoded_value):
    """
    Factorize the encoded value to restore the original digit
    
    Args:
        encoded_value: Large number returned by encoder()
    
    Returns:
        (Original digit (0-9), Dictionary {prime: exponent})
    """
    if encoded_value == 1:
        return 0, {}
    
    # Prime factorization
    factors = factorize(encoded_value)
    
    # Number of prime factors in PRIME_POOL = digit
    digit = sum(1 for prime in factors.keys() if prime in PRIME_POOL)

    digit = digit - 1
    return digit, factors


def main():
    """Encode and output current time every second"""
    parser = argparse.ArgumentParser(description="Time Encoder: Encode HHMMSS as products of large primes")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed encoding/decoding information")
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Prime pool generated: {len(PRIME_POOL)} primes (2 ~ {PRIME_POOL[-1]})")
        print("Encoding time every second... (Ctrl+C to exit)\n")
    
    labels = ["H10", "H1", "M10", "M1", "S10", "S1"]
    last_time_str = ""
    
    try:
        while True:
            begin = time.time()

            now = datetime.now()
            time_str = now.strftime("%H%M%S")
            
            # Only output when time changes
            if time_str != last_time_str:
                last_time_str = time_str
                
                # Encoding
                original_digits = [int(d) for d in time_str]
                encoded_values = []
                
                for i, digit in enumerate(original_digits):
                    encoded = encoder(digit, target_bits=128)
                    encoded_values.append(encoded)
                
                # Decoding
                if args.verbose:
                    print(f"\n{'='*70}")
                    print(f"Time: {now.strftime('%H:%M:%S')}")
                    print(f"{'='*70}")
                    
                    for i, (digit, encoded) in enumerate(zip(original_digits, encoded_values)):
                        print(f"{labels[i].ljust(4)} (digit={digit})")
                        print(f"  → Bits: {encoded.bit_length()}")
                        print(f"  → Value: {encoded}")
                    
                    # Decoding
                    print(f"\n{'─'*70}")
                    decoded_digits = []
                    for i, encoded in enumerate(encoded_values):
                        decoded, factors = decoder(encoded)
                        decoded_digits.append(decoded)
                        
                        status = "✓" if decoded == original_digits[i] else "✗"
                        print(f"{status.ljust(2)} {labels[i].ljust(4)}: {original_digits[i]} → encoded → decoded: {decoded}, factors: {factors}")
                    
                    # Verification results
                    print(f"{'─'*70}")
                    if original_digits == decoded_digits:
                        print("✓ Decoding successful!")
                    else:
                        print("✗ Decoding failed")
                        print(f"  Original: {original_digits}")
                        print(f"  Restored: {decoded_digits}")
                    
                    end = time.time()
                    print(f"Elapsed time: {(end - begin)*1000:.2f}ms")
                    print(f"{'='*70}\n")
                else:
                    # Default mode: print encoded numbers line by line
                    for encoded in encoded_values:
                        print(encoded)
                    print()  # Separate with blank line
                    print(f"{'-'*70}\n")
            
            # Check every 0.1 seconds
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        if args.verbose:
            print("\n\nExiting. Bye!")
        else:
            print()


if __name__ == "__main__":
    main()

