use crypto_primitives::crypto_bigint_uint::Uint;
use std::fmt::Debug;

pub trait PrimalityTest<R>: Debug + Clone {
    fn is_probably_prime(candidate: &R) -> bool;
}

/// Baillie–PSW primality test (a base-2 Miller–Rabin round followed by a
/// strong Lucas test, `crypto_primes::is_prime`).
///
/// The name is historical: the original implementation ran a *single*
/// base-2 Miller–Rabin round. That is not enough for a Fiat–Shamir-drawn
/// prime: the candidate stream is under the prover's control (it depends on
/// the committed witness), so a grinding prover could steer it onto a base-2
/// pseudoprime and run the PIOP modulo a composite, capping soundness well
/// below the field size. Baillie–PSW has no known counterexamples and none
/// exist below 2^64, so the projecting prime is prime in practice.
#[derive(Debug, Clone, Copy)]
pub struct MillerRabin {}

impl<const LIMBS: usize> PrimalityTest<Uint<LIMBS>> for MillerRabin {
    fn is_probably_prime(candidate: &Uint<LIMBS>) -> bool {
        crypto_primes::is_prime(crypto_primes::Flavor::Any, candidate.inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U128;

    fn u(v: u128) -> Uint<{ U128::LIMBS }> {
        Uint::new(U128::from_u128(v))
    }

    #[test]
    fn small_primes_and_composites() {
        for p in [2u128, 3, 5, 7, 11, 13, 65537, 4_294_967_311] {
            assert!(MillerRabin::is_probably_prime(&u(p)), "{p} is prime");
        }
        for c in [0u128, 1, 4, 9, 15, 21, 561, 65535, 4_294_967_297] {
            assert!(!MillerRabin::is_probably_prime(&u(c)), "{c} is composite");
        }
    }

    /// Base-2 strong pseudoprimes that a lone base-2 Miller–Rabin round
    /// accepts; Baillie–PSW must reject them.
    #[test]
    fn rejects_base_two_strong_pseudoprimes() {
        for c in [2047u128, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633] {
            assert!(!MillerRabin::is_probably_prime(&u(c)), "{c} is a base-2 pseudoprime");
        }
    }

    /// The Mersenne prime 2^127 − 1 and the secp256k1 base prime.
    #[test]
    fn large_known_primes() {
        assert!(MillerRabin::is_probably_prime(&u((1u128 << 127) - 1)));
        let secp = Uint::new(crypto_bigint::U256::from_be_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
        ));
        assert!(MillerRabin::is_probably_prime(&secp));
    }
}
