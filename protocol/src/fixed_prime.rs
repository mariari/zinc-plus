//! Fixed projecting primes.
//!
//! In Zinc+, the projecting prime `q` for the
//! `\phi_q : Z[X] -> F_q[X]` step (Step 1 of the protocol) is drawn from
//! the Fiat–Shamir transcript — that is the sound, general behaviour and
//! the default (`ZincTypes::FIXED_PROJECTING_PRIME = None`). A type bundle
//! may instead pin a fixed prime by setting `FIXED_PROJECTING_PRIME =
//! Some(bytes)`; the SHA+ECDSA demo pins the **secp256k1 base field prime**
//! `p = 2^256 − 2^32 − 977`
//!   `= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`
//! because its EC constraints are secp256k1-specific algebraic identities
//! that only hold modulo that prime.
//!
//! Soundness: a fixed projecting prime is in general NOT sound — a
//! constraint-grinding adversary can craft witnesses whose ideal-check
//! residues vanish mod a known `q`. For the targeted SHA+ECDSA proving
//! application this does not break soundness (the relevant constraints
//! are honest mod `p`). Do not pin a prime for other applications without
//! re-doing the soundness analysis.

use crypto_primitives::{ConstIntSemiring, PrimeField};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::from_ref::FromRef;

/// secp256k1 base field prime, little-endian byte order (32 bytes).
///
/// `Uint<LIMBS>::read_transcription_bytes_exact` interprets its input
/// as little-endian limb chunks (see `transcript::traits` impl), so we
/// store the prime in that same order.
pub const SECP256K1_P_LE_BYTES: [u8; 32] = [
    0x2F, 0xFC, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// Build `F::Config` from a fixed prime given as little-endian
/// transcription bytes (the `FMod` encoding), installing it as the
/// projecting field modulus.
///
/// Panics if `bytes.len()` differs from `FMod::NUM_BYTES` — a pinned
/// prime must match the instantiated `Fmod` width exactly.
pub fn fixed_field_cfg_from_bytes<F, FMod>(bytes: &[u8]) -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
{
    assert_eq!(
        FMod::NUM_BYTES,
        bytes.len(),
        "FIXED_PROJECTING_PRIME has {} bytes but Fmod is {} bytes wide",
        bytes.len(),
        FMod::NUM_BYTES,
    );
    let prime = FMod::read_transcription_bytes_exact(bytes);
    F::make_cfg(&F::Modulus::from_ref(&prime)).expect("the fixed projecting modulus is prime")
}

/// Build `F::Config` from the secp256k1 base prime.
///
/// Panics if `FMod` cannot hold a 256-bit value (its `NUM_BYTES` differs
/// from `SECP256K1_P_LE_BYTES.len()`).
pub fn secp256k1_field_cfg<F, FMod>() -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable,
    F::Modulus: FromRef<FMod>,
{
    fixed_field_cfg_from_bytes::<F, FMod>(&SECP256K1_P_LE_BYTES)
}

/// Step-1 projecting-field selection shared by every prover/verifier
/// path: a pinned prime when `fixed` is `Some`, otherwise a fresh prime
/// drawn from the Fiat–Shamir transcript (whose state at this point
/// already covers the witness commitments and the public columns, so the
/// prime is a function of the committed witness, as soundness requires).
pub fn projecting_field_cfg<F, FMod, PrimeTest, T>(
    fixed: Option<&[u8]>,
    transcript: &mut T,
) -> F::Config
where
    F: PrimeField,
    FMod: ConstTranscribable + ConstIntSemiring,
    F::Modulus: FromRef<FMod>,
    PrimeTest: PrimalityTest<FMod>,
    T: Transcript,
{
    match fixed {
        Some(bytes) => fixed_field_cfg_from_bytes::<F, FMod>(bytes),
        None => transcript.get_random_field_cfg::<F, FMod, PrimeTest>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint};
    use zinc_transcript::traits::GenTranscribable;

    /// `SECP256K1_P_LE_BYTES` decodes to the well-known secp256k1 base
    /// prime when read through the transcription convention used by
    /// `Uint<LIMBS>` (little-endian limb chunks).
    #[test]
    fn secp256k1_p_le_bytes_decode_to_prime() {
        let prime = Uint::<4>::read_transcription_bytes_exact(&SECP256K1_P_LE_BYTES);
        assert_eq!(
            prime.as_words(),
            &[
                0xFFFF_FFFE_FFFF_FC2F,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
            ],
        );
    }

    /// Construction succeeds for the concrete `MontyField<4>` / `Uint<4>`
    /// combination used by the e2e bench.
    #[test]
    fn secp256k1_field_cfg_constructs() {
        let _cfg = secp256k1_field_cfg::<MontyField<4>, Uint<4>>();
    }
}
