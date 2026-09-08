pub mod traits;

use crate::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crypto_primitives::{ConstIntSemiring, PrimeField};
use zinc_primality::PrimalityTest;
use zinc_utils::add;

/// A cryptographic transcript implementation using the BLAKE3 hash
/// function. Used for Fiat-Shamir transformations in zero-knowledge proof
/// systems.
#[derive(Debug, Clone)]
pub struct Blake3Transcript {
    /// The underlying BLAKE3 hasher that maintains the transcript state.
    hasher: blake3::Hasher,
}

impl Default for Blake3Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Blake3Transcript {
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    /// Generates a specified number of pseudorandom bytes based on the current
    /// transcript state. Uses a counter-based approach to generate enough
    /// bytes from the hasher.
    ///
    /// Note that this does NOT update the internal state of the hasher
    #[allow(clippy::arithmetic_side_effects)]
    fn fill_with_random_bytes(&mut self, buf: &mut [u8]) {
        self.hasher.finalize_xof().fill(buf);
    }

}

impl Transcript for Blake3Transcript {
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.fill_with_random_bytes(&mut buf);
        self.hasher.update(&[0x12]);
        self.hasher.update(&buf);
        self.hasher.update(&[0x34]);
        T::read_transcription_bytes_exact(&buf)
    }

    /// Draws a prime of exactly `8 · R::NUM_BYTES` bits from the transcript:
    /// candidates are uniform random `R` values with the most-significant
    /// bit forced to 1 (so the prime has the full bit length and the
    /// soundness accounting can use `2^(bits-1)` as a lower bound) and the
    /// least-significant bit forced to 1, tested with `T` until one passes.
    /// The random bytes are absorbed as they are drawn, so prover and
    /// verifier stay in lockstep regardless of how many candidates fail.
    #[allow(clippy::arithmetic_side_effects)]
    fn get_prime<R: ConstIntSemiring + ConstTranscribable, T: PrimalityTest<R>>(&mut self) -> R {
        let buf = &mut vec![0u8; R::NUM_BYTES];
        loop {
            self.fill_with_random_bytes(buf);
            // `R` is transcribed as little-endian limbs, so the last byte
            // holds the most-significant bits and the first byte the
            // least-significant ones.
            if let Some(last) = buf.last_mut() {
                *last |= 0x80;
            }
            buf[0] |= 0x01;
            self.absorb_inner(buf);
            let prime_candidate: R = R::read_transcription_bytes_exact(buf);
            if T::is_probably_prime(&prime_candidate) {
                return prime_candidate;
            }
        }
    }

    fn absorb_inner(&mut self, v: &[u8]) {
        self.hasher.update(v);
    }
}

pub fn read_field_cfg<F>(bytes: &[u8]) -> F::Config
where
    F: PrimeField,
    F::Modulus: ConstTranscribable,
{
    let mod_size = F::Modulus::NUM_BYTES;
    let modulus = F::Modulus::read_transcription_bytes_exact(&bytes[..mod_size]);
    F::make_cfg(&modulus).expect("valid field modulus in proof transcription")
}

pub fn read_field_vec_with_cfg<F>(bytes: &[u8], field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    let inner_size = F::Inner::NUM_BYTES;
    bytes
        .chunks_exact(inner_size)
        .map(F::Inner::read_transcription_bytes_exact)
        .map(|inner| F::new_unchecked_with_cfg(inner, field_cfg))
        .collect()
}

pub fn append_field_cfg<'a, F>(buf: &'a mut [u8], modulus: &F::Modulus) -> &'a mut [u8]
where
    F: PrimeField,
    F::Modulus: ConstTranscribable,
{
    let mod_size = F::Modulus::NUM_BYTES;
    let (buf, rest) = buf.split_at_mut(mod_size);
    modulus.write_transcription_bytes_exact(buf);
    rest
}

pub fn append_field_vec_inner<'a, F>(buf: &'a mut [u8], slice: &[F]) -> &'a mut [u8]
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    let inner_size = F::Inner::NUM_BYTES;
    let mut offset = 0;
    for elem in slice {
        let offset_end = add!(offset, inner_size);
        elem.inner()
            .write_transcription_bytes_exact(&mut buf[offset..offset_end]);
        offset = offset_end;
    }
    &mut buf[offset..]
}
