//! Small per-call cache for UAIR scalar projections.
//!
//! A UAIR's `constrain_general` is called many times per proof — once per
//! `comb_fn` evaluation in the combined sumcheck, and once per trace row in
//! the ideal-check combined-polynomial builder. Within each call, the same
//! scalar constant (e.g. an `n`, `p`, or `X` [`DensePolynomial`]) is
//! typically passed to `mbs` / `from_ref` many times by reference to a
//! local. Hashing the full scalar key to look up the projected value is
//! expensive (~1 µs for a 1.3 KB `DensePolynomial<_, 32>` key), and pointer
//! identity is a cheap proxy that catches the common case.
//!
//! [`ScalarProjCache`] is stack-allocated with a small fixed capacity so it
//! costs ~nothing for UAIRs with 0–1 `mbs` sites, and amortizes the HashMap
//! lookup across many reuses for wide UAIRs (SHA-256, ECDSA). If the UAIR
//! uses more distinct scalars than [`SCALAR_PROJ_CACHE_CAP`], the extras
//! simply fall through to the HashMap — correctness is unaffected, the
//! extras just don't benefit from the cache.
//!
//! The value slots use `MaybeUninit` so construction does not touch them —
//! this matters because UAIRs with no `mbs`/`from_ref` calls (e.g. the
//! binary-decomposition test UAIR) would otherwise pay ~100 ns of zeroing
//! per `comb_fn` call for a cache they never use.
//!
//! [`DensePolynomial`]: zinc_poly::univariate::dense::DensePolynomial

use std::mem::MaybeUninit;

pub const SCALAR_PROJ_CACHE_CAP: usize = 8;

pub struct ScalarProjCache<S, V> {
    ptrs: [*const S; SCALAR_PROJ_CACHE_CAP],
    vals: [MaybeUninit<V>; SCALAR_PROJ_CACHE_CAP],
    len: usize,
}

impl<S, V: Clone> ScalarProjCache<S, V> {
    pub fn new() -> Self {
        Self {
            ptrs: [std::ptr::null(); SCALAR_PROJ_CACHE_CAP],
            // MaybeUninit::uninit() is const-eval'd and produces no stores.
            vals: [const { MaybeUninit::uninit() }; SCALAR_PROJ_CACHE_CAP],
            len: 0,
        }
    }

    pub fn get(&self, scalar: &S) -> Option<V> {
        let ptr = scalar as *const S;
        for i in 0..self.len {
            if self.ptrs[i] == ptr {
                // SAFETY: vals[i] is initialized because i < self.len; the
                // only way to grow `len` is via `push`, which writes before
                // incrementing.
                return Some(unsafe { self.vals[i].assume_init_ref() }.clone());
            }
        }
        None
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn push(&mut self, scalar: &S, value: V) {
        if self.len < SCALAR_PROJ_CACHE_CAP {
            self.ptrs[self.len] = scalar as *const S;
            self.vals[self.len].write(value);
            self.len += 1;
        }
        // Silently drop if full: correctness is preserved (HashMap fallback
        // in the caller), perf just degrades toward the pre-cache path.
    }
}

impl<S, V: Clone> Default for ScalarProjCache<S, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, V> Drop for ScalarProjCache<S, V> {
    fn drop(&mut self) {
        // SAFETY: vals[0..self.len] are the slots written by `push`, so
        // they are the initialized prefix. The tail (len..CAP) is still
        // uninitialized and must not be dropped.
        for i in 0..self.len {
            unsafe {
                self.vals[i].assume_init_drop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One cache round-trip against a scalar living in this frame. The
    /// scalar dies when the call returns, so its stack slot is free for
    /// the next call's scalar. `#[inline(never)]` pins the frame layout,
    /// making the address reuse deterministic across sequential calls at
    /// the same call depth.
    #[inline(never)]
    fn round_trip(
        cache: &mut ScalarProjCache<u64, u64>,
        scalar: u64,
        proj: u64,
    ) -> (usize, Option<u64>) {
        let scalar = std::hint::black_box(scalar);
        let addr = &scalar as *const u64 as usize;
        let hit = cache.get(&scalar);
        if hit.is_none() {
            cache.push(&scalar, proj);
        }
        (addr, hit)
    }

    /// Documents a footgun rather than guarding desired behavior: the
    /// cache keys on the scalar's ADDRESS, so a scalar passed through a
    /// short-lived temporary can alias a dead entry for a DIFFERENT
    /// value, and `get` returns the stale projection. A caller that
    /// evaluates each constant into a fresh per-iteration local (e.g. an
    /// interpreted UAIR) sees every constant after the first evaluate as
    /// the first; prover and verifier corrupt identically, so only a
    /// ground-truth check catches it. Fix options: key by scalar value
    /// (compare or hash) instead of pointer, or require `from_ref`/`mbs`
    /// scalars to outlive the whole constrain call so temporaries can
    /// never reuse a slot.
    #[test]
    fn pointer_reuse_returns_stale_projection() {
        let mut cache = ScalarProjCache::<u64, u64>::new();

        let (addr_a, hit_a) = round_trip(&mut cache, 41, 0xA);
        assert_eq!(hit_a, None, "first probe must miss");

        let (addr_b, hit_b) = round_trip(&mut cache, 43, 0xB);

        if addr_a != addr_b {
            // The collision needs address reuse; if this build laid the
            // two frames out differently there is nothing to demonstrate.
            eprintln!(
                "skipping: temporaries at distinct addresses \
                 ({addr_a:#x} vs {addr_b:#x})"
            );
            return;
        }

        // The collision: scalar 43 receives scalar 41's projection.
        assert_eq!(
            hit_b,
            Some(0xA),
            "expected the same-address temporary to hit the stale entry"
        );
    }
}
