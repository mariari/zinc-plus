//! Multi-point reducer for **int** columns (scalar-valued MLEs over `F_q`).
//!
//! The int twin of [`crate::bin_multipoint_reducer`]: reduces `T` batches of
//! scalar MLE evaluation claims at distinct points on a shared batch of
//! witness int columns into a single Zip+ opening at one reduced point,
//! saving `T − 1` int-lane opens. Used to discharge the `r_inner` claims
//! of `Word`-table lookups (one per int lookup group) together with the
//! step-7 `r_0` claim.
//!
//! # Setting
//!
//! Let `(col_0, …, col_{n-1})` be the witness int columns, projected to
//! `F_q` (`v_j(x) = col_j[x] mod q`), over `num_vars` variables.
//!
//! Inputs (T claims): for each `t`, a point `r^(t) ∈ F_q^{num_vars}` and
//! scalars `evals_t[j]` claimed to equal `MLE[v_j](r^(t))`.
//!
//! # Reducer
//!
//! 1. Sample `gammas[j]` (n scalars) and `betas[t]` (T scalars) from the
//!    Fiat–Shamir transcript — AFTER every `evals_t` has been absorbed by
//!    the caller, otherwise a prover could choose the claimed evals to fit
//!    the challenges.
//! 2. `y_t = Σ_j gammas[j] · evals_t[j]`, `Y = Σ_t betas[t] · y_t`.
//! 3. Degree-2 sumcheck on `Y = Σ_x P(x) · M(x)` with
//!    `P(x) = Σ_j gammas[j] · v_j(x)` and `M(x) = Σ_t betas[t] · eq(x, r^(t))`.
//! 4. The sumcheck output `(r*, v* = P(r*) · M(r*))` reduces every claim to
//!    `P(r*) = v* / M(r*)`. The caller has the prover send the per-column
//!    evals at `r*`, checks `Σ_j gammas[j] · eval*_j = P(r*)`, and opens the
//!    int commitment once at `r*` with those evals (alpha-weighted by the
//!    Zip+ per-polynomial alphas).
//!
//! Soundness mirrors the bin reducer: β folds the claims, γ batches the
//! columns, the sumcheck pins one point, and the Zip+ open binds `P(r*)`
//! to the committed columns, so every claimed `evals_t[j]` is bound.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::{ArithErrors, build_eq_x_r_inner},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField};

pub use crate::bin_multipoint_reducer::{Proof, Reduced, ReducerError};
use crate::sumcheck::MLSumcheck;

/// One claim: a point `r^(t)` and the prover-supplied scalar evals at that
/// point (one per int column, in witness-int-col order).
#[derive(Clone, Debug)]
pub struct IntClaim<F: PrimeField> {
    pub point: Vec<F>,
    /// `evals[j]` should equal `MLE[col_j](point)`.
    pub evals: Vec<F>,
}

pub struct IntMultipointReducer<F>(PhantomData<F>);

impl<F> IntMultipointReducer<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Run the reducer prover. `cols[j]` are the projected witness int
    /// columns (Montgomery-inner MLEs); `claims[t].evals[j]` MUST equal
    /// `MLE[cols[j]](claims[t].point)` for an honest prover.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove(
        transcript: &mut impl Transcript,
        cols: &[DenseMultilinearExtension<F::Inner>],
        claims: &[IntClaim<F>],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, Reduced<F>), ReducerError<F>> {
        assert!(!cols.is_empty(), "reducer needs at least one int col");
        assert!(!claims.is_empty(), "reducer needs at least one claim");
        let n_cols = cols.len();
        let zero = F::zero_with_cfg(field_cfg);
        let zero_inner = zero.inner().clone();

        let gammas: Vec<F> = transcript.get_field_challenges(n_cols, field_cfg);
        let betas: Vec<F> = transcript.get_field_challenges(claims.len(), field_cfg);

        let n_hyper = 1usize << num_vars;
        for col in cols {
            assert_eq!(col.evaluations.len(), n_hyper, "int column length must be 2^num_vars");
        }

        // P(x) = Σ_j γ_j · v_j(x) on the hypercube.
        let p_evals: Vec<F::Inner> = cfg_into_iter!(0..n_hyper)
            .map(|x_idx| {
                let mut s = zero.clone();
                for (j, col) in cols.iter().enumerate() {
                    let v = F::new_unchecked_with_cfg(col.evaluations[x_idx].clone(), field_cfg);
                    s = s + &(gammas[j].clone() * &v);
                }
                s.into_inner()
            })
            .collect();
        let p_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, p_evals, zero_inner.clone());

        // M(x) = Σ_t β_t · eq(x, r^(t)) on the hypercube.
        let eq_tables: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(claims)
            .map(|c| build_eq_x_r_inner::<F>(&c.point, field_cfg).expect("eq build"))
            .collect();
        let m_evals: Vec<F::Inner> = cfg_into_iter!(0..n_hyper)
            .map(|x_idx| {
                let mut s = zero.clone();
                for (t, eq_t) in eq_tables.iter().enumerate() {
                    let e_f = F::new_unchecked_with_cfg(eq_t.evaluations[x_idx].clone(), field_cfg);
                    s = s + &(betas[t].clone() * &e_f);
                }
                s.into_inner()
            })
            .collect();
        let m_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, m_evals, zero_inner);

        let mles = vec![p_mle.clone(), m_mle];
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            2,
            |v: &[F]| v[0].clone() * &v[1],
            field_cfg,
        );
        let r_star = sumcheck_state.randomness.clone();
        let p_at_r_star = p_mle
            .evaluate_with_config(&r_star, field_cfg)
            .expect("p_mle eval at r*");

        Ok((
            Proof { sumcheck_proof },
            Reduced {
                point: r_star,
                gammas_flat: gammas,
                p_eval: p_at_r_star,
            },
        ))
    }

    /// Run the reducer verifier on the prover-supplied claims.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify(
        transcript: &mut impl Transcript,
        proof: &Proof<F>,
        claims: &[IntClaim<F>],
        n_cols: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Reduced<F>, ReducerError<F>> {
        assert!(!claims.is_empty(), "reducer needs at least one claim");
        let gammas: Vec<F> = transcript.get_field_challenges(n_cols, field_cfg);
        let betas: Vec<F> = transcript.get_field_challenges(claims.len(), field_cfg);

        let zero = F::zero_with_cfg(field_cfg);
        let y_t: Vec<F> = claims
            .iter()
            .map(|c| {
                assert_eq!(c.evals.len(), n_cols, "claim must carry one eval per int col");
                c.evals
                    .iter()
                    .zip(gammas.iter())
                    .fold(zero.clone(), |s, (e, g)| s + &(g.clone() * e))
            })
            .collect();
        let total: F = betas
            .iter()
            .zip(y_t.iter())
            .fold(zero.clone(), |acc, (b, y)| acc + &(b.clone() * y));

        if proof.sumcheck_proof.claimed_sum != total {
            return Err(ReducerError::ClaimedSumMismatch {
                got: proof.sumcheck_proof.claimed_sum.clone(),
                expected: total,
            });
        }

        let sub =
            MLSumcheck::verify_as_subprotocol(transcript, num_vars, 2, &proof.sumcheck_proof, field_cfg)?;
        let r_star = sub.point.clone();
        let m_at_r_star = m_evaluation_at_point(claims, &betas, &r_star, field_cfg)?;
        if m_at_r_star == zero {
            return Err(ReducerError::ZeroMSelector);
        }
        let one = F::one_with_cfg(field_cfg);
        let m_inv = one / &m_at_r_star;
        let p_at_r_star = sub.expected_evaluation.clone() * &m_inv;

        Ok(Reduced {
            point: r_star,
            gammas_flat: gammas,
            p_eval: p_at_r_star,
        })
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn m_evaluation_at_point<F: PrimeField>(
    claims: &[IntClaim<F>],
    betas: &[F],
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<F, ArithErrors> {
    let one = F::one_with_cfg(field_cfg);
    let mut s = F::zero_with_cfg(field_cfg);
    for (claim, beta) in claims.iter().zip(betas.iter()) {
        let eq = zinc_poly::utils::eq_eval(r_star, &claim.point, one.clone())?;
        s = s + &(beta.clone() * &eq);
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;

    fn rand_col(n_vars: usize, rng: &mut impl RngCore) -> DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner> {
        let len = 1usize << n_vars;
        let evals = (0..len).map(|_| F::from(rng.next_u64()).into_inner()).collect();
        DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, F::from(0u64).into_inner())
    }

    fn eval_at(col: &DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>, point: &[F]) -> F {
        let eq = zinc_poly::utils::build_eq_x_r_vec(point, &()).unwrap();
        col.evaluations
            .iter()
            .zip(eq.iter())
            .fold(F::from(0u64), |acc, (v, e)| acc + F::new_unchecked_with_cfg(v.clone(), &()) * e)
    }

    #[test]
    fn round_trip_t3_n4() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(5);
        let n_vars = 6;
        let cols: Vec<_> = (0..4).map(|_| rand_col(n_vars, &mut rng)).collect();
        let claims: Vec<IntClaim<F>> = (0..3)
            .map(|_| {
                let point: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
                let evals = cols.iter().map(|c| eval_at(c, &point)).collect();
                IntClaim { point, evals }
            })
            .collect();

        let mut p_ts = Blake3Transcript::new();
        let (proof, p_red) =
            IntMultipointReducer::<F>::prove(&mut p_ts, &cols, &claims, n_vars, &cfg).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let v_red = IntMultipointReducer::<F>::verify(&mut v_ts, &proof, &claims, cols.len(), n_vars, &cfg)
            .expect("verify");
        assert_eq!(p_red.point, v_red.point);
        assert_eq!(p_red.gammas_flat, v_red.gammas_flat);
        assert_eq!(p_red.p_eval, v_red.p_eval);
        // P(r*) = Σ_j γ_j · MLE[col_j](r*)
        let direct = cols
            .iter()
            .zip(v_red.gammas_flat.iter())
            .fold(F::from(0u64), |acc, (c, g)| acc + g.clone() * eval_at(c, &v_red.point));
        assert_eq!(direct, v_red.p_eval);
    }

    #[test]
    fn tampered_eval_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(6);
        let n_vars = 5;
        let cols: Vec<_> = (0..2).map(|_| rand_col(n_vars, &mut rng)).collect();
        let point: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
        let mut claim = IntClaim { point: point.clone(), evals: cols.iter().map(|c| eval_at(c, &point)).collect() };
        claim.evals[1] = claim.evals[1].clone() + F::from(1u64);
        let claims = vec![claim];
        let mut p_ts = Blake3Transcript::new();
        let (proof, _) =
            IntMultipointReducer::<F>::prove(&mut p_ts, &cols, &claims, n_vars, &cfg).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let res = IntMultipointReducer::<F>::verify(&mut v_ts, &proof, &claims, cols.len(), n_vars, &cfg);
        assert!(matches!(res, Err(ReducerError::ClaimedSumMismatch { .. })));
    }
}
