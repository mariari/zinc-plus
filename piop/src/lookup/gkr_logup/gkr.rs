//! GKR-based LogUp fractional sumcheck.
//!
//! Implements the GKR (Goldwasser-Kalai-Rothblum) protocol for proving
//! fractional sum identities, as described in:
//!
//!   Papini & Haböck, "Improving logarithmic derivative lookups using GKR"
//!   <https://eprint.iacr.org/2023/1284>
//!
//! ## Key advantage over the standard LogUp
//!
//! The prover only needs to send the **multiplicity vector** `m` — no
//! inverse vectors `u`, `v` are required.  This saves transmitting
//! `O(W + N)` field elements at the cost of `O(log²(max(W,N)))` extra
//! field elements in the GKR layer proofs.
//!
//! ## Protocol overview
//!
//! Given leaf fractions `p_i / q_i` for `i = 0..2^d`, the GKR fractional
//! sumcheck proves `Σ_i p_i/q_i = root_p/root_q` using a binary tree
//! of fraction additions verified layer-by-layer from root to leaves
//! using sumchecks.  At the leaves, the verifier checks evaluations against
//! the known input polynomials.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::{
    MLSumcheck, SumcheckProof,
    prover::{NatEvaluatedPolyWithoutConstant, ProverMsg},
};

use super::structs::{
    BatchedGkrFractionProof, BatchedGkrLayerProof,
    GkrFractionProof, GkrLayerProof, GkrLogupError as LookupError,
};

// ---------------------------------------------------------------------------
// Fraction tree helpers
// ---------------------------------------------------------------------------

/// A single layer of the fraction tree, storing numerators and denominators.
#[derive(Clone, Debug)]
pub(super) struct FractionLayer<F> {
    pub(super) p: Vec<F>, // numerators
    pub(super) q: Vec<F>, // denominators
}

/// Build the full fraction tree bottom-up.
///
/// The tree has `d + 1` layers (layer `d` = leaves, layer 0 = root).
/// Returns layers `[layer_d, layer_{d-1}, ..., layer_0]`, i.e. indexed
/// from leaves to root.
///
/// Layer indexing: GKR layer `k` has `2^k` entries.
/// - Layer 0 (root): 1 entry
/// - Layer `d` (leaves): `2^d` entries
///
/// Transition from layer `k+1` to layer `k`:
/// ```text
/// p_k[i] = p_{k+1}[i] · q_{k+1}[i + 2^k] + p_{k+1}[i + 2^k] · q_{k+1}[i]
/// q_k[i] = q_{k+1}[i] · q_{k+1}[i + 2^k]
/// ```
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn build_fraction_tree<F: InnerTransparentField + Send + Sync>(
    leaf_p: Vec<F>,
    leaf_q: Vec<F>,
) -> Vec<FractionLayer<F>>
where
    F::Config: Sync,
{
    let d = zinc_utils::log2(leaf_p.len()) as usize;
    debug_assert_eq!(leaf_p.len(), 1 << d);
    debug_assert_eq!(leaf_q.len(), 1 << d);

    // layers[0] = leaves (layer d), layers[d] = root (layer 0)
    let mut layers = Vec::with_capacity(d + 1);
    layers.push(FractionLayer {
        p: leaf_p,
        q: leaf_q,
    });

    for level in (0..d).rev() {
        // Going from GKR layer (level+1) to layer (level).
        let child = layers.last().expect("tree is non-empty during construction");
        let half = 1usize << level;

        let (parent_p, parent_q): (Vec<F>, Vec<F>) = if half >= 64 {
            cfg_into_iter!(0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
                    (pl.clone() * qr + &(pr.clone() * ql), ql.clone() * qr)
                })
                .unzip()
        } else {
            (0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
                    (pl.clone() * qr + &(pr.clone() * ql), ql.clone() * qr)
                })
                .unzip()
        };

        layers.push(FractionLayer {
            p: parent_p,
            q: parent_q,
        });
    }

    // layers is now [leaves, ..., root]
    layers
}

/// Like [`build_fraction_tree`], but optimised for the common LogUp case
/// where **every** leaf numerator is `one` (multiplicity-1 witness lookups).
///
/// At the first (leaf) level of the tree the standard formula
///   `p_parent = p_l·q_r + p_r·q_l`
/// simplifies to `p_parent = q_l + q_r` because `p_l = p_r = 1`.
/// This replaces 2 field multiplications with 1 addition per node at
/// the widest layer, saving ~⅔ of the first-level work.
///
/// **Pre-condition**: `leaf_q.len()` must be a power of 2 and every
/// leaf `p` value is `one`.  Callers must ensure there is no padding
/// (i.e. `data_len == 2^d`) or that padding entries also satisfy `p=1`.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn build_fraction_tree_ones_leaf<F: InnerTransparentField + Send + Sync>(
    one: F,
    leaf_q: Vec<F>,
) -> Vec<FractionLayer<F>>
where
    F::Config: Sync,
{
    let d = zinc_utils::log2(leaf_q.len()) as usize;
    debug_assert_eq!(leaf_q.len(), 1 << d);

    let leaf_p = vec![one; 1 << d];

    let mut layers = Vec::with_capacity(d + 1);
    layers.push(FractionLayer {
        p: leaf_p,
        q: leaf_q,
    });

    if d == 0 {
        return layers;
    }

    // First level: exploit p_leaf == 1 everywhere.
    // p_parent = 1·q_r + 1·q_l = q_l + q_r   (saves 2 muls per node)
    // q_parent = q_l · q_r                     (unchanged)
    {
        let child = layers.last().expect("tree is non-empty");
        let half = 1usize << (d - 1);

        let (parent_p, parent_q): (Vec<F>, Vec<F>) = if half >= 64 {
            cfg_into_iter!(0..half)
                .map(|i| {
                    let ql = &child.q[i];
                    let qr = &child.q[i + half];
                    (ql.clone() + qr, ql.clone() * qr)
                })
                .unzip()
        } else {
            (0..half)
                .map(|i| {
                    let ql = &child.q[i];
                    let qr = &child.q[i + half];
                    (ql.clone() + qr, ql.clone() * qr)
                })
                .unzip()
        };

        layers.push(FractionLayer {
            p: parent_p,
            q: parent_q,
        });
    }

    // Remaining levels: standard formula.
    for level in (0..d.saturating_sub(1)).rev() {
        let child = layers.last().expect("tree is non-empty during construction");
        let half = 1usize << level;

        let (parent_p, parent_q): (Vec<F>, Vec<F>) = if half >= 64 {
            cfg_into_iter!(0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
                    (pl.clone() * qr + &(pr.clone() * ql), ql.clone() * qr)
                })
                .unzip()
        } else {
            (0..half)
                .map(|i| {
                    let pl = &child.p[i];
                    let ql = &child.q[i];
                    let pr = &child.p[i + half];
                    let qr = &child.q[i + half];
                    (pl.clone() * qr + &(pr.clone() * ql), ql.clone() * qr)
                })
                .unzip()
        };

        layers.push(FractionLayer {
            p: parent_p,
            q: parent_q,
        });
    }

    layers
}

/// Evaluate a k-variable MLE (given as evaluations over {0,1}^k in
/// little-endian order) at a point in F^k.
#[allow(clippy::arithmetic_side_effects, dead_code)]
pub(super) fn evaluate_mle_at<F: InnerTransparentField>(
    evals: &[F],
    point: &[F],
    field_cfg: &F::Config,
) -> F {
    let n = point.len();
    debug_assert_eq!(evals.len(), 1 << n);
    if n == 0 {
        return evals[0].clone();
    }

    let mut current = evals.to_vec();
    for r_i in point {
        let half = current.len() / 2;
        let one_minus_r = F::one_with_cfg(field_cfg) - r_i;
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            next.push(
                one_minus_r.clone() * &current[2 * j]
                    + &(r_i.clone() * &current[2 * j + 1]),
            );
        }
        current = next;
    }
    current[0].clone()
}

// ---------------------------------------------------------------------------
// GKR fractional sumcheck: prover + verifier
// ---------------------------------------------------------------------------

/// Run the GKR prover for a single fractional sumcheck.
///
// ---------------------------------------------------------------------------
// Layer sumcheck prover (eq-factored)
// ---------------------------------------------------------------------------

/// Prover of one GKR layer's (batched) sumcheck, specialised to its shape
///
/// ```text
///   Σ_{x ∈ {0,1}^k} eq(x, r) · Σ_ℓ δ^ℓ · (pla_ℓ(x) · qr_ℓ(x) + pr_ℓ(x) · ql_ℓ(x)),
///   pla_ℓ = pl_ℓ + α · ql_ℓ,
/// ```
///
/// producing exactly the messages and transcript interaction of
/// [`crate::sumcheck::MLSumcheck::prove_as_subprotocol`] run on the MLEs
/// `[eq, pla_0, ql_0, pr_0, qr_0, …]` with degree 3 (the verifier is
/// unchanged: it still runs the generic sumcheck verifier).
///
/// The eq factor is not carried as a folded multilinear. With
/// `x = (s_{<i}, X, x')` it splits as
/// `E_{i-1} · eq(X, r_i) · eq(x', r_{>i})` with `E_{i-1} = Π_{j<i} eq(s_j, r_j)`,
/// so round `i` needs only the degree-2 inner sum
/// `H_i(X) = Σ_{x'} eq(x', r_{>i}) · h(s_{<i}, X, x')` at `X ∈ {0, 1, 2}`
/// (its value at `3` follows by extrapolation), and the round polynomial
/// is `E_{i-1} · eq(X, r_i) · H_i(X)`. Per hypercube pair and tree that is
/// three evaluations of `h` instead of four of `eq · h`, one array fewer to
/// fold each round, and no multiplication by `δ^ℓ` per pair (the per-tree
/// sums are weighted once).
///
/// `per_tree[ℓ] = (pl, pr, ql, qr)`, each `2^k` long. Returns the proof,
/// the challenges `s` and, per tree, the fully folded
/// `[pla(s), ql(s), pr(s), qr(s)]`.
#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
fn layer_sumcheck_prove<F>(
    transcript: &mut impl Transcript,
    k: usize,
    r: &[F],
    per_tree: &[(&[F], &[F], &[F], &[F])],
    alpha: &F,
    delta_powers: &[F],
    field_cfg: &F::Config,
) -> (SumcheckProof<F>, Vec<F>, Vec<[F; 4]>)
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    debug_assert!(k >= 1, "layer sumcheck needs at least one variable");
    debug_assert_eq!(r.len(), k);
    let num_trees = per_tree.len();
    debug_assert_eq!(delta_powers.len(), num_trees);
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];

    // Same preamble as the generic driver.
    transcript.absorb_random_field(&F::from_with_cfg(k as u64, field_cfg), &mut buf);
    transcript.absorb_random_field(&F::from_with_cfg(3u64, field_cfg), &mut buf);

    // Round 1 reads the layer slices (`pla` materialised once); the fold
    // after each round writes the half-size arrays [pla, ql, pr, qr].
    let mut work: Vec<[Vec<F>; 4]> = cfg_into_iter!(per_tree)
        .map(|(pl, pr, ql, qr)| {
            let pla: Vec<F> = pl
                .iter()
                .zip(ql.iter())
                .map(|(pl, ql)| pl.clone() + &(ql.clone() * alpha))
                .collect();
            [pla, ql.to_vec(), pr.to_vec(), qr.to_vec()]
        })
        .collect();

    let mut messages = Vec::with_capacity(k);
    let mut s: Vec<F> = Vec::with_capacity(k);
    // E_{i-1} = Π_{j<i} eq(s_j, r_j).
    let mut e_prefix = one.clone();
    let mut claimed_sum = zero.clone();

    for i in 0..k {
        let r_i = &r[i];
        // eq(x', r_{>i}) over the remaining variables (`[1]` in the last round).
        let eq_tail: Vec<F> = if i + 1 < k {
            build_eq_x_r_vec(&r[i + 1..], field_cfg).expect("eq table build")
        } else {
            vec![one.clone()]
        };

        // H(0), H(1), H(2) = Σ_ℓ δ^ℓ · Σ_b eq_tail[b] · h_{ℓ,b}(X).
        let mut h = [zero.clone(), zero.clone(), zero.clone()];
        for (arrays, dp) in work.iter().zip(delta_powers.iter()) {
            let [pla, ql, pr, qr] = arrays;
            let sums = layer_round_sums(pla, ql, pr, qr, &eq_tail, &zero);
            if dp == &one {
                for (acc, v) in h.iter_mut().zip(sums) {
                    *acc += &v;
                }
            } else {
                for (acc, v) in h.iter_mut().zip(sums) {
                    *acc += &(v * dp);
                }
            }
        }
        let [h0, h1, h2] = h;
        // Degree 2 in X: H(3) = 3·H(2) − 3·H(1) + H(0).
        let three_h2 = h2.clone() + &h2 + &h2;
        let three_h1 = h1.clone() + &h1 + &h1;
        let h3 = three_h2 - &three_h1 + &h0;

        // eq(X, r_i) = (1 − r_i) + X · (2·r_i − 1).
        let e0 = one.clone() - r_i;
        let slope = r_i.clone() + r_i - &one;
        let e1 = e0.clone() + &slope;
        let e2 = e1.clone() + &slope;
        let e3 = e2.clone() + &slope;

        let p0 = e_prefix.clone() * &e0 * &h0;
        let p1 = e_prefix.clone() * &e1 * &h1;
        let p2 = e_prefix.clone() * &e2 * &h2;
        let p3 = e_prefix.clone() * &e3 * &h3;
        if i == 0 {
            claimed_sum = p0 + &p1;
        }
        let tail = vec![p1, p2, p3];
        transcript.absorb_random_field_slice(&tail, &mut buf);
        messages.push(ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail)));
        let s_i: F = transcript.get_field_challenge(field_cfg);
        transcript.absorb_random_field(&s_i, &mut buf);

        // Fold the working arrays at s_i and extend the eq prefix.
        work = cfg_into_iter!(work)
            .map(|arrays| arrays.map(|a| fold_at(&a, &s_i)))
            .collect();
        // eq(s_i, r_i) = 1 − s_i − r_i + 2·s_i·r_i.
        let sr = s_i.clone() * r_i;
        e_prefix *= &(one.clone() - &s_i - r_i + &sr + &sr);
        s.push(s_i);
    }

    let finals: Vec<[F; 4]> = work
        .into_iter()
        .map(|arrays| arrays.map(|a| a.into_iter().next().expect("folded to one entry")))
        .collect();
    (
        SumcheckProof {
            messages,
            claimed_sum,
        },
        s,
        finals,
    )
}

/// One tree's contribution to a layer round:
/// `Σ_b eq_tail[b] · h_b(X)` at `X ∈ {0, 1, 2}`, where
/// `h_b(X) = pla_b(X) · qr_b(X) + pr_b(X) · ql_b(X)` on the pair
/// `(2b, 2b + 1)` (each array linearly interpolated in `X`).
#[allow(clippy::arithmetic_side_effects)]
fn layer_round_sums<F>(
    pla: &[F],
    ql: &[F],
    pr: &[F],
    qr: &[F],
    eq_tail: &[F],
    zero: &F,
) -> [F; 3]
where
    F: InnerTransparentField + Send + Sync,
{
    let half = eq_tail.len();
    debug_assert_eq!(pla.len(), 2 * half);
    let pair = |b: usize| -> [F; 3] {
        let (a0, a1) = (&pla[2 * b], &pla[2 * b + 1]);
        let (c0, c1) = (&ql[2 * b], &ql[2 * b + 1]);
        let (p0, p1) = (&pr[2 * b], &pr[2 * b + 1]);
        let (q0, q1) = (&qr[2 * b], &qr[2 * b + 1]);
        let t0 = a0.clone() * q0 + &(p0.clone() * c0);
        let t1 = a1.clone() * q1 + &(p1.clone() * c1);
        // Values at X = 2: 2·v1 − v0.
        let a2 = a1.clone() + a1 - a0;
        let c2 = c1.clone() + c1 - c0;
        let p2 = p1.clone() + p1 - p0;
        let q2 = q1.clone() + q1 - q0;
        let t2 = a2 * &q2 + &(p2 * &c2);
        let w = &eq_tail[b];
        [t0 * w, t1 * w, t2 * w]
    };
    let add3 = |mut acc: [F; 3], v: [F; 3]| -> [F; 3] {
        for (a, x) in acc.iter_mut().zip(v) {
            *a += &x;
        }
        acc
    };
    let init = || [zero.clone(), zero.clone(), zero.clone()];
    #[cfg(feature = "parallel")]
    {
        (0..half)
            .into_par_iter()
            .fold(init, |acc, b| add3(acc, pair(b)))
            .reduce(init, add3)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..half).fold(init(), |acc, b| add3(acc, pair(b)))
    }
}

/// `out[b] = a[2b] + s · (a[2b + 1] − a[2b])`: the multilinear fold of the
/// first variable at `s` (the pair layout of the sumcheck driver).
#[allow(clippy::arithmetic_side_effects)]
fn fold_at<F>(a: &[F], s: &F) -> Vec<F>
where
    F: InnerTransparentField + Send + Sync,
{
    let half = a.len() / 2;
    cfg_into_iter!(0..half)
        .map(|b| {
            let lo = &a[2 * b];
            let hi = &a[2 * b + 1];
            lo.clone() + &((hi.clone() - lo) * s)
        })
        .collect()
}

/// Proves `Σ_{x ∈ {0,1}^d} p(x)/q(x) = root_p/root_q` and returns
/// a `(GkrFractionProof, eval_point)` pair.  The evaluation point is
/// tracked during the prove so callers don't need to re-derive it.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn gkr_fraction_prove<F>(
    transcript: &mut impl Transcript,
    layers: &[FractionLayer<F>], // [leaves, ..., root]
    field_cfg: &F::Config,
) -> (GkrFractionProof<F>, Vec<F>)
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let d = layers.len() - 1; // number of GKR levels
    let root = layers.last().expect("tree is non-empty");
    let root_p = root.p[0].clone();
    let root_q = root.q[0].clone();

    // Absorb root values into transcript.
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&root_p, &mut buf);
    transcript.absorb_random_field(&root_q, &mut buf);

    if d == 0 {
        return (GkrFractionProof {
            root_p,
            root_q,
            layer_proofs: vec![],
        }, vec![]);
    }

    let mut layer_proofs = Vec::with_capacity(d);

    let mut v_p = root_p.clone();
    let mut v_q = root_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        // Child layer in our array: layers[d - (round + 1)]
        let child_layer_idx = d - (round + 1);
        let child_layer = &layers[child_layer_idx];

        let k = round;
        let half = 1usize << k;

        let p_left_vals = &child_layer.p[..half];
        let p_right_vals = &child_layer.p[half..];
        let q_left_vals = &child_layer.q[..half];
        let q_right_vals = &child_layer.q[half..];

        // Get batching challenge α_layer.
        let alpha: F = transcript.get_field_challenge(field_cfg);

        if k == 0 {
            // Round 0: zero sumcheck variables — direct algebraic check.
            let pl = p_left_vals[0].clone();
            let pr = p_right_vals[0].clone();
            let ql = q_left_vals[0].clone();
            let qr = q_right_vals[0].clone();

            transcript.absorb_random_field(&pl, &mut buf);
            transcript.absorb_random_field(&pr, &mut buf);
            transcript.absorb_random_field(&ql, &mut buf);
            transcript.absorb_random_field(&qr, &mut buf);

            debug_assert!({
                let lhs = v_p.clone() + &(alpha.clone() * &v_q);
                let cross = pl.clone() * &qr + &(pr.clone() * &ql);
                let prod = ql.clone() * &qr;
                let rhs = cross + &(alpha.clone() * &prod);
                lhs == rhs
            });

            layer_proofs.push(GkrLayerProof {
                sumcheck_proof: None,
                p_left: pl.clone(),
                p_right: pr.clone(),
                q_left: ql.clone(),
                q_right: qr.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            let one_minus_lambda = one - &lambda;
            v_p = one_minus_lambda.clone() * &pl + &(lambda.clone() * &pr);
            v_q = one_minus_lambda * &ql + &(lambda.clone() * &qr);
            r_k = vec![lambda];
        } else {
            // Round k ≥ 1: sumcheck over k variables (see
            // `layer_sumcheck_prove`; single tree, so δ = [1]).
            let per_tree = [(p_left_vals, p_right_vals, q_left_vals, q_right_vals)];
            let one = F::one_with_cfg(field_cfg);
            let (sumcheck_proof, s, finals) = layer_sumcheck_prove(
                transcript,
                k,
                &r_k,
                &per_tree,
                &alpha,
                std::slice::from_ref(&one),
                field_cfg,
            );
            let [pla_at_s, ql_at_s, pr_at_s, qr_at_s] = finals
                .into_iter()
                .next()
                .expect("one tree");
            // pla(s) = pl(s) + α·ql(s); recover pl(s).
            let pl_at_s = pla_at_s - &(ql_at_s.clone() * &alpha);

            transcript.absorb_random_field(&pl_at_s, &mut buf);
            transcript.absorb_random_field(&pr_at_s, &mut buf);
            transcript.absorb_random_field(&ql_at_s, &mut buf);
            transcript.absorb_random_field(&qr_at_s, &mut buf);

            layer_proofs.push(GkrLayerProof {
                sumcheck_proof: Some(sumcheck_proof),
                p_left: pl_at_s.clone(),
                p_right: pr_at_s.clone(),
                q_left: ql_at_s.clone(),
                q_right: qr_at_s.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one = F::one_with_cfg(field_cfg);
            let one_minus_lambda = one - &lambda;
            v_p = one_minus_lambda.clone() * &pl_at_s + &(lambda.clone() * &pr_at_s);
            v_q = one_minus_lambda * &ql_at_s + &(lambda.clone() * &qr_at_s);
            r_k = s;
            r_k.push(lambda);
        }
    }

    (GkrFractionProof {
        root_p,
        root_q,
        layer_proofs,
    }, r_k)
}

/// Result of verifying a GKR fractional sumcheck.
pub(super) struct GkrFractionVerifyResult<F> {
    /// The evaluation point at the leaf layer: `r ∈ F^d`.
    pub(super) point: Vec<F>,
    /// Expected evaluation of the numerator MLE: `p̃(r)`.
    pub(super) expected_p: F,
    /// Expected evaluation of the denominator MLE: `q̃(r)`.
    pub(super) expected_q: F,
}

/// Run the GKR verifier for a single fractional sumcheck.
///
/// Returns the leaf-layer evaluation point and expected (p, q) values.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn gkr_fraction_verify<F>(
    transcript: &mut impl Transcript,
    proof: &GkrFractionProof<F>,
    num_vars: usize, // d = log2(num_leaves)
    field_cfg: &F::Config,
) -> Result<GkrFractionVerifyResult<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let d = num_vars;
    let one = F::one_with_cfg(field_cfg);

    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&proof.root_p, &mut buf);
    transcript.absorb_random_field(&proof.root_q, &mut buf);

    if d == 0 {
        return Ok(GkrFractionVerifyResult {
            point: vec![],
            expected_p: proof.root_p.clone(),
            expected_q: proof.root_q.clone(),
        });
    }

    if proof.layer_proofs.len() != d {
        return Err(LookupError::GkrLeafMismatch);
    }

    let mut v_p = proof.root_p.clone();
    let mut v_q = proof.root_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        let k = round;
        let layer_proof = &proof.layer_proofs[round];

        let alpha: F = transcript.get_field_challenge(field_cfg);

        if k == 0 {
            let pl = &layer_proof.p_left;
            let pr = &layer_proof.p_right;
            let ql = &layer_proof.q_left;
            let qr = &layer_proof.q_right;

            transcript.absorb_random_field(pl, &mut buf);
            transcript.absorb_random_field(pr, &mut buf);
            transcript.absorb_random_field(ql, &mut buf);
            transcript.absorb_random_field(qr, &mut buf);

            let lhs = v_p.clone() + &(alpha.clone() * &v_q);
            let cross = pl.clone() * qr + &(pr.clone() * ql);
            let prod = ql.clone() * qr;
            let rhs = cross + &(alpha * &prod);

            if lhs != rhs {
                return Err(LookupError::GkrLayer0Mismatch {
                    expected: lhs,
                    got: rhs,
                });
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            v_p = one_minus_lambda.clone() * pl + &(lambda.clone() * pr);
            v_q = one_minus_lambda * ql + &(lambda.clone() * qr);
            r_k = vec![lambda];
        } else {
            let sumcheck_proof = layer_proof
                .sumcheck_proof
                .as_ref()
                .ok_or(LookupError::GkrLeafMismatch)?;

            let claimed_sum = v_p.clone() + &(alpha.clone() * &v_q);
            if sumcheck_proof.claimed_sum != claimed_sum {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: claimed_sum,
                    got: sumcheck_proof.claimed_sum.clone(),
                });
            }

            let subclaim = MLSumcheck::verify_as_subprotocol(
                transcript,
                k,
                3,
                sumcheck_proof,
                field_cfg,
            )?;

            let s = &subclaim.point;
            let pl = &layer_proof.p_left;
            let pr = &layer_proof.p_right;
            let ql = &layer_proof.q_left;
            let qr = &layer_proof.q_right;

            transcript.absorb_random_field(pl, &mut buf);
            transcript.absorb_random_field(pr, &mut buf);
            transcript.absorb_random_field(ql, &mut buf);
            transcript.absorb_random_field(qr, &mut buf);

            let eq_val = zinc_poly::utils::eq_eval(s, &r_k, one.clone())?;

            let cross = pl.clone() * qr + &(pr.clone() * ql);
            let prod = ql.clone() * qr;
            let inner = cross + &(alpha * &prod);
            let expected = eq_val * &inner;

            if expected != subclaim.expected_evaluation {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: subclaim.expected_evaluation,
                    got: expected,
                });
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            v_p = one_minus_lambda.clone() * pl + &(lambda.clone() * pr);
            v_q = one_minus_lambda * ql + &(lambda.clone() * qr);
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    Ok(GkrFractionVerifyResult {
        point: r_k,
        expected_p: v_p,
        expected_q: v_q,
    })
}

// ---------------------------------------------------------------------------
// Batched GKR fractional sumcheck (L trees, layer-wise batching)
// ---------------------------------------------------------------------------

/// Result of the batched GKR fractional sumcheck prover.
pub(super) struct BatchedGkrFractionProveResult<F: PrimeField> {
    /// The proof containing per-tree roots and batched layer proofs.
    pub(super) proof: BatchedGkrFractionProof<F>,
    /// The shared evaluation point at the leaf layer: `r ∈ F^d`.
    pub(super) eval_point: Vec<F>,
}

/// Run the batched GKR prover for L fraction trees simultaneously.
///
/// All L trees must have the same depth `d`. The prover processes one
/// GKR layer at a time, batching the L per-tree sumchecks into a single
/// sumcheck with `1 + 4L` MLEs at degree 3.
///
/// Returns `(BatchedGkrFractionProof, eval_point)`.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn batched_gkr_fraction_prove<F>(
    transcript: &mut impl Transcript,
    all_layers: &[Vec<FractionLayer<F>>], // L trees, each [leaves, ..., root]
    field_cfg: &F::Config,
) -> BatchedGkrFractionProveResult<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let num_trees = all_layers.len(); // L
    let d = all_layers[0].len() - 1; // number of GKR levels

    // Collect per-tree roots.
    let roots_p: Vec<F> = all_layers
        .iter()
        .map(|layers| layers.last().expect("tree non-empty").p[0].clone())
        .collect();
    let roots_q: Vec<F> = all_layers
        .iter()
        .map(|layers| layers.last().expect("tree non-empty").q[0].clone())
        .collect();

    // Absorb all roots into transcript.
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for ell in 0..num_trees {
        transcript.absorb_random_field(&roots_p[ell], &mut buf);
        transcript.absorb_random_field(&roots_q[ell], &mut buf);
    }

    if d == 0 {
        return BatchedGkrFractionProveResult {
            proof: BatchedGkrFractionProof {
                roots_p,
                roots_q,
                layer_proofs: vec![],
            },
            eval_point: vec![],
        };
    }

    let mut layer_proofs = Vec::with_capacity(d);

    // Per-tree running values.
    let mut v_ps: Vec<F> = roots_p.clone();
    let mut v_qs: Vec<F> = roots_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        let k = round;
        let half = 1usize << k;

        // Collect per-tree child arrays for this layer.
        let child_layer_idx = d - (round + 1);
        let per_tree: Vec<(&[F], &[F], &[F], &[F])> = all_layers
            .iter()
            .map(|layers| {
                let cl = &layers[child_layer_idx];
                (
                    &cl.p[..half],
                    &cl.p[half..],
                    &cl.q[..half],
                    &cl.q[half..],
                )
            })
            .collect();

        // Get batching challenges: α_layer (cross-product) and δ (tree batching).
        let alpha: F = transcript.get_field_challenge(field_cfg);
        let delta: F = transcript.get_field_challenge(field_cfg);

        // Precompute δ powers: δ^0, δ^1, ..., δ^{L-1}
        let one = F::one_with_cfg(field_cfg);
        let mut delta_powers = Vec::with_capacity(num_trees);
        let mut dp = one.clone();
        for _ in 0..num_trees {
            delta_powers.push(dp.clone());
            dp *= &delta;
        }

        if k == 0 {
            // Round 0: zero sumcheck variables — direct algebraic check per tree.
            let mut p_lefts = Vec::with_capacity(num_trees);
            let mut p_rights = Vec::with_capacity(num_trees);
            let mut q_lefts = Vec::with_capacity(num_trees);
            let mut q_rights = Vec::with_capacity(num_trees);

            for ell in 0..num_trees {
                let (pl_vals, pr_vals, ql_vals, qr_vals) = per_tree[ell];
                let pl = pl_vals[0].clone();
                let pr = pr_vals[0].clone();
                let ql = ql_vals[0].clone();
                let qr = qr_vals[0].clone();

                transcript.absorb_random_field(&pl, &mut buf);
                transcript.absorb_random_field(&pr, &mut buf);
                transcript.absorb_random_field(&ql, &mut buf);
                transcript.absorb_random_field(&qr, &mut buf);

                debug_assert!({
                    let lhs = v_ps[ell].clone() + &(alpha.clone() * &v_qs[ell]);
                    let cross = pl.clone() * &qr + &(pr.clone() * &ql);
                    let prod = ql.clone() * &qr;
                    let rhs = cross + &(alpha.clone() * &prod);
                    lhs == rhs
                });

                p_lefts.push(pl);
                p_rights.push(pr);
                q_lefts.push(ql);
                q_rights.push(qr);
            }

            layer_proofs.push(BatchedGkrLayerProof {
                sumcheck_proof: None,
                p_lefts: p_lefts.clone(),
                p_rights: p_rights.clone(),
                q_lefts: q_lefts.clone(),
                q_rights: q_rights.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            // Per-tree post-absorb fold — independent across ell.
            v_ps = cfg_into_iter!(0..num_trees)
                .map(|ell| {
                    one_minus_lambda.clone() * &p_lefts[ell]
                        + &(lambda.clone() * &p_rights[ell])
                })
                .collect();
            v_qs = cfg_into_iter!(0..num_trees)
                .map(|ell| {
                    one_minus_lambda.clone() * &q_lefts[ell]
                        + &(lambda.clone() * &q_rights[ell])
                })
                .collect();
            r_k = vec![lambda];
        } else {
            // Round k ≥ 1: batched sumcheck over k variables.
            // The combined claim = Σ_ℓ δ^ℓ · (v_p[ℓ] + α · v_q[ℓ]); the
            // sumcheck polynomial is
            //   eq(x, r) · Σ_ℓ δ^ℓ · ((pl_ℓ + α·ql_ℓ) · qr_ℓ + pr_ℓ · ql_ℓ)
            // (see `layer_sumcheck_prove` for the eq-factored prover).
            let (sumcheck_proof, s, finals) = layer_sumcheck_prove(
                transcript,
                k,
                &r_k,
                &per_tree,
                &alpha,
                &delta_powers,
                field_cfg,
            );

            let mut p_lefts = Vec::with_capacity(num_trees);
            let mut p_rights = Vec::with_capacity(num_trees);
            let mut q_lefts = Vec::with_capacity(num_trees);
            let mut q_rights = Vec::with_capacity(num_trees);

            for [pla, ql, pr, qr] in finals {
                // pla(s) = pl(s) + α·ql(s); recover pl(s) = pla(s) − α·ql(s)
                // since the (pl, pr, ql, qr) absorbed are the per-tree
                // layer-transition evals (pla is an internal sumcheck
                // artifact, never sent in the proof and never absorbed).
                let pl = pla - &(ql.clone() * &alpha);

                transcript.absorb_random_field(&pl, &mut buf);
                transcript.absorb_random_field(&pr, &mut buf);
                transcript.absorb_random_field(&ql, &mut buf);
                transcript.absorb_random_field(&qr, &mut buf);

                p_lefts.push(pl);
                p_rights.push(pr);
                q_lefts.push(ql);
                q_rights.push(qr);
            }

            layer_proofs.push(BatchedGkrLayerProof {
                sumcheck_proof: Some(sumcheck_proof),
                p_lefts: p_lefts.clone(),
                p_rights: p_rights.clone(),
                q_lefts: q_lefts.clone(),
                q_rights: q_rights.clone(),
            });

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            v_ps = cfg_into_iter!(0..num_trees)
                .map(|ell| {
                    one_minus_lambda.clone() * &p_lefts[ell]
                        + &(lambda.clone() * &p_rights[ell])
                })
                .collect();
            v_qs = cfg_into_iter!(0..num_trees)
                .map(|ell| {
                    one_minus_lambda.clone() * &q_lefts[ell]
                        + &(lambda.clone() * &q_rights[ell])
                })
                .collect();
            r_k = s;
            r_k.push(lambda);
        }
    }

    BatchedGkrFractionProveResult {
        proof: BatchedGkrFractionProof {
            roots_p,
            roots_q,
            layer_proofs,
        },
        eval_point: r_k,
    }
}

/// Result of verifying a batched GKR fractional sumcheck.
pub(super) struct BatchedGkrFractionVerifyResult<F> {
    /// The shared evaluation point at the leaf layer.
    pub(super) point: Vec<F>,
    /// Per-tree expected numerator MLE evaluations: `expected_ps[ℓ]`.
    pub(super) expected_ps: Vec<F>,
    /// Per-tree expected denominator MLE evaluations: `expected_qs[ℓ]`.
    pub(super) expected_qs: Vec<F>,
}

/// Run the batched GKR verifier for L fraction trees simultaneously.
///
/// Returns the shared leaf-layer point and per-tree expected (p, q) values.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn batched_gkr_fraction_verify<F>(
    transcript: &mut impl Transcript,
    proof: &BatchedGkrFractionProof<F>,
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<BatchedGkrFractionVerifyResult<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let d = num_vars;
    let num_trees = proof.roots_p.len();
    let one = F::one_with_cfg(field_cfg);

    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for ell in 0..num_trees {
        transcript.absorb_random_field(&proof.roots_p[ell], &mut buf);
        transcript.absorb_random_field(&proof.roots_q[ell], &mut buf);
    }

    if d == 0 {
        return Ok(BatchedGkrFractionVerifyResult {
            point: vec![],
            expected_ps: proof.roots_p.clone(),
            expected_qs: proof.roots_q.clone(),
        });
    }

    if proof.layer_proofs.len() != d {
        return Err(LookupError::GkrLeafMismatch);
    }

    let mut v_ps: Vec<F> = proof.roots_p.clone();
    let mut v_qs: Vec<F> = proof.roots_q.clone();
    let mut r_k: Vec<F> = Vec::new();

    for round in 0..d {
        let k = round;
        let layer_proof = &proof.layer_proofs[round];

        let alpha: F = transcript.get_field_challenge(field_cfg);
        let delta: F = transcript.get_field_challenge(field_cfg);

        let mut delta_powers = Vec::with_capacity(num_trees);
        let mut dp = one.clone();
        for _ in 0..num_trees {
            delta_powers.push(dp.clone());
            dp *= &delta;
        }

        if k == 0 {
            for ell in 0..num_trees {
                let pl = &layer_proof.p_lefts[ell];
                let pr = &layer_proof.p_rights[ell];
                let ql = &layer_proof.q_lefts[ell];
                let qr = &layer_proof.q_rights[ell];

                transcript.absorb_random_field(pl, &mut buf);
                transcript.absorb_random_field(pr, &mut buf);
                transcript.absorb_random_field(ql, &mut buf);
                transcript.absorb_random_field(qr, &mut buf);

                let lhs = v_ps[ell].clone() + &(alpha.clone() * &v_qs[ell]);
                let cross = pl.clone() * qr + &(pr.clone() * ql);
                let prod = ql.clone() * qr;
                let rhs = cross + &(alpha.clone() * &prod);

                if lhs != rhs {
                    return Err(LookupError::GkrLayer0Mismatch {
                        expected: lhs,
                        got: rhs,
                    });
                }
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            for ell in 0..num_trees {
                v_ps[ell] = one_minus_lambda.clone() * &layer_proof.p_lefts[ell]
                    + &(lambda.clone() * &layer_proof.p_rights[ell]);
                v_qs[ell] = one_minus_lambda.clone() * &layer_proof.q_lefts[ell]
                    + &(lambda.clone() * &layer_proof.q_rights[ell]);
            }
            r_k = vec![lambda];
        } else {
            let sumcheck_proof = layer_proof
                .sumcheck_proof
                .as_ref()
                .ok_or(LookupError::GkrLeafMismatch)?;

            // Batched claimed sum = Σ_ℓ δ^ℓ · (v_p[ℓ] + α · v_q[ℓ])
            let mut claimed_sum = F::zero_with_cfg(field_cfg);
            for ell in 0..num_trees {
                let term = v_ps[ell].clone() + &(alpha.clone() * &v_qs[ell]);
                claimed_sum += &(delta_powers[ell].clone() * &term);
            }
            if sumcheck_proof.claimed_sum != claimed_sum {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: claimed_sum,
                    got: sumcheck_proof.claimed_sum.clone(),
                });
            }

            let subclaim = MLSumcheck::verify_as_subprotocol(
                transcript,
                k,
                3,
                sumcheck_proof,
                field_cfg,
            )?;

            let s = &subclaim.point;

            // Absorb per-tree evaluations and check combined final eval.
            let eq_val = zinc_poly::utils::eq_eval(s, &r_k, one.clone())?;
            let mut combined_inner = F::zero_with_cfg(field_cfg);

            for ell in 0..num_trees {
                let pl = &layer_proof.p_lefts[ell];
                let pr = &layer_proof.p_rights[ell];
                let ql = &layer_proof.q_lefts[ell];
                let qr = &layer_proof.q_rights[ell];

                transcript.absorb_random_field(pl, &mut buf);
                transcript.absorb_random_field(pr, &mut buf);
                transcript.absorb_random_field(ql, &mut buf);
                transcript.absorb_random_field(qr, &mut buf);

                let pl_plus_alpha_ql = pl.clone() + &(alpha.clone() * ql);
                let inner = pl_plus_alpha_ql * qr + &(pr.clone() * ql);
                combined_inner += &(delta_powers[ell].clone() * &inner);
            }

            let expected = eq_val * &combined_inner;
            if expected != subclaim.expected_evaluation {
                return Err(LookupError::FinalEvaluationMismatch {
                    expected: subclaim.expected_evaluation,
                    got: expected,
                });
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            let one_minus_lambda = one.clone() - &lambda;
            for ell in 0..num_trees {
                v_ps[ell] = one_minus_lambda.clone() * &layer_proof.p_lefts[ell]
                    + &(lambda.clone() * &layer_proof.p_rights[ell]);
                v_qs[ell] = one_minus_lambda.clone() * &layer_proof.q_lefts[ell]
                    + &(lambda.clone() * &layer_proof.q_rights[ell]);
            }
            r_k = s.clone();
            r_k.push(lambda);
        }
    }

    Ok(BatchedGkrFractionVerifyResult {
        point: r_k,
        expected_ps: v_ps,
        expected_qs: v_qs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::Blake3Transcript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    #[test]
    fn fraction_tree_single_leaf() {
        let p = vec![F::from(3u32)];
        let q = vec![F::from(7u32)];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].p[0], F::from(3u32));
        assert_eq!(tree[0].q[0], F::from(7u32));
    }

    #[test]
    fn fraction_tree_two_leaves() {
        let p = vec![F::from(3u32), F::from(5u32)];
        let q = vec![F::from(7u32), F::from(11u32)];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 2);
        let root = tree.last().unwrap();
        assert_eq!(root.p[0], F::from(68u32));
        assert_eq!(root.q[0], F::from(77u32));
    }

    #[test]
    fn fraction_tree_four_leaves() {
        let p = vec![F::from(1u32); 4];
        let q = vec![
            F::from(2u32),
            F::from(3u32),
            F::from(5u32),
            F::from(7u32),
        ];
        let tree = build_fraction_tree(p, q);
        assert_eq!(tree.len(), 3);
        let root = tree.last().unwrap();
        assert_eq!(root.p[0], F::from(247u32));
        assert_eq!(root.q[0], F::from(210u32));
    }

    #[test]
    fn gkr_fraction_prove_verify_two_leaves() {
        let p = vec![F::from(3u32), F::from(5u32)];
        let q = vec![F::from(7u32), F::from(11u32)];
        let tree = build_fraction_tree(p, q);

        let mut pt = Blake3Transcript::new();
        let (proof, eval_point) = gkr_fraction_prove(&mut pt, &tree, &());

        let mut vt = Blake3Transcript::new();
        let result = gkr_fraction_verify(&mut vt, &proof, 1, &())
            .expect("verifier should accept");

        assert_eq!(result.point.len(), 1);
        assert_eq!(eval_point.len(), 1);
    }

    #[test]
    fn gkr_fraction_prove_verify_eight_leaves() {
        let p: Vec<F> = (1..=8u32).map(F::from).collect();
        let q: Vec<F> = (10..=17u32).map(F::from).collect();
        let tree = build_fraction_tree(p, q);

        let mut pt = Blake3Transcript::new();
        let (proof, eval_point) = gkr_fraction_prove(&mut pt, &tree, &());

        let mut vt = Blake3Transcript::new();
        let result = gkr_fraction_verify(&mut vt, &proof, 3, &())
            .expect("verifier should accept");

        assert_eq!(result.point.len(), 3);
        assert_eq!(eval_point.len(), 3);
    }
}
