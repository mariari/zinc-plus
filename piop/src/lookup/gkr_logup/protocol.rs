//! Top-level GKR-LogUp prover and verifier per lookup group.
//!
//! Implements the chunks-in-clear polynomial-valued lift design:
//!
//! - **Chunks are NOT sent in the proof and NOT separately committed.**
//!   The prover sends per-`(ell, k)` polynomial-valued chunk lifts
//!   `c_k'^(ell) = MLE[v_k^(ell)](r_inner) ∈ F_q[X]_{<chunk_width}` —
//!   `chunk_width` field elements per (lookup, chunk).
//! - The witness-side GKR runs over ψ_a-projected chunk values; its
//!   leaf identity at the descent point `r = (r_inner, r_outer)`
//!   reduces to `expected_qs[ell] = β - Σ_k eq_outer(k, r_outer) ·
//!   ψ_a(c_k'^(ell))`.
//! - The verifier sub-claim returned to the protocol layer is the
//!   combined parent polynomial `c^(ell) = Σ_k X^{k·chunk_width} ·
//!   c_k'^(ell) = MLE[v^(ell)](r_inner)`. The protocol layer binds
//!   this to the parent column's PCS commitment by opening Zip+ at
//!   `r_inner` (a second opening, beyond the step-7 one at `r_0`).
//!
//! See `IMPLEMENTATION.md` (gleaming-pony plan) for the full design
//! discussion.

use std::collections::HashMap;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
    utils::{ArithErrors, build_eq_x_r_vec, eq_eval_at_index},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::LookupTableType;
use zinc_utils::{cfg_iter, cfg_into_iter, inner_transparent_field::InnerTransparentField};

use super::gkr::{
    batched_gkr_fraction_prove, batched_gkr_fraction_verify, build_fraction_tree,
    build_fraction_tree_ones_leaf, gkr_fraction_prove, gkr_fraction_verify,
    BatchedGkrFractionProveResult,
};
use super::structs::{
    GkrFractionProof, GkrLogupError, GkrLogupGroupMeta, GkrLogupGroupProof,
    GkrLogupGroupSubclaim,
};
use super::tables::{
    generate_bitpoly_table, generate_prescribed_table, generate_word_table,
    prescribed_multiplicities,
};

// ---------------------------------------------------------------------------
// Public input shape for `prove_group`.
// ---------------------------------------------------------------------------

/// Inputs to [`prove_group`] for a single lookup group of binary_poly
/// columns. MVP supports `binary_poly<D>` parents with
/// `LookupTableType::BitPoly { width: D, chunk_width: Some(cw) }` where
/// `cw` divides `D`. Each parent column appears as a
/// `DenseMultilinearExtension<BinaryPoly<D>>` value (the trace's
/// committed binary_poly column).
pub struct BinaryPolyLookupInstance<'a, F: PrimeField, const D: usize> {
    /// L parent column MLEs (binary_poly-valued).
    pub parent_columns: Vec<&'a DenseMultilinearExtension<BinaryPoly<D>>>,
    /// L flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type — must be `BitPoly { width: D, chunk_width: Some(cw) }`.
    pub table_type: LookupTableType,
    /// Projecting element `a` used by ψ_a, threaded from step 3.
    pub projecting_element_f: &'a F,
    /// Number of MLE variables of each parent column (= log2(W)).
    pub n_vars: usize,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove one GKR-LogUp lookup group with chunks-in-clear poly-lift.
///
/// Returns the lookup proof, the group meta to embed in the outer
/// proof, and the verifier sub-claim the protocol layer must discharge
/// against the parent column's PCS commitment via a Zip+ opening at
/// `r_inner`.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group<F, const D: usize>(
    transcript: &mut impl Transcript,
    instance: &BinaryPolyLookupInstance<'_, F, D>,
    field_cfg: &F::Config,
) -> Result<
    (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
    GkrLogupError<F>,
>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let (width, chunk_width) = match instance.table_type {
        LookupTableType::BitPoly { width, chunk_width: Some(cw) } => (width, cw),
        LookupTableType::BitPoly { width, chunk_width: None } => (width, width),
        _ => {
            return Err(GkrLogupError::WitnessNotInTable);
        }
    };
    assert_eq!(width, D, "table width must equal binary_poly degree D");
    assert!(chunk_width > 0 && width % chunk_width == 0, "chunk_width must divide width");
    let num_chunks = width / chunk_width;
    let num_lookups = instance.parent_columns.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;

    // ---- Step 1: Extract integer chunk indices directly from BitPoly bits ----
    //
    // For each (ell, k, i), the chunk's value is the integer
    //   n_{k,i} = Σ_{p=0..chunk_width} bit_{k·cw+p}(v^(ell)[i]) · 2^p
    // and the ψ_a-projected scalar is `subtable[n_{k,i}]` (where the
    // subtable is laid out so position `n` ↔ ψ_a of the BitPoly with
    // bit pattern `n`). This skips all field arithmetic for the
    // chunk-projection step — bit walks dominate.
    //
    // `chunks_idx[ell][k][i] ∈ [0, 2^chunk_width)` and is reused below
    // for both the multiplicity histogram (Step 3) and the witness
    // fraction tree's `leaf_q` construction (Step 5), avoiding any
    // intermediate `chunks_psi: Vec<Vec<Vec<F>>>` materialization.
    let a = instance.projecting_element_f.clone();
    let zero = F::zero_with_cfg(field_cfg);

    let chunks_idx: Vec<Vec<Vec<u32>>> = cfg_iter!(instance.parent_columns)
        .map(|parent| {
            // For each row, pack the entry's bits into a u32 once via
            // the BinaryPoly iter abstraction (works under both the
            // `Vec<Boolean>` and packed-u64 representations). Then
            // extract per-chunk indices by shifting.
            let row_packed: Vec<u32> = (0..witness_len)
                .map(|i| {
                    let mut packed: u32 = 0;
                    for (idx, b) in parent.evaluations[i].iter().enumerate() {
                        if b.into_inner() {
                            packed |= 1u32 << idx;
                        }
                    }
                    packed
                })
                .collect();
            let chunk_mask: u32 = if chunk_width == 32 {
                u32::MAX
            } else {
                (1u32 << chunk_width) - 1
            };
            (0..num_chunks)
                .map(|k| {
                    let shift = (k * chunk_width) as u32;
                    row_packed
                        .iter()
                        .map(|&p| (p >> shift) & chunk_mask)
                        .collect()
                })
                .collect()
        })
        .collect();

    // ---- Step 2: Build subtable T = ψ_a({0,1}^{<chunk_width}[X]) ----
    let subtable: Vec<F> = generate_bitpoly_table(chunk_width, &a, field_cfg);

    // ---- Steps 3-8: multiplicities, challenges, fraction trees, GKR ----
    //
    // Shared with every other table type: from here to the lifts, the
    // proof is a function of the chunk indices and the subtable alone.
    let agg_mults = histogram_multiplicities::<F>(&chunks_idx, subtable.len(), field_cfg);
    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Indexed(&chunks_idx),
        &subtable,
        &agg_mults,
        num_lookups,
        num_chunks,
        witness_len,
        field_cfg,
    );

    // ---- Step 9: Polynomial-valued chunk lifts ----
    //
    // For each (ell, k), c_k'^(ell) = MLE[v_k^(ell)](r_inner) is a
    // polynomial in F_q[X]_{<chunk_width}. We compute the parent's
    // full lifted eval (D coefficients) and split into chunks of
    // chunk_width coefficients each.
    let r_full = &witness_result.eval_point;
    assert!(
        r_full.len() >= n_vars,
        "GKR descent must have at least n_vars row variables"
    );
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();

    // Batch the L parent lifts so the eq(·, r_inner) table is built
    // once and the bit walks run in parallel across the L parents.
    let parent_lifts =
        compute_binary_poly_lifts::<F, D>(&instance.parent_columns, &r_inner, field_cfg);
    let chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>> = parent_lifts
        .into_iter()
        .map(|mut parent_lifted| {
            // `compute_binary_poly_lifts` returns trimmed polys; if the parent
            // column has structurally-zero high bits across all rows (e.g.
            // SHA's `S = w >> k` columns), the trimmed length can be < width.
            // Zero-pad up to `width` so chunk slicing always sees full chunks.
            if parent_lifted.coeffs.len() < width {
                parent_lifted.coeffs.resize(width, zero.clone());
            }
            (0..num_chunks)
                .map(|k| {
                    let lo = k * chunk_width;
                    let hi = lo + chunk_width;
                    DynamicPolynomialF::new_trimmed(parent_lifted.coeffs[lo..hi].to_vec())
                })
                .collect()
        })
        .collect();

    // ---- Step 10: Combine chunks → parent claim at r_inner ----
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = (0..num_lookups)
        .map(|ell| combine_chunks::<F>(&chunk_lifts[ell], chunk_width, width, &zero))
        .collect();

    // ---- Sanity (debug) ----
    debug_assert!({
        // Multiplicity sum invariant.
        let expected_per_lookup =
            F::from_with_cfg((num_chunks * witness_len) as u64, field_cfg);
        agg_mults.iter().all(|agg| {
            let sum: F = agg.iter().cloned().fold(zero.clone(), |a, b| a + &b);
            sum == expected_per_lookup
        })
    });

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups,
        num_chunks,
        chunk_width,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let proof = GkrLogupGroupProof {
        chunk_lifts,
        aggregated_multiplicities: agg_mults,
        witness_gkr: witness_result.proof,
        table_gkr,
        bin_lifts_at_r_inner: Vec::new(),
        int_evals_at_r_inner: Vec::new(),
    };
    let subclaim = GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial,
        parent_columns: meta.parent_columns.clone(),
    };
    Ok((proof, meta, subclaim))
}

/// The half of a lookup group's proof that does not care what the parent
/// columns are made of: the transcript's challenges, the witness and
/// table fraction trees, and both GKR runs.
///
/// Everything above this point differs per table type -- how a column's
/// cells become chunk indices, which subtable those index into, and where
/// the table's multiplicities come from -- and everything below it
/// differs again, in how the parent's lift is taken. In between, a chunk
/// index is a chunk index.
pub(super) struct FractionPhase<F: PrimeField> {
    pub witness_result: BatchedGkrFractionProveResult<F>,
    pub table_gkr: GkrFractionProof<F>,
}

/// Where a group's witness leaves come from.
///
/// A whole-column lookup gives every leaf the numerator one and reads its
/// denominator out of the table, so a cell is an index into it. A
/// selection reads the cell's own value -- the columns hold cells the
/// table says nothing about -- and the numerator is what says whether the
/// selection picked the cell.
pub(super) enum WitnessLeaves<'a, F: PrimeField> {
    /// `chunks_idx[ell][k][i]` is the subtable position of chunk `k` of
    /// row `i` of the `ell`-th parent column.
    Indexed(&'a [Vec<Vec<u32>>]),
    /// `cells` is the group's columns laid end to end, and
    /// `selections[ell]` names the positions in it that selection reads,
    /// once per time it reads them.
    Selected {
        cells: &'a [F],
        selections: &'a [Vec<u32>],
    },
}

/// The multiplicity of each table entry among the chunk indices: the
/// histogram a table with nothing prescribed about it has to be told.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn histogram_multiplicities<F>(
    chunks_idx: &[Vec<Vec<u32>>],
    table_len: usize,
    field_cfg: &F::Config,
) -> Vec<Vec<F>>
where
    F: PrimeField + FromPrimitiveWithConfig + Send + Sync,
    F::Config: Sync,
{
    // Direct array tally on integer chunk indices — no hashmap. Saves
    // L·K·W hashmap lookups (~2M at typical sizes) and frees the
    // table-index hashmap allocation.
    cfg_iter!(chunks_idx)
        .map(|lookup_chunks| {
            let mut counts = vec![0u64; table_len];
            for chunk in lookup_chunks {
                for &n in chunk {
                    counts[n as usize] += 1;
                }
            }
            counts
                .into_iter()
                .map(|c| F::from_with_cfg(c, field_cfg))
                .collect()
        })
        .collect()
}

#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub(super) fn prove_fraction_phase<F>(
    transcript: &mut impl Transcript,
    leaves: &WitnessLeaves<'_, F>,
    subtable: &[F],
    agg_mults: &[Vec<F>],
    num_lookups: usize,
    num_chunks: usize,
    witness_len: usize,
    field_cfg: &F::Config,
) -> FractionPhase<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let table_len = subtable.len();

    // ---- Step 4: Absorb agg multiplicities, sample β, α ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for agg in agg_mults {
        transcript.absorb_random_field_slice(agg, &mut buf);
    }
    let beta: F = transcript.get_field_challenge(field_cfg);
    let alpha: F = transcript.get_field_challenge(field_cfg);

    // α^ell powers.
    let mut alpha_powers = Vec::with_capacity(num_lookups);
    let mut ap = one.clone();
    for _ in 0..num_lookups {
        alpha_powers.push(ap.clone());
        ap = ap * &alpha;
    }

    // ---- Step 5: Build L witness fraction trees ----
    let per_lookup_leaves = num_chunks * witness_len; // K · W (always a power of 2)
    let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
    let w_size = 1usize << w_num_vars;
    let leaves_already_pow2 = per_lookup_leaves == w_size;

    // L witness fraction trees built in parallel — each tree's
    // construction is independent (different ell), and `build_fraction_tree`
    // itself is the heavy part (O(K·W) field ops per tree).
    let witness_trees: Vec<_> = match leaves {
        WitnessLeaves::Indexed(chunks_idx) => {
            // Pre-compute β − subtable[n] for each n ∈ [0, table_len). Per-leaf
            // construction below becomes a single index + clone (no field op).
            let beta_minus_subtable: Vec<F> =
                subtable.iter().map(|s| beta.clone() - s).collect();
            cfg_into_iter!(0..num_lookups)
                .map(|ell| {
                    let mut leaf_q = Vec::with_capacity(w_size);
                    for k in 0..num_chunks {
                        for i in 0..witness_len {
                            let n = chunks_idx[ell][k][i] as usize;
                            leaf_q.push(beta_minus_subtable[n].clone());
                        }
                    }
                    if leaves_already_pow2 {
                        build_fraction_tree_ones_leaf(one.clone(), leaf_q)
                    } else {
                        let mut leaf_p = vec![one.clone(); per_lookup_leaves];
                        leaf_p.resize(w_size, zero.clone());
                        leaf_q.resize(w_size, one.clone());
                        build_fraction_tree(leaf_p, leaf_q)
                    }
                })
                .collect()
        }
        WitnessLeaves::Selected { cells, selections } => {
            // Every selection reads the same cells, so the denominators are
            // built once and only the numerator tells the trees apart. A
            // cell nobody selected gets a zero there and contributes
            // nothing, which is also what the padding gets -- so a selected
            // group has no pad entry to pin and no row to slide a value into.
            let mut leaf_q: Vec<F> = cells.iter().map(|c| beta.clone() - c).collect();
            leaf_q.resize(w_size, one.clone());
            cfg_into_iter!(0..num_lookups)
                .map(|ell| {
                    let mut leaf_p = vec![zero.clone(); w_size];
                    for &j in &selections[ell] {
                        leaf_p[j as usize] += &one;
                    }
                    build_fraction_tree(leaf_p, leaf_q.clone())
                })
                .collect()
        }
    };

    // ---- Step 6: Build α-batched table fraction tree ----
    let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
    let t_size = 1usize << t_num_vars;
    let (mut t_leaf_p, mut t_leaf_q): (Vec<F>, Vec<F>) = cfg_into_iter!(0..table_len, 256)
        .map(|j| {
            let mut combined_mult = zero.clone();
            for ell in 0..num_lookups {
                combined_mult = combined_mult + &(alpha_powers[ell].clone() * &agg_mults[ell][j]);
            }
            (combined_mult, beta.clone() - &subtable[j])
        })
        .unzip();
    t_leaf_p.resize(t_size, zero.clone());
    t_leaf_q.resize(t_size, one.clone());
    let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

    // ---- Step 7: Witness GKR ----
    let witness_result = batched_gkr_fraction_prove(transcript, &witness_trees, field_cfg);

    // ---- Step 8: Table GKR ----
    let (table_gkr, _table_eval_point) = gkr_fraction_prove(transcript, &table_tree, field_cfg);

    FractionPhase { witness_result, table_gkr }
}

/// A prescribed cell is looked up whole, so its group has one chunk and
/// no width of its own; this is the width the rest of the protocol asks
/// for and never divides anything by.
const PRESCRIBED_CHUNK_WIDTH: usize = 1;

/// The table a `Prescribed` type names and the multiplicity of each of
/// its entries, both of which anyone holding the type can build.
///
/// Refused when the type cannot mean what it says: a pad that is one of
/// the values would leave that entry's multiplicity ambiguous, and values
/// that outnumber the rows cannot all be laid down.
fn prescribed_table_side<F>(
    values: &[u64],
    pad: u64,
    witness_len: usize,
    field_cfg: &F::Config,
) -> Result<(Vec<F>, Vec<F>), GkrLogupError<F>>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    if values.contains(&pad) || values.len() > witness_len {
        return Err(GkrLogupError::MalformedPrescribedTable);
    }
    Ok((
        generate_prescribed_table(values, pad, field_cfg),
        prescribed_multiplicities(values.len(), witness_len, field_cfg),
    ))
}

/// The prescribed half of [`prove_group`]: the column is the table.
///
/// A cell is looked up whole -- one chunk, since the table is a multiset
/// and not a range to decompose -- and the multiplicities are the table's
/// own rather than a count of what the witness happened to hold. That is
/// what makes the LogUp identity say multiset equality: the column holds
/// each value once and the pad in every other row, or the identity fails.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group_prescribed<F, I>(
    transcript: &mut impl Transcript,
    instance: &IntLookupInstance<'_, I>,
    field_cfg: &F::Config,
) -> Result<
    (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
    GkrLogupError<F>,
>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
    I: ConstTranscribable + Clone + Send + Sync,
{
    let (values, pad) = match &instance.table_type {
        LookupTableType::Prescribed { values, pad } => (values, *pad),
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    let num_lookups = instance.parent_columns.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;
    let zero = F::zero_with_cfg(field_cfg);

    // ---- Steps 1-2: the table, and where each cell sits in it ----
    //
    // A value the table does not name has no place to sit, so the column
    // is refused here rather than proved against a table it misses.
    let (subtable, multiplicities) =
        prescribed_table_side::<F>(values, pad, witness_len, field_cfg)?;
    let mut position: HashMap<u64, u32> = HashMap::with_capacity(subtable.len());
    for (index, entry) in values.iter().chain(std::iter::once(&pad)).enumerate() {
        position.insert(*entry, u32::try_from(index).expect("a table this long has no index"));
    }
    let mut buf = vec![0u8; I::NUM_BYTES];
    let mut chunks_idx: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_lookups);
    for parent in &instance.parent_columns {
        let mut chunk = Vec::with_capacity(witness_len);
        for i in 0..witness_len {
            let value = int_table_index(&parent.evaluations[i], 64, &mut buf);
            match value.and_then(|value| position.get(&value)) {
                Some(index) => chunk.push(*index),
                None => return Err(GkrLogupError::WitnessNotInTable),
            }
        }
        chunks_idx.push(vec![chunk]);
    }

    // Every column is checked against the same prescribed table, so every
    // one of them carries the same multiplicities.
    let agg_mults = vec![multiplicities; num_lookups];
    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Indexed(&chunks_idx),
        &subtable,
        &agg_mults,
        num_lookups,
        1,
        witness_len,
        field_cfg,
    );

    // ---- Step 9: the entry a cell indexes is the cell's own value ----
    //
    // So the single chunk's lift is the parent column's multilinear
    // evaluation at r_inner: the same number the witness leaf check
    // reconstructs and the protocol layer binds against the commitment.
    let r_full = &witness_result.eval_point;
    assert!(r_full.len() >= n_vars, "GKR descent must have at least n_vars row variables");
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let eq_table = build_eq_x_r_vec(&r_inner, field_cfg)?;

    let chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>> = chunks_idx
        .iter()
        .map(|per_chunk| {
            per_chunk
                .iter()
                .map(|chunk| {
                    let mut acc = zero.clone();
                    for (i, n) in chunk.iter().enumerate() {
                        acc += &(eq_table[i].clone() * &subtable[*n as usize]);
                    }
                    DynamicPolynomialF::new_trimmed(vec![acc])
                })
                .collect()
        })
        .collect();
    let combined_polynomial: Vec<DynamicPolynomialF<F>> =
        chunk_lifts.iter().map(|lifts| lifts[0].clone()).collect();

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups,
        num_chunks: 1,
        chunk_width: PRESCRIBED_CHUNK_WIDTH,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let proof = GkrLogupGroupProof {
        chunk_lifts,
        // The verifier builds this table's multiplicities itself, so the
        // proof does not carry them.
        aggregated_multiplicities: Vec::new(),
        witness_gkr: witness_result.proof,
        table_gkr,
        bin_lifts_at_r_inner: Vec::new(),
        int_evals_at_r_inner: Vec::new(),
    };
    let subclaim = GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial,
        parent_columns: meta.parent_columns.clone(),
    };
    Ok((proof, meta, subclaim))
}

/// Inputs to [`prove_group_selected`] for a group whose table names the
/// cells it speaks for.
///
/// The columns arrive already projected into the field: a cell no
/// selection names still sits in a denominator, and no table says what it
/// may be, so its value has to be the one the commitment carries rather
/// than one a table can index.
pub struct SelectedLookupInstance<'a, F: PrimeField> {
    /// The C columns this group's selections reach across.
    pub parent_columns: Vec<&'a DenseMultilinearExtension<F::Inner>>,
    /// C flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type -- must be `Selected`.
    pub table_type: LookupTableType,
    /// Number of MLE variables of each column (= log2(W)).
    pub n_vars: usize,
}

/// The selections a permuted table makes, the firsts of its pairs then the
/// seconds: the tree at `ell + pairs` answers the tree at `ell`.
fn paired_selections(pairs: &[(Vec<(u32, u32)>, Vec<(u32, u32)>)]) -> Vec<Vec<(u32, u32)>> {
    pairs
        .iter()
        .map(|(first, _)| first.clone())
        .chain(pairs.iter().map(|(_, second)| second.clone()))
        .collect()
}

/// The weight each tree's fraction carries when the roots are crossed:
/// α^ell, and for the second of a permuted pair the negation of its
/// first's, so the pair cancels exactly when it holds one multiset.
#[allow(clippy::arithmetic_side_effects)]
fn tree_weights<F>(table_type: &LookupTableType, alpha_powers: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    match table_type {
        LookupTableType::Permuted { pairs } => {
            let half = pairs.len();
            (0..alpha_powers.len())
                .map(|ell| {
                    if ell < half {
                        alpha_powers[ell].clone()
                    } else {
                        -alpha_powers[ell - half].clone()
                    }
                })
                .collect()
        }
        _ => alpha_powers.to_vec(),
    }
}

/// The flat leaf position of every cell each selection names, over the
/// group's columns laid end to end.
///
/// Refused when a selection names a cell the group does not have: a slot
/// past its columns or a row past its rows indexes nothing at all.
#[allow(clippy::arithmetic_side_effects)]
fn selection_positions<F: PrimeField>(
    selections: &[Vec<(u32, u32)>],
    num_columns: usize,
    witness_len: usize,
) -> Result<Vec<Vec<u32>>, GkrLogupError<F>> {
    selections
        .iter()
        .map(|selection| {
            selection
                .iter()
                .map(|&(slot, row)| {
                    let (slot, row) = (slot as usize, row as usize);
                    if slot >= num_columns || row >= witness_len {
                        return Err(GkrLogupError::MalformedSelection);
                    }
                    u32::try_from(slot * witness_len + row)
                        .map_err(|_| GkrLogupError::MalformedSelection)
                })
                .collect()
        })
        .collect()
}

/// The selected half of [`prove_group`]: the table names its own cells.
///
/// Where a prescribed table speaks for a whole column and pads the rest,
/// a selection speaks for the cells the signature names and says nothing
/// about the others. So the group runs one tree per selection over the
/// same columns, the numerator carrying the selection and the denominator
/// the column's own value -- and an unselected cell, numerator zero,
/// contributes exactly nothing to either side.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group_selected<F>(
    transcript: &mut impl Transcript,
    instance: &SelectedLookupInstance<'_, F>,
    field_cfg: &F::Config,
) -> Result<
    (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
    GkrLogupError<F>,
>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    // A permuted group is a selected one whose table is empty: the second
    // selection of each pair is a tree of its own, weighed against the
    // first when the roots are crossed.
    let (values, selections) = match &instance.table_type {
        LookupTableType::Selected { values, selections } => (values.clone(), selections.clone()),
        LookupTableType::Permuted { pairs } => (Vec::new(), paired_selections(pairs)),
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    let num_columns = instance.parent_columns.len();
    let num_lookups = selections.len();
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;
    let positions = selection_positions::<F>(&selections, num_columns, witness_len)?;

    // The cells the selections read: the group's columns end to end, in
    // the order the specs declare them.
    let cells: Vec<F> = instance
        .parent_columns
        .iter()
        .flat_map(|column| {
            column
                .evaluations
                .iter()
                .map(|value| F::new_unchecked_with_cfg(value.clone(), field_cfg))
        })
        .collect();

    // The table is the multiset itself, each entry once. Nothing here
    // comes from the witness, so the proof carries no multiplicities.
    let subtable: Vec<F> = values
        .iter()
        .map(|value| F::from_with_cfg(*value, field_cfg))
        .collect();
    let agg_mults = vec![vec![F::one_with_cfg(field_cfg); values.len()]; num_lookups];

    let FractionPhase { witness_result, table_gkr } = prove_fraction_phase(
        transcript,
        &WitnessLeaves::Selected { cells: &cells, selections: &positions },
        &subtable,
        &agg_mults,
        num_lookups,
        num_columns,
        witness_len,
        field_cfg,
    );

    // One claim per column, shared by every selection that reaches into
    // it: the column's own multilinear evaluation at r_inner, which is
    // what the protocol layer binds against the commitment.
    let r_full = &witness_result.eval_point;
    assert!(r_full.len() >= n_vars, "GKR descent must have at least n_vars row variables");
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let eq_table = build_eq_x_r_vec(&r_inner, field_cfg)?;
    let column_lifts: Vec<DynamicPolynomialF<F>> = cells
        .chunks_exact(witness_len)
        .map(|column| {
            let mut acc = F::zero_with_cfg(field_cfg);
            for (eq, cell) in eq_table.iter().zip(column) {
                acc += &(eq.clone() * cell);
            }
            DynamicPolynomialF::new_trimmed(vec![acc])
        })
        .collect();

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups,
        num_chunks: num_columns,
        chunk_width: PRESCRIBED_CHUNK_WIDTH,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let proof = GkrLogupGroupProof {
        // Every selection reads the same columns, so the group's lifts are
        // one row rather than one per selection.
        chunk_lifts: vec![column_lifts.clone()],
        aggregated_multiplicities: Vec::new(),
        witness_gkr: witness_result.proof,
        table_gkr,
        bin_lifts_at_r_inner: Vec::new(),
        int_evals_at_r_inner: Vec::new(),
    };
    let subclaim = GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial: column_lifts,
        parent_columns: meta.parent_columns.clone(),
    };
    Ok((proof, meta, subclaim))
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify one GKR-LogUp lookup group's proof. Returns the verifier
/// sub-claim that the protocol layer must discharge by opening Zip+ on
/// the parent column at `subclaim.r_inner` and matching against
/// `subclaim.combined_polynomial`.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_group<F>(
    transcript: &mut impl Transcript,
    proof: &GkrLogupGroupProof<F>,
    meta: &GkrLogupGroupMeta,
    projecting_element_f: &F,
    field_cfg: &F::Config,
) -> Result<GkrLogupGroupSubclaim<F>, GkrLogupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let (width, chunk_width) = match &meta.table_type {
        LookupTableType::BitPoly { width, chunk_width: Some(cw) } => (*width, *cw),
        LookupTableType::BitPoly { width, chunk_width: None } => (*width, *width),
        LookupTableType::Word { .. } => return Err(GkrLogupError::WitnessNotInTable),
        LookupTableType::Prescribed { .. }
        | LookupTableType::Selected { .. }
        | LookupTableType::Permuted { .. } => (PRESCRIBED_CHUNK_WIDTH, PRESCRIBED_CHUNK_WIDTH),
    };
    assert!(chunk_width > 0 && width % chunk_width == 0);

    // Every selection reads the group's columns, so the group carries one
    // row of lifts; a whole-column lookup reads only its own parent, so it
    // carries one row per parent.
    let selections = match &meta.table_type {
        LookupTableType::Selected { selections, .. } => Some(selections.clone()),
        LookupTableType::Permuted { pairs } => Some(paired_selections(pairs)),
        _ => None,
    };
    let selections = selections.as_deref();
    let num_lookups = meta.num_lookups;
    let num_chunks = meta.num_chunks;
    let witness_len = meta.witness_len;
    let n_vars = (witness_len as f64).log2() as usize;
    assert_eq!(1usize << n_vars, witness_len, "witness_len must be a power of 2");
    match selections {
        // A selection's leaves run over the group's own columns, a block
        // each; every other table decomposes a cell into chunks.
        Some(sels) => {
            assert_eq!(num_chunks, meta.parent_columns.len());
            assert_eq!(num_lookups, sels.len());
        }
        None => assert_eq!(num_chunks, width / chunk_width),
    }
    let lift_rows = if selections.is_some() { 1 } else { num_lookups };
    if proof.chunk_lifts.len() != lift_rows {
        return Err(GkrLogupError::GkrLeafMismatch);
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    // ---- Reconstruct subtable + shifts ----
    // The subtable a chunk index reads against. A prescribed or selected
    // table is its values, and its multiplicities come with them: the
    // verifier builds that whole side and takes none of it from the proof.
    let (subtable, prescribed_mults) = match &meta.table_type {
        LookupTableType::Prescribed { values, pad } => {
            let (table, multiplicities) =
                prescribed_table_side::<F>(values, *pad, witness_len, field_cfg)?;
            (table, Some(vec![multiplicities; num_lookups]))
        }
        LookupTableType::Selected { values, .. } => {
            // No pad: a cell no selection names is not looked up at all,
            // so there is no row left over for the table to account for.
            let table: Vec<F> = values
                .iter()
                .map(|value| F::from_with_cfg(*value, field_cfg))
                .collect();
            let multiplicities = vec![one.clone(); values.len()];
            (table, Some(vec![multiplicities; num_lookups]))
        }
        // No table at all: the second selections are the first ones' table.
        LookupTableType::Permuted { .. } => (Vec::new(), Some(vec![Vec::new(); num_lookups])),
        _ => (
            generate_bitpoly_table::<F>(chunk_width, projecting_element_f, field_cfg),
            None,
        ),
    };
    let table_len = subtable.len();
    let aggregated_multiplicities = prescribed_mults
        .as_deref()
        .unwrap_or(&proof.aggregated_multiplicities);
    assert_eq!(aggregated_multiplicities.len(), num_lookups);

    // ---- Step 1: Absorb agg multiplicities, sample β, α ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    for agg in aggregated_multiplicities {
        transcript.absorb_random_field_slice(agg, &mut buf);
    }
    let beta: F = transcript.get_field_challenge(field_cfg);
    let alpha: F = transcript.get_field_challenge(field_cfg);

    let mut alpha_powers = Vec::with_capacity(num_lookups);
    let mut ap = one.clone();
    for _ in 0..num_lookups {
        alpha_powers.push(ap.clone());
        ap = ap * &alpha;
    }

    // ---- Step 2: Witness + table GKR verify ----
    let per_lookup_leaves = num_chunks * witness_len;
    let w_num_vars = zinc_utils::log2(per_lookup_leaves.next_power_of_two()) as usize;
    let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;

    let witness_result =
        batched_gkr_fraction_verify(transcript, &proof.witness_gkr, w_num_vars, field_cfg)?;
    let table_result =
        gkr_fraction_verify(transcript, &proof.table_gkr, t_num_vars, field_cfg)?;

    // ---- Step 3: Cross-check roots ----
    {
        let weights = tree_weights(&meta.table_type, &alpha_powers);
        let roots_q = &proof.witness_gkr.roots_q;
        let q_w_product: F = roots_q.iter().cloned().fold(one.clone(), |acc, q| acc * &q);
        let mut lhs = zero.clone();
        if num_lookups == 1 {
            lhs = lhs + &(weights[0].clone() * &proof.witness_gkr.roots_p[0]);
        } else if num_lookups > 1 {
            let mut prefix = Vec::with_capacity(num_lookups);
            prefix.push(one.clone());
            for i in 1..num_lookups {
                prefix.push(prefix[i - 1].clone() * &roots_q[i - 1]);
            }
            let mut suffix = vec![one.clone(); num_lookups];
            for i in (0..num_lookups - 1).rev() {
                suffix[i] = suffix[i + 1].clone() * &roots_q[i + 1];
            }
            for ell in 0..num_lookups {
                let others_q = prefix[ell].clone() * &suffix[ell];
                lhs = lhs + &(weights[ell].clone() * &proof.witness_gkr.roots_p[ell] * &others_q);
            }
        }
        lhs = lhs * &proof.table_gkr.root_q;
        let rhs = proof.table_gkr.root_p.clone() * &q_w_product;
        if lhs != rhs {
            return Err(GkrLogupError::GkrRootMismatch);
        }
    }

    // ---- Step 4: Multiplicity sums + table-side leaf check ----
    //
    // How many leaves each tree's numerators turn on: a whole-column
    // lookup turns on every leaf it has, a selection only the cells it
    // names. The table side has to account for exactly that many.
    let leaves_read = |ell: usize| match selections {
        Some(sels) => sels[ell].len(),
        None => num_chunks * witness_len,
    };
    let combined_mults: Vec<F> = {
        let mut combined = vec![zero.clone(); table_len];
        for ell in 0..num_lookups {
            let alpha_ell = &alpha_powers[ell];
            let mut m_sum = zero.clone();
            for j in 0..table_len {
                let scaled = alpha_ell.clone() * &aggregated_multiplicities[ell][j];
                combined[j] = combined[j].clone() + &scaled;
                m_sum = m_sum + &aggregated_multiplicities[ell][j];
            }
            // A permuted group has no table to account for its leaves.
            let expected = if table_len == 0 { 0 } else { leaves_read(ell) as u64 };
            if m_sum != F::from_with_cfg(expected, field_cfg) {
                return Err(GkrLogupError::MultiplicitySumMismatch { expected, got: 0 });
            }
        }
        combined
    };

    if table_result.point.is_empty() {
        let expected_p = if table_len > 0 { combined_mults[0].clone() } else { zero.clone() };
        let expected_q = if table_len > 0 { beta.clone() - &subtable[0] } else { one.clone() };
        if expected_p != table_result.expected_p || expected_q != table_result.expected_q {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    } else {
        let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
        let mut p_eval = zero.clone();
        let mut q_eval = zero.clone();
        for j in 0..table_len {
            p_eval = p_eval + &(combined_mults[j].clone() * &eq_at_t[j]);
            q_eval = q_eval + &((beta.clone() - &subtable[j]) * &eq_at_t[j]);
        }
        for j in table_len..eq_at_t.len() {
            q_eval = q_eval + &eq_at_t[j];
        }
        if p_eval != table_result.expected_p {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
        if q_eval != table_result.expected_q {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // ---- Step 5: Witness-side leaf check using chunk lifts ----
    //
    // r = (r_inner, r_outer) with r_inner of length n_vars (low bits)
    // and r_outer of length log2(K) (high bits).
    let r_full = &witness_result.point;
    assert_eq!(r_full.len(), w_num_vars);
    assert_eq!(
        w_num_vars,
        n_vars + zinc_utils::log2(num_chunks.next_power_of_two()) as usize
    );
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let r_outer: Vec<F> = r_full[n_vars..].to_vec();

    // A single chunk descends on no outer variables at all, and the empty
    // product is one.
    let eq_at_outer = match r_outer.is_empty() {
        true => vec![one.clone()],
        false => build_eq_x_r_vec(&r_outer, field_cfg)?,
    };
    // How much of the outer cube the group's leaves cover. The rest is
    // padding, whose numerator is zero and denominator one, so it adds
    // `1 - covered` to every q and nothing to any p. Whenever the leaf
    // count is a power of two this is one and both corrections vanish.
    let covered: F = eq_at_outer[..num_chunks]
        .iter()
        .cloned()
        .fold(zero.clone(), |acc, eq| acc + &eq);
    let padding = one.clone() - &covered;

    // expected_p^(ell)(r) is the numerator MLE at the descent point: the
    // covered mass where every leaf is read, and Σ eq over the named cells
    // where a selection says which. The selection is declared, so the
    // verifier sums those eq terms itself -- nothing about it is
    // committed, and it costs the cells it names rather than the trace.
    for ell in 0..num_lookups {
        let expected_p = match selections {
            Some(sels) => sels[ell].iter().fold(zero.clone(), |acc, &(slot, row)| {
                let eq_row = eq_eval_at_index(&r_inner, row as usize, &one);
                acc + &(eq_at_outer[slot as usize].clone() * &eq_row)
            }),
            None => covered.clone(),
        };
        if witness_result.expected_ps[ell] != expected_p {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // For q^(ell), reconstruct from the lifts of the columns the tree's
    // denominators read:
    //   expected_qs[ell] = β·covered - Σ_k eq_outer(k, r_outer) · ψ_a(c_k')
    //                       + padding
    for ell in 0..num_lookups {
        let lifts = &proof.chunk_lifts[if selections.is_some() { 0 } else { ell }];
        if lifts.len() != num_chunks {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
        // The protocol binds only the first coefficient of an int lift; reject longer ones.
        if meta.table_type.reads_int_columns() && lifts.iter().any(|lift| lift.coeffs.len() > 1) {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
        let mut psi_combined = zero.clone();
        for k in 0..num_chunks {
            let psi =
                eval_at_projecting_element::<F>(&lifts[k], projecting_element_f, field_cfg);
            psi_combined = psi_combined + &(eq_at_outer[k].clone() * &psi);
        }
        let expected_q_local = beta.clone() * &covered - &psi_combined + &padding;
        if expected_q_local != witness_result.expected_qs[ell] {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // ---- Step 6: Combine chunk lifts into parent polynomial claim ----
    let combined_polynomial: Vec<DynamicPolynomialF<F>> = match &meta.table_type {
        // A selection's chunks are whole columns, each standing for
        // itself, so the group's claim is one lift per column and there
        // is nothing to recombine.
        LookupTableType::Selected { .. } | LookupTableType::Permuted { .. } => {
            proof.chunk_lifts[0].clone()
        }
        // BitPoly chunks are coefficient blocks.
        LookupTableType::BitPoly { .. } => (0..num_lookups)
            .map(|ell| combine_chunks::<F>(&proof.chunk_lifts[ell], chunk_width, width, &zero))
            .collect(),
        // A prescribed cell is looked up whole: its one chunk is the claim.
        _ => proof.chunk_lifts.iter().map(|lifts| lifts[0].clone()).collect(),
    };

    Ok(GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial,
        parent_columns: meta.parent_columns.clone(),
    })
}

// ---------------------------------------------------------------------------
// Int columns against a `Word` table (range checks on integer cells)
// ---------------------------------------------------------------------------

/// Inputs to [`prove_group_int`]: a group of witness **int** columns
/// range-checked against `LookupTableType::Word { width, chunk_width: None }`,
/// i.e. every cell must be an integer in `[0, 2^width)`.
///
/// The group is proved as ONE LogUp instance whose "chunks" are the `L`
/// columns themselves: the witness fraction tree has `L · W` leaves
/// (`leaf[k·W + i] = (1, β − v_k[i])`), there is a single aggregated
/// multiplicity vector over the `2^width` table, and the GKR descent point
/// splits as `r = (r_inner ∈ F^{n_vars}, r_outer ∈ F^{⌈log2 L⌉})` with the
/// leaf identity `q̃(r) = β − Σ_k eq(k, r_outer) · MLE[v_k](r_inner)` (plus
/// the padding term when `L · W` is not a power of two).
///
/// The prover sends the `L` scalars `MLE[v_k](r_inner)` as degree-0
/// `chunk_lifts[0][k]`; the protocol layer binds them to the int
/// commitment with the int multipoint reducer and a single Zip+ open.
///
/// Soundness of the range check: the int commitment's extractor yields a
/// bounded-height rational `v = a/b` per cell; the LogUp forces
/// `a·b⁻¹ ≡ t (mod q)` for a table entry `t < 2^width`, i.e.
/// `q | (a − t·b)`, and since `|a − t·b| < q` for the prime sizes in use
/// this gives `v = t` exactly — an integer in range.
pub struct IntLookupInstance<'a, I> {
    /// L parent int columns (the committed witness int MLEs).
    pub parent_columns: Vec<&'a DenseMultilinearExtension<I>>,
    /// L flat-trace column indices, mirrored into the proof's group meta.
    pub parent_column_indices: Vec<usize>,
    /// Lookup table type — must be `Word { width, chunk_width: None }`.
    pub table_type: LookupTableType,
    /// Number of MLE variables of each parent column (= log2(W)).
    pub n_vars: usize,
}

/// Read an int cell as a `Word` table index.
///
/// The cell's little-endian transcription bytes must encode a
/// non-negative integer below `2^width`: every byte beyond the low eight
/// must be zero (which also rejects negative two's-complement values), and
/// the low 64-bit value must be `< 2^width`. `buf` must be
/// `I::NUM_BYTES` long (scratch space).
#[allow(clippy::arithmetic_side_effects)] // `width < 64` is checked before the shift
pub fn int_table_index<I: ConstTranscribable>(v: &I, width: usize, buf: &mut [u8]) -> Option<u64> {
    debug_assert_eq!(buf.len(), I::NUM_BYTES);
    v.write_transcription_bytes_exact(buf);
    let n = buf.len().min(8);
    let mut le = [0u8; 8];
    le[..n].copy_from_slice(&buf[..n]);
    if buf[n..].iter().any(|&b| b != 0) {
        return None;
    }
    let val = u64::from_le_bytes(le);
    if width < 64 && val >= (1u64 << width) {
        return None;
    }
    Some(val)
}

/// The scalar carried by a degree-0 lift (`[]` after trimming means zero).
pub fn lift_scalar<F: PrimeField>(poly: &DynamicPolynomialF<F>, zero: &F) -> F {
    poly.coeffs.first().cloned().unwrap_or_else(|| zero.clone())
}

/// `eq(·, point)` table, or the one-entry table `[1]` for an empty point
/// (a single chunk / a single column leaves no outer variables).
fn eq_table_or_unit<F: PrimeField>(point: &[F], field_cfg: &F::Config) -> Result<Vec<F>, ArithErrors> {
    if point.is_empty() {
        Ok(vec![F::one_with_cfg(field_cfg)])
    } else {
        build_eq_x_r_vec(point, field_cfg)
    }
}

/// Scalar MLE evaluations of a batch of `F`-valued columns (given as
/// Montgomery-inner MLEs, e.g. the projected witness int columns) at a
/// shared point: `Σ_i eq(i, point) · col[i]`. Builds the eq table once and
/// parallelizes across columns.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_int_column_evals<F>(
    cols: &[&DenseMultilinearExtension<F::Inner>],
    point: &[F],
    field_cfg: &F::Config,
) -> Vec<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Send + Sync,
    F::Config: Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let eq_table =
        build_eq_x_r_vec(point, field_cfg).expect("compute_int_column_evals: eq table build failed");
    cfg_iter!(cols)
        .map(|col| {
            let mut acc = zero.clone();
            for (i, inner) in col.evaluations.iter().enumerate() {
                let v = F::new_unchecked_with_cfg(inner.clone(), field_cfg);
                acc = acc + &(eq_table[i].clone() * &v);
            }
            acc
        })
        .collect()
}

/// Prove one `Word`-table lookup group over int columns (see
/// [`IntLookupInstance`]).
///
/// Returns the lookup proof, the group meta to embed in the outer proof,
/// and the verifier sub-claim: `combined_polynomial[ell]` is the degree-0
/// lift holding `MLE[v_ell](r_inner)`, which the protocol layer must bind
/// to the int commitment (int multipoint reducer + Zip+ open).
///
/// Fails with [`GkrLogupError::WitnessNotInTable`] if any cell is not an
/// integer in `[0, 2^width)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_group_int<F, I>(
    transcript: &mut impl Transcript,
    instance: &IntLookupInstance<'_, I>,
    field_cfg: &F::Config,
) -> Result<
    (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
    GkrLogupError<F>,
>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
    I: ConstTranscribable + Send + Sync,
{
    let width = match instance.table_type {
        LookupTableType::Word { width, chunk_width: None } => width,
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    assert!(
        (1..=24).contains(&width),
        "Word table width must be in 1..=24 (the table has 2^width entries), got {width}"
    );
    let num_cols = instance.parent_columns.len();
    assert!(num_cols > 0, "int lookup group needs at least one column");
    assert_eq!(num_cols, instance.parent_column_indices.len());
    let n_vars = instance.n_vars;
    let witness_len = 1usize << n_vars;
    for col in &instance.parent_columns {
        assert_eq!(col.evaluations.len(), witness_len, "int column length must be 2^n_vars");
    }
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);

    // ---- Step 1: table index of every cell (rejects out-of-range cells) ----
    let idx: Vec<Vec<u32>> = cfg_iter!(instance.parent_columns)
        .map(|col| {
            let mut buf = vec![0u8; I::NUM_BYTES];
            col.evaluations
                .iter()
                .map(|v| int_table_index(v, width, &mut buf).map(|n| n as u32))
                .collect::<Option<Vec<u32>>>()
        })
        .collect::<Option<Vec<Vec<u32>>>>()
        .ok_or(GkrLogupError::WitnessNotInTable)?;

    // ---- Step 2: the table T = {0, …, 2^width − 1} over F ----
    let table: Vec<F> = generate_word_table::<F>(width, field_cfg);
    let table_len = table.len();

    // ---- Step 3: one multiplicity histogram over the whole group ----
    // (per-thread partial histograms over columns, summed.)
    let counts: Vec<u64> = {
        let fold_col = |mut hist: Vec<u64>, col: &Vec<u32>| {
            for &n in col {
                hist[n as usize] += 1;
            }
            hist
        };
        let sum = |mut a: Vec<u64>, b: Vec<u64>| {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
            a
        };
        #[cfg(feature = "parallel")]
        {
            idx.par_iter()
                .fold(|| vec![0u64; table_len], fold_col)
                .reduce(|| vec![0u64; table_len], sum)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = sum;
            idx.iter().fold(vec![0u64; table_len], fold_col)
        }
    };
    let agg_mults: Vec<Vec<F>> =
        vec![counts.into_iter().map(|c| F::from_with_cfg(c, field_cfg)).collect()];

    // ---- Step 4: absorb the multiplicities, sample β (and α, unused with a
    // single instance but kept for transcript symmetry with the bin path) ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(&agg_mults[0], &mut buf);
    let beta: F = transcript.get_field_challenge(field_cfg);
    let _alpha: F = transcript.get_field_challenge(field_cfg);

    // ---- Step 5: the witness fraction tree over L·W leaves ----
    let leaves = num_cols * witness_len;
    let w_num_vars = zinc_utils::log2(leaves.next_power_of_two()) as usize;
    let w_size = 1usize << w_num_vars;
    let beta_minus_table: Vec<F> = table.iter().map(|t| beta.clone() - t).collect();
    let mut leaf_q: Vec<F> = Vec::with_capacity(w_size);
    #[cfg(feature = "parallel")]
    leaf_q.par_extend(
        idx.par_iter()
            .flat_map_iter(|col| col.iter().map(|&n| beta_minus_table[n as usize].clone())),
    );
    #[cfg(not(feature = "parallel"))]
    leaf_q.extend(
        idx.iter()
            .flat_map(|col| col.iter().map(|&n| beta_minus_table[n as usize].clone())),
    );
    let witness_tree = if leaves == w_size {
        build_fraction_tree_ones_leaf(one.clone(), leaf_q)
    } else {
        let mut leaf_p = vec![one.clone(); leaves];
        leaf_p.resize(w_size, zero.clone());
        leaf_q.resize(w_size, one.clone());
        build_fraction_tree(leaf_p, leaf_q)
    };

    // ---- Step 6: the table fraction tree ----
    let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
    let t_size = 1usize << t_num_vars;
    let mut t_leaf_p: Vec<F> = agg_mults[0].clone();
    let mut t_leaf_q: Vec<F> = beta_minus_table;
    t_leaf_p.resize(t_size, zero.clone());
    t_leaf_q.resize(t_size, one.clone());
    let table_tree = build_fraction_tree(t_leaf_p, t_leaf_q);

    // ---- Step 7/8: GKRs ----
    let witness_result = batched_gkr_fraction_prove(transcript, &[witness_tree], field_cfg);
    let (table_gkr, _table_eval_point) = gkr_fraction_prove(transcript, &table_tree, field_cfg);

    // ---- Step 9: per-column evals at r_inner (the degree-0 "chunk lifts") ----
    let r_full = &witness_result.eval_point;
    assert!(r_full.len() >= n_vars, "GKR descent must have at least n_vars row variables");
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let eq_inner = build_eq_x_r_vec(&r_inner, field_cfg).map_err(GkrLogupError::EqBuildError)?;
    let evals: Vec<F> = cfg_iter!(idx)
        .map(|col| {
            let mut acc = zero.clone();
            for (i, &n) in col.iter().enumerate() {
                acc = acc + &(eq_inner[i].clone() * &table[n as usize]);
            }
            acc
        })
        .collect();
    let chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>> = vec![
        evals
            .iter()
            .map(|e| DynamicPolynomialF::new_trimmed(vec![e.clone()]))
            .collect(),
    ];

    let meta = GkrLogupGroupMeta {
        table_type: instance.table_type.clone(),
        num_lookups: 1,
        num_chunks: num_cols,
        chunk_width: width,
        witness_len,
        parent_columns: instance.parent_column_indices.clone(),
    };
    let subclaim = GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial: chunk_lifts[0].clone(),
        parent_columns: meta.parent_columns.clone(),
    };
    let proof = GkrLogupGroupProof {
        chunk_lifts,
        aggregated_multiplicities: agg_mults,
        witness_gkr: witness_result.proof,
        table_gkr,
        bin_lifts_at_r_inner: Vec::new(),
        int_evals_at_r_inner: Vec::new(),
    };
    Ok((proof, meta, subclaim))
}

/// Verify one `Word`-table lookup group over int columns. Returns the
/// sub-claim `(r_inner, [MLE[v_ell](r_inner)]_ell)` the protocol layer
/// must bind to the int commitment.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_group_int<F>(
    transcript: &mut impl Transcript,
    proof: &GkrLogupGroupProof<F>,
    meta: &GkrLogupGroupMeta,
    field_cfg: &F::Config,
) -> Result<GkrLogupGroupSubclaim<F>, GkrLogupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let width = match &meta.table_type {
        LookupTableType::Word { width, chunk_width: None } => *width,
        _ => return Err(GkrLogupError::WitnessNotInTable),
    };
    if !(1..=24).contains(&width) {
        return Err(GkrLogupError::WitnessNotInTable);
    }
    let num_cols = meta.num_chunks;
    let witness_len = meta.witness_len;
    if !witness_len.is_power_of_two() || num_cols == 0 {
        return Err(GkrLogupError::GkrLeafMismatch);
    }
    let n_vars = zinc_utils::log2(witness_len) as usize;
    let table_len = 1usize << width;
    // Shape checks on the prover-supplied payload.
    if meta.num_lookups != 1
        || meta.chunk_width != width
        || meta.parent_columns.len() != num_cols
        || proof.chunk_lifts.len() != 1
        || proof.chunk_lifts[0].len() != num_cols
        || proof.aggregated_multiplicities.len() != 1
        || proof.aggregated_multiplicities[0].len() != table_len
        || proof.witness_gkr.roots_p.len() != 1
        || proof.witness_gkr.roots_q.len() != 1
    {
        return Err(GkrLogupError::GkrLeafMismatch);
    }
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let table: Vec<F> = generate_word_table::<F>(width, field_cfg);
    let mults = &proof.aggregated_multiplicities[0];

    // ---- Step 1: absorb multiplicities, sample β (and α) ----
    let mut buf = vec![0u8; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(mults, &mut buf);
    let beta: F = transcript.get_field_challenge(field_cfg);
    let _alpha: F = transcript.get_field_challenge(field_cfg);

    // ---- Step 2: witness + table GKR verify ----
    let leaves = num_cols * witness_len;
    let w_num_vars = zinc_utils::log2(leaves.next_power_of_two()) as usize;
    let w_size = 1usize << w_num_vars;
    let t_num_vars = zinc_utils::log2(table_len.next_power_of_two()) as usize;
    let witness_result =
        batched_gkr_fraction_verify(transcript, &proof.witness_gkr, w_num_vars, field_cfg)?;
    let table_result = gkr_fraction_verify(transcript, &proof.table_gkr, t_num_vars, field_cfg)?;

    // ---- Step 3: roots cross-check: p_w / q_w == p_t / q_t ----
    let lhs = proof.witness_gkr.roots_p[0].clone() * &proof.table_gkr.root_q;
    let rhs = proof.table_gkr.root_p.clone() * &proof.witness_gkr.roots_q[0];
    if lhs != rhs {
        return Err(GkrLogupError::GkrRootMismatch);
    }

    // ---- Step 4: multiplicity sum + table-side leaf check ----
    let m_sum: F = mults.iter().cloned().fold(zero.clone(), |a, b| a + &b);
    if m_sum != F::from_with_cfg(leaves as u64, field_cfg) {
        return Err(GkrLogupError::MultiplicitySumMismatch {
            expected: leaves as u64,
            got: 0,
        });
    }
    if table_result.point.is_empty() {
        let expected_q = beta.clone() - &table[0];
        if mults[0] != table_result.expected_p || expected_q != table_result.expected_q {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    } else {
        let eq_at_t = build_eq_x_r_vec(&table_result.point, field_cfg)?;
        let mut p_eval = zero.clone();
        let mut q_eval = zero.clone();
        for j in 0..table_len {
            p_eval = p_eval + &(mults[j].clone() * &eq_at_t[j]);
            q_eval = q_eval + &((beta.clone() - &table[j]) * &eq_at_t[j]);
        }
        for e in eq_at_t.iter().skip(table_len) {
            q_eval = q_eval + e;
        }
        if p_eval != table_result.expected_p || q_eval != table_result.expected_q {
            return Err(GkrLogupError::GkrLeafMismatch);
        }
    }

    // ---- Step 5: witness-side leaf check from the per-column evals ----
    let r_full = &witness_result.point;
    if r_full.len() != w_num_vars {
        return Err(GkrLogupError::GkrLeafMismatch);
    }
    let r_inner: Vec<F> = r_full[..n_vars].to_vec();
    let r_outer: Vec<F> = r_full[n_vars..].to_vec();
    let mut padding = zero.clone();
    if leaves != w_size {
        let eq_at_full = build_eq_x_r_vec(r_full, field_cfg)?;
        for e in eq_at_full.iter().skip(leaves) {
            padding = padding + e;
        }
    }
    // p̃(r) = Σ_{j<LW} eq(j, r) = 1 − pad.
    if witness_result.expected_ps[0] != one.clone() - &padding {
        return Err(GkrLogupError::GkrLeafMismatch);
    }
    // q̃(r) = β·(1 − pad) − Σ_k eq(k, r_outer)·MLE[v_k](r_inner) + pad.
    let eq_at_outer = eq_table_or_unit(&r_outer, field_cfg)?;
    let mut combined = zero.clone();
    for k in 0..num_cols {
        let e_k = lift_scalar(&proof.chunk_lifts[0][k], &zero);
        combined = combined + &(eq_at_outer[k].clone() * &e_k);
    }
    let expected_q = beta.clone() * &(one.clone() - &padding) - &combined + &padding;
    if expected_q != witness_result.expected_qs[0] {
        return Err(GkrLogupError::GkrLeafMismatch);
    }

    Ok(GkrLogupGroupSubclaim {
        r_inner,
        combined_polynomial: proof.chunk_lifts[0].clone(),
        parent_columns: meta.parent_columns.clone(),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the polynomial-valued MLE evaluation of a `binary_poly<D>`
/// column at `point ∈ F_q^{n_vars}`. Returns a `DynamicPolynomialF<F>`
/// of degree `< D` whose coefficient `p` equals
/// `Σ_i eq(i, point) · bit_p(parent[i])`.
///
/// Convenience wrapper around [`compute_binary_poly_lifts`] for the
/// single-column case. Callers with multiple columns at the same
/// `point` should call [`compute_binary_poly_lifts`] directly to share
/// the eq-table build and parallelize across columns.
pub fn compute_binary_poly_lift<F, const D: usize>(
    parent: &DenseMultilinearExtension<BinaryPoly<D>>,
    point: &[F],
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Send + Sync,
    F::Config: Sync,
{
    compute_binary_poly_lifts::<F, D>(&[parent], point, field_cfg)
        .into_iter()
        .next()
        .expect("single-col lift")
}

/// Compute polynomial-valued MLE evaluations of a batch of
/// `binary_poly<D>` columns at a shared `point ∈ F_q^{n_vars}`.
///
/// Builds the `eq(·, point)` table ONCE (an O(2^n_vars) operation),
/// then parallelizes the per-column bit-conditional sum across rayon
/// threads. Each column's inner accumulation uses `+=` on `coeffs[p]`
/// (no per-add allocation).
///
/// Drop-in replacement for calling [`compute_binary_poly_lift`] N
/// times in a serial loop — saves N-1 redundant eq builds plus N-fold
/// parallelism on the `O(N · 2^n_vars · D)` bit walk.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_binary_poly_lifts<F, const D: usize>(
    cols: &[&DenseMultilinearExtension<BinaryPoly<D>>],
    point: &[F],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Send + Sync,
    F::Config: Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let eq_table = build_eq_x_r_vec(point, field_cfg)
        .expect("compute_binary_poly_lifts: eq table build failed");
    cfg_iter!(cols)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (i, entry) in col.iter().enumerate() {
                for (p, c) in entry.iter().enumerate() {
                    if c.into_inner() {
                        coeffs[p] += &eq_table[i];
                    }
                }
            }
            DynamicPolynomialF::new_trimmed(coeffs)
        })
        .collect()
}

/// Combine K chunk lifts into the parent's combined polynomial:
///   `combined = Σ_k X^{k · chunk_width} · chunks[k]`
/// where the result has `width` coefficients (`width = K · chunk_width`).
pub fn combine_chunks<F: PrimeField>(
    chunks: &[DynamicPolynomialF<F>],
    chunk_width: usize,
    width: usize,
    zero: &F,
) -> DynamicPolynomialF<F> {
    let mut coeffs = vec![zero.clone(); width];
    for (k, chunk) in chunks.iter().enumerate() {
        let lo = k * chunk_width;
        for (p, c) in chunk.coeffs.iter().enumerate() {
            if lo + p < width {
                coeffs[lo + p] = c.clone();
            }
        }
    }
    DynamicPolynomialF::new_trimmed(coeffs)
}

/// `eq(index, r)` read off the index's bits rather than out of a table:
/// a verifier wanting a handful of these pays for those and not for the
/// whole cube. Agrees with `build_eq_x_r_vec`, whose entry `j` pairs
/// `r[nu]` with bit `nu` of `j`.
#[allow(clippy::arithmetic_side_effects)]
/// Evaluate a `DynamicPolynomialF<F>` at the projecting element `a`
/// (`ψ_a` on a polynomial of degree < some bound). Horner from the
/// highest coefficient down.
#[allow(clippy::arithmetic_side_effects)]
fn eval_at_projecting_element<F: PrimeField>(
    poly: &DynamicPolynomialF<F>,
    a: &F,
    field_cfg: &F::Config,
) -> F {
    let mut acc = F::zero_with_cfg(field_cfg);
    for c in poly.coeffs.iter().rev() {
        acc = acc * a + c;
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{Field, crypto_bigint_const_monty::ConstMontyField};
    use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;
    type Inner = <F as Field>::Inner;

    fn rand_binary_poly_col(
        n_vars: usize,
        rng: &mut impl RngCore,
    ) -> DenseMultilinearExtension<BinaryPoly<32>> {
        let len = 1usize << n_vars;
        let evals: Vec<BinaryPoly<32>> =
            (0..len).map(|_| BinaryPoly::<32>::from(rng.next_u32())).collect();
        DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, BinaryPoly::<32>::zero())
    }

    /// A column holding a permutation of 1..=9 and nothing else but pad
    /// is the multiset the table prescribes, so it proves and verifies,
    /// and the claim the verifier is left holding is the column's own
    /// multilinear evaluation.
    #[test]
    fn round_trip_prescribed_permutation() {
        let cfg = ();
        let n_vars = 4; // W = 16: nine values and seven pads.
        let cells: Vec<i64> = vec![4, 9, 2, 3, 5, 7, 8, 1, 6, 0, 0, 0, 0, 0, 0, 0];
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells.clone(), 0i64);

        let a: F = F::from(7u64);
        let instance = IntLookupInstance::<'_, i64> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, prover_sub) =
            prove_group_prescribed::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        assert!(
            proof.aggregated_multiplicities.is_empty(),
            "a prescribed table's multiplicities are the verifier's own"
        );

        let mut v_ts = Blake3Transcript::new();
        let verifier_sub =
            verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");

        assert_eq!(prover_sub.r_inner, verifier_sub.r_inner);
        assert_eq!(prover_sub.combined_polynomial, verifier_sub.combined_polynomial);

        let eq = build_eq_x_r_vec(&verifier_sub.r_inner, &cfg).expect("eq");
        let mut expected = F::from(0u64);
        for (i, c) in cells.iter().enumerate() {
            expected = expected + &(eq[i].clone() * &F::from(*c as u64));
        }
        assert_eq!(
            verifier_sub.combined_polynomial[0],
            DynamicPolynomialF::new_trimmed(vec![expected])
        );
    }

    /// A 3x3 latin square laid down one row per column, over `rows` rows
    /// so every column has cells no selection names.
    fn latin_square<const C: usize>(
        square: [[u32; 3]; C],
        rows: usize,
    ) -> Vec<DenseMultilinearExtension<Inner>> {
        let n_vars = zinc_utils::log2(rows) as usize;
        let zero = F::from(0u32).inner().clone();
        square
            .iter()
            .map(|row| {
                let mut evals: Vec<Inner> =
                    row.iter().map(|v| F::from(*v).inner().clone()).collect();
                evals.resize(rows, zero.clone());
                DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, zero.clone())
            })
            .collect()
    }

    /// Three row selections and three column selections over the three
    /// columns: the shape a sudoku's obligations take, at the size a unit
    /// test can read.
    fn latin_selections() -> Vec<Vec<(u32, u32)>> {
        let rows = (0..3u32).map(|r| (0..3u32).map(|p| (r, p)).collect());
        let columns = (0..3u32).map(|p| (0..3u32).map(|r| (r, p)).collect());
        rows.chain(columns).collect()
    }

    fn latin_table(selections: Vec<Vec<(u32, u32)>>) -> LookupTableType {
        LookupTableType::Selected { values: (1..=3).collect(), selections }
    }

    fn prove_latin(
        columns: &[DenseMultilinearExtension<Inner>],
        table_type: LookupTableType,
    ) -> Result<
        (GkrLogupGroupProof<F>, GkrLogupGroupMeta, GkrLogupGroupSubclaim<F>),
        GkrLogupError<F>,
    > {
        let instance = SelectedLookupInstance::<'_, F> {
            parent_columns: columns.iter().collect(),
            parent_column_indices: (0..columns.len()).collect(),
            table_type,
            n_vars: zinc_utils::log2(columns[0].evaluations.len()) as usize,
        };
        prove_group_selected::<F>(&mut Blake3Transcript::new(), &instance, &())
    }

    /// A latin square's selections bound to an all-ones grid through second coefficients.
    #[test]
    fn a_selected_lift_past_one_coefficient_is_refused() {
        let rows = 8;
        let fake = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], rows);
        let truth = latin_square([[1, 1, 1], [1, 1, 1], [1, 1, 1]], rows);
        let (mut proof, meta, sub) =
            prove_latin(&fake, latin_table(latin_selections())).expect("prove");
        let a: F = F::from(7u64);
        let eq = build_eq_x_r_vec(&sub.r_inner, &()).expect("eq");
        let eval = |col: &DenseMultilinearExtension<Inner>| {
            col.evaluations.iter().zip(&eq).fold(F::from(0u64), |acc, (c, e)| {
                acc + &(e.clone() * &F::new_unchecked_with_cfg(c.clone(), &()))
            })
        };
        for (lift, column) in proof.chunk_lifts[0].iter_mut().zip(&truth) {
            let held = eval(column);
            let shown = (lift.coeffs[0].clone() - held.clone()) / &a;
            *lift = DynamicPolynomialF::new_trimmed(vec![held, shown]);
        }
        let res = verify_group::<F>(&mut Blake3Transcript::new(), &proof, &meta, &a, &());
        assert!(matches!(res, Err(GkrLogupError::GkrLeafMismatch)), "{res:?}");
    }

    /// Six selections over three columns -- neither count a power of two --
    /// prove and verify as one group, and the claim the verifier is left
    /// holding is one lift per column rather than one per selection.
    #[test]
    fn round_trip_selected_latin_square() {
        let a: F = F::from(7u64);
        let columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        let (proof, meta, prover_sub) =
            prove_latin(&columns, latin_table(latin_selections())).expect("prove");
        assert_eq!(meta.num_lookups, 6);
        assert_eq!(proof.chunk_lifts.len(), 1, "every selection reads the same columns");
        assert!(proof.aggregated_multiplicities.is_empty());

        let mut v_ts = Blake3Transcript::new();
        let sub = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &()).expect("verify");
        assert_eq!(prover_sub.combined_polynomial, sub.combined_polynomial);

        let eq = build_eq_x_r_vec(&sub.r_inner, &()).expect("eq");
        for (slot, column) in columns.iter().enumerate() {
            let expected = column.evaluations.iter().zip(eq.iter()).fold(
                F::from(0u64),
                |acc, (cell, eq_i)| {
                    acc + &(eq_i.clone() * &F::new_unchecked_with_cfg(cell.clone(), &()))
                },
            );
            assert_eq!(
                sub.combined_polynomial[slot],
                DynamicPolynomialF::new_trimmed(vec![expected])
            );
        }
    }

    /// Each row of the square beside a sorted copy of itself: three pairs
    /// over six columns, no value prescribed anywhere.
    fn sorted_pairs() -> LookupTableType {
        let pairs = (0..3u32)
            .map(|r| {
                (
                    (0..3u32).map(|p| (r, p)).collect(),
                    (0..3u32).map(|p| (3 + r, p)).collect(),
                )
            })
            .collect();
        LookupTableType::Permuted { pairs }
    }

    fn square_with_sorted_rows() -> Vec<DenseMultilinearExtension<Inner>> {
        latin_square(
            [[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            8,
        )
    }

    /// A permuted group proves each pair holds one multiset and verifies
    /// with the same single row of column claims a selected group leaves.
    #[test]
    fn round_trip_permuted_rows() {
        let a: F = F::from(7u64);
        let columns = square_with_sorted_rows();
        let (proof, meta, prover_sub) = prove_latin(&columns, sorted_pairs()).expect("prove");
        assert_eq!(meta.num_lookups, 6);
        assert_eq!(proof.chunk_lifts.len(), 1);
        assert!(proof.aggregated_multiplicities.is_empty());
        let mut v_ts = Blake3Transcript::new();
        let sub = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &()).expect("verify");
        assert_eq!(prover_sub.combined_polynomial, sub.combined_polynomial);
    }

    /// The values are the prover's own: shifting every cell of a pair by
    /// the same amount still proves, since nothing prescribes them.
    #[test]
    fn a_permuted_pair_prescribes_no_value() {
        let a: F = F::from(7u64);
        let columns = latin_square(
            [[41, 42, 43], [2, 3, 1], [3, 1, 2], [41, 42, 43], [1, 2, 3], [1, 2, 3]],
            8,
        );
        let (proof, meta, _) = prove_latin(&columns, sorted_pairs()).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        verify_group::<F>(&mut v_ts, &proof, &meta, &a, &()).expect("verify");
    }

    /// The teeth: one cell of a sorted copy changed, so the pair holds two
    /// multisets, and the crossed roots no longer cancel.
    #[test]
    fn a_pair_holding_two_multisets_is_refused() {
        let a: F = F::from(7u64);
        let mut columns = square_with_sorted_rows();
        columns[3].evaluations[1] = F::from(9u64).inner().clone();
        let (proof, meta, _) = prove_latin(&columns, sorted_pairs()).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &());
        assert!(
            matches!(res, Err(GkrLogupError::GkrRootMismatch)),
            "a pair of two multisets must fail the LogUp identity, got {res:?}"
        );
    }

    /// A cell no selection names is genuinely unconstrained: the lookup
    /// says nothing about it, so a column carrying anything at all in its
    /// unnamed rows still proves and verifies.
    #[test]
    fn an_unnamed_cell_is_unconstrained() {
        let a: F = F::from(7u64);
        let mut columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        columns[0].evaluations[5] = F::from(4242u64).inner().clone();
        let (proof, meta, _) =
            prove_latin(&columns, latin_table(latin_selections())).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        verify_group::<F>(&mut v_ts, &proof, &meta, &a, &()).expect("verify");
    }

    /// The teeth: a value slid out of a named cell into an unnamed one of
    /// the same column. Every cell of the column is still a value of the
    /// table, and the selection is simply one short -- there is no pad
    /// entry for the missing one to hide behind.
    #[test]
    fn a_value_slid_out_of_a_selection_is_refused() {
        let a: F = F::from(7u64);
        let mut columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        columns[0].evaluations[5] = columns[0].evaluations[2].clone();
        columns[0].evaluations[2] = F::from(0u64).inner().clone();
        let (proof, meta, _) =
            prove_latin(&columns, latin_table(latin_selections())).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &());
        assert!(
            matches!(res, Err(GkrLogupError::GkrRootMismatch)),
            "a selection one value short must fail the LogUp identity, got {res:?}"
        );
    }

    /// A selection naming a cell the group's columns do not have indexes
    /// nothing at all, so it is refused where the reason is legible.
    #[test]
    fn a_selection_outside_the_columns_is_refused() {
        let a: F = F::from(7u64);
        let columns = latin_square([[1, 2, 3], [2, 3, 1], [3, 1, 2]], 8);
        for bad in [(3u32, 0u32), (0, 8)] {
            let table = latin_table(vec![vec![(0, 0), (0, 1), bad]]);
            assert!(matches!(
                prove_latin(&columns, table),
                Err(GkrLogupError::MalformedSelection)
            ));
        }
    }

    /// The teeth: every cell is in the table and the column is still the
    /// wrong multiset -- a nine twice and no eight. The prover can build
    /// this proof, and the verifier, counting the table side itself, must
    /// reject it.
    #[test]
    fn a_repeated_value_is_not_the_prescribed_multiset() {
        let cfg = ();
        let n_vars = 4;
        let cells: Vec<i64> = vec![4, 9, 2, 3, 5, 7, 9, 1, 6, 0, 0, 0, 0, 0, 0, 0];
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells, 0i64);
        let a: F = F::from(7u64);
        let instance = IntLookupInstance::<'_, i64> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, _) =
            prove_group_prescribed::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg);
        assert!(
            matches!(res, Err(GkrLogupError::GkrRootMismatch)),
            "a repeated value must fail the LogUp identity itself, got {res:?}"
        );
    }

    /// A cell the table never names has no place in it, so the column is
    /// refused where the reason is legible rather than proved against a
    /// table it misses.
    #[test]
    fn a_value_outside_the_prescribed_table_is_refused() {
        let cfg = ();
        let n_vars = 4;
        for bad in [10i64, -1i64] {
            let mut cells: Vec<i64> = vec![4, 9, 2, 3, 5, 7, 8, 1, 6, 0, 0, 0, 0, 0, 0, 0];
            cells[0] = bad;
            let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, cells, 0i64);
            let instance = IntLookupInstance::<'_, i64> {
                parent_columns: vec![&parent],
                parent_column_indices: vec![0],
                table_type: LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
                n_vars,
            };
            let mut ts = Blake3Transcript::new();
            assert!(
                prove_group_prescribed::<F, i64>(&mut ts, &instance, &cfg).is_err(),
                "a cell of {bad} is not in the prescribed table and must not prove"
            );
        }
    }

    /// A pad that is one of the values would stand for two multiplicities
    /// at once, and values that outnumber the rows cannot all be laid
    /// down: neither table is a multiset a column could hold.
    #[test]
    fn a_table_that_is_not_a_multiset_is_refused() {
        let cfg = ();
        let n_vars = 2; // W = 4.
        let parent = DenseMultilinearExtension::from_evaluations_vec(n_vars, vec![1i64; 4], 0i64);
        for table_type in [
            LookupTableType::Prescribed { values: vec![1, 2, 3], pad: 2 },
            LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
        ] {
            let instance = IntLookupInstance::<'_, i64> {
                parent_columns: vec![&parent],
                parent_column_indices: vec![0],
                table_type,
                n_vars,
            };
            let mut ts = Blake3Transcript::new();
            assert!(prove_group_prescribed::<F, i64>(&mut ts, &instance, &cfg).is_err());
        }
    }

    #[test]
    fn round_trip_l1_k4_bitpoly32() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(42);
        let n_vars = 6; // W = 64
        let parent = rand_binary_poly_col(n_vars, &mut rng);

        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };

        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type: table_type.clone(),
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, prover_sub) =
            prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");

        let mut v_ts = Blake3Transcript::new();
        let verifier_sub =
            verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");

        assert_eq!(prover_sub.r_inner, verifier_sub.r_inner);
        assert_eq!(prover_sub.combined_polynomial, verifier_sub.combined_polynomial);

        // Sanity: combined_polynomial[0] should equal MLE[parent](r_inner).
        let parent_lift =
            compute_binary_poly_lift::<F, 32>(&parent, &verifier_sub.r_inner, &cfg);
        assert_eq!(verifier_sub.combined_polynomial[0], parent_lift);
    }

    #[test]
    fn round_trip_l2_k4_bitpoly32() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(7);
        let n_vars = 5;
        let p1 = rand_binary_poly_col(n_vars, &mut rng);
        let p2 = rand_binary_poly_col(n_vars, &mut rng);

        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&p1, &p2],
            parent_column_indices: vec![0, 1],
            table_type,
            projecting_element_f: &a,
            n_vars,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, _) = prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");
        let mut v_ts = Blake3Transcript::new();
        let sub = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg).expect("verify");
        assert_eq!(sub.combined_polynomial.len(), 2);
    }

    #[test]
    fn tampered_chunk_lift_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(99);
        let n_vars = 5;
        let parent = rand_binary_poly_col(n_vars, &mut rng);
        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type,
            projecting_element_f: &a,
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        let (mut proof, meta, _) = prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");

        // Tamper a chunk lift coefficient.
        proof.chunk_lifts[0][0].coeffs[0] =
            proof.chunk_lifts[0][0].coeffs[0].clone() + F::from(1u64);

        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg);
        assert!(res.is_err(), "verifier must reject tampered chunk lift");
    }

    #[test]
    fn tampered_multiplicity_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(123);
        let n_vars = 5;
        let parent = rand_binary_poly_col(n_vars, &mut rng);
        let a: F = F::from(rng.next_u64());
        let table_type = LookupTableType::BitPoly { width: 32, chunk_width: Some(8) };
        let instance = BinaryPolyLookupInstance::<'_, F, 32> {
            parent_columns: vec![&parent],
            parent_column_indices: vec![0],
            table_type,
            projecting_element_f: &a,
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        let (mut proof, meta, _) = prove_group::<F, 32>(&mut p_ts, &instance, &cfg).expect("prove");

        proof.aggregated_multiplicities[0][0] =
            proof.aggregated_multiplicities[0][0].clone() + F::from(1u64);

        let mut v_ts = Blake3Transcript::new();
        let res = verify_group::<F>(&mut v_ts, &proof, &meta, &a, &cfg);
        assert!(res.is_err(), "verifier must reject tampered multiplicity");
    }

    // ---- int-column (Word table) variant ----

    fn rand_int_col(n_vars: usize, width: usize, rng: &mut impl RngCore) -> DenseMultilinearExtension<i64> {
        let len = 1usize << n_vars;
        let mask = (1u64 << width) - 1;
        let evals: Vec<i64> = (0..len).map(|_| (rng.next_u64() & mask) as i64).collect();
        DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, 0i64)
    }

    fn direct_int_eval(col: &DenseMultilinearExtension<i64>, point: &[F]) -> F {
        let eq = build_eq_x_r_vec(point, &()).unwrap();
        let mut acc = F::from(0u64);
        for (i, v) in col.evaluations.iter().enumerate() {
            acc = acc + eq[i].clone() * F::from(*v as u64);
        }
        acc
    }

    fn int_round_trip(num_cols: usize, width: usize, n_vars: usize, seed: u64) {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(seed);
        let cols: Vec<_> = (0..num_cols).map(|_| rand_int_col(n_vars, width, &mut rng)).collect();
        let instance = IntLookupInstance::<'_, i64> {
            parent_columns: cols.iter().collect(),
            parent_column_indices: (0..num_cols).collect(),
            table_type: LookupTableType::Word { width, chunk_width: None },
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        let (proof, meta, prover_sub) =
            prove_group_int::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        assert_eq!(meta.num_chunks, num_cols);
        let mut v_ts = Blake3Transcript::new();
        let sub = verify_group_int::<F>(&mut v_ts, &proof, &meta, &cfg).expect("verify");
        assert_eq!(prover_sub.r_inner, sub.r_inner);
        assert_eq!(sub.combined_polynomial.len(), num_cols);
        let zero = F::from(0u64);
        for (k, col) in cols.iter().enumerate() {
            assert_eq!(
                lift_scalar(&sub.combined_polynomial[k], &zero),
                direct_int_eval(col, &sub.r_inner),
                "column {k} eval at r_inner"
            );
        }
        // serialization round trip of the group payload via the top-level proof
        let full = super::super::structs::GkrLogupLookupProof { groups: vec![proof], group_meta: vec![meta] };
        let mut bytes = vec![0u8; zinc_transcript::traits::Transcribable::get_num_bytes(&full)];
        zinc_transcript::traits::GenTranscribable::write_transcription_bytes_exact(&full, &mut bytes);
        let back = <super::super::structs::GkrLogupLookupProof<F> as zinc_transcript::traits::GenTranscribable>::read_transcription_bytes_exact(&bytes);
        assert_eq!(full, back);
    }

    #[test]
    fn int_round_trip_word16_l4() {
        int_round_trip(4, 16, 6, 11);
    }

    #[test]
    fn int_round_trip_word8_l3_padded() {
        // 3 columns: L·W is not a power of two → exercises the padding path.
        int_round_trip(3, 8, 5, 12);
    }

    #[test]
    fn int_round_trip_word8_l1() {
        int_round_trip(1, 8, 5, 13);
    }

    #[test]
    fn int_out_of_range_cell_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(14);
        let n_vars = 5;
        let mut col = rand_int_col(n_vars, 8, &mut rng);
        col.evaluations[7] = 256; // = 2^8, outside Word(8)
        let neg = {
            let mut c = rand_int_col(n_vars, 8, &mut rng);
            c.evaluations[3] = -1;
            c
        };
        for bad in [col, neg] {
            let instance = IntLookupInstance::<'_, i64> {
                parent_columns: vec![&bad],
                parent_column_indices: vec![0],
                table_type: LookupTableType::Word { width: 8, chunk_width: None },
                n_vars,
            };
            let mut p_ts = Blake3Transcript::new();
            let res = prove_group_int::<F, i64>(&mut p_ts, &instance, &cfg);
            assert!(
                matches!(res, Err(GkrLogupError::WitnessNotInTable)),
                "out-of-range cell must be rejected by the prover"
            );
        }
    }

    /// A chunked Word cell recombines by place value, a linear map with
    /// room in it: chunk lifts sent after the descent can meet both the leaf
    /// check and the recombination for a cell out of range. Both sides
    /// refuse the chunked table over int columns.
    #[test]
    fn int_chunked_word_refused() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(16);
        let n_vars = 5;
        let col = rand_int_col(n_vars, 16, &mut rng);
        let chunked = LookupTableType::Word { width: 16, chunk_width: Some(8) };
        let instance = IntLookupInstance::<'_, i64> {
            parent_columns: vec![&col],
            parent_column_indices: vec![0],
            table_type: chunked.clone(),
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        assert!(matches!(
            prove_group_int::<F, i64>(&mut p_ts, &instance, &cfg),
            Err(GkrLogupError::WitnessNotInTable)
        ));

        let whole = IntLookupInstance {
            table_type: LookupTableType::Word { width: 16, chunk_width: None },
            ..instance
        };
        let mut p_ts = Blake3Transcript::new();
        let (proof, mut meta, _) = prove_group_int::<F, i64>(&mut p_ts, &whole, &cfg).expect("prove");
        meta.table_type = chunked;
        let a = F::from(7u64);
        assert!(verify_group_int::<F>(&mut Blake3Transcript::new(), &proof, &meta, &cfg).is_err());
        assert!(verify_group::<F>(&mut Blake3Transcript::new(), &proof, &meta, &a, &cfg).is_err());
    }

    #[test]
    fn int_tampered_eval_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(15);
        let n_vars = 5;
        let cols: Vec<_> = (0..2).map(|_| rand_int_col(n_vars, 8, &mut rng)).collect();
        let instance = IntLookupInstance::<'_, i64> {
            parent_columns: cols.iter().collect(),
            parent_column_indices: vec![0, 1],
            table_type: LookupTableType::Word { width: 8, chunk_width: None },
            n_vars,
        };
        let mut p_ts = Blake3Transcript::new();
        let (mut proof, meta, _) = prove_group_int::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        let e = lift_scalar(&proof.chunk_lifts[0][1], &F::from(0u64));
        proof.chunk_lifts[0][1] = DynamicPolynomialF::new_trimmed(vec![e + F::from(1u64)]);
        let mut v_ts = Blake3Transcript::new();
        assert!(verify_group_int::<F>(&mut v_ts, &proof, &meta, &cfg).is_err());

        // and a tampered multiplicity
        let mut p_ts = Blake3Transcript::new();
        let (mut proof, meta, _) = prove_group_int::<F, i64>(&mut p_ts, &instance, &cfg).expect("prove");
        proof.aggregated_multiplicities[0][0] =
            proof.aggregated_multiplicities[0][0].clone() + F::from(1u64);
        let mut v_ts = Blake3Transcript::new();
        assert!(verify_group_int::<F>(&mut v_ts, &proof, &meta, &cfg).is_err());
    }

    #[test]
    fn int_table_index_encoding() {
        let mut buf = vec![0u8; 8];
        assert_eq!(int_table_index(&255i64, 8, &mut buf), Some(255));
        assert_eq!(int_table_index(&256i64, 8, &mut buf), None);
        assert_eq!(int_table_index(&-1i64, 8, &mut buf), None);
        assert_eq!(int_table_index(&0i64, 1, &mut buf), Some(0));
        let mut buf16 = vec![0u8; 16];
        assert_eq!(int_table_index(&65535i128, 16, &mut buf16), Some(65535));
        assert_eq!(int_table_index(&(1i128 << 70), 16, &mut buf16), None);
    }
}
