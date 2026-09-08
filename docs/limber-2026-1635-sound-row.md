# Limber's Zinc+ row, made sound: range checks, random prime, security target

**Date:** 2026-09-08 · **Branch:** `main-beta` (integration branch
`main-beta-limber-updates`, commits `693acee..`) · **Machine:** MacBook,
Apple M4 (10 cores, 16 GB), same box and methodology as the two earlier
notes (`docs/limber-2026-1635-reproduction.md`, 2026-08-21, and
`docs/limber-2026-1635-folded-main-beta.md`, 2026-08-26): single-threaded,
`simd unchecked`, `-C target-cpu=native`, medians, proof sizes raw + zstd-3.

This note records the four changes the Limber authors asked about in the
"Limber: Implementation Questions" thread (3–4 September 2026), two
soundness fixes found on the way, how to run everything, and the resulting
numbers.

## 1. What changed on `main-beta`

### 1.1 Integer range checks (the "rational witness" question)

> *It seems like the protocol only extracts rational witness values; would
> we need to add an additional range check to ensure integer witness
> values?* — Yes. Zip+'s extractor yields rational cells; the paper enforces
> integrality by lookup-constraining. That primitive now exists on
> `main-beta` for int columns:

- A UAIR declares `LookupColumnSpec { column_index, table_type:
  LookupTableType::Word { width, chunk_width: None } }` on witness int
  columns; the protocol proves every such cell is an integer in
  `[0, 2^width)` (`piop/src/lookup/gkr_logup/protocol.rs::prove_group_int`
  / `verify_group_int`, wired at protocol steps 4b and 7).
- Mechanism: the GKR-LogUp of the `main-beta-lookup` branch (merged), run
  directly on the projected int cells — one LogUp instance per group whose
  "chunks" are the columns, one multiplicity vector over the `2^width`
  table, and the per-column evaluations at the GKR point discharged by a
  new **int multipoint reducer** (`piop/src/int_multipoint_reducer.rs`)
  that folds them, together with the step-7 claim, into the single int Zip+
  opening. Soundness: with the extractor's height bound, `a·b⁻¹ ≡ t (mod q)`
  for a table entry `t` forces `v = a/b = t` exactly.
- The MultiSwap statement with range checks (`LIMB16=1`) represents each
  of `a, b, c, u` as 128 little-endian 16-bit limb columns (512 int columns,
  one modmul per row; `nvars = 13` ⇔ 8192 modmuls ⊇ 6,209) and recombines
  them by Horner inside the constraint `a·b − c − u·N = 0`. The fat-cell
  statements (`WIDE8` / `WIDE16` / `FOLD=1`) are kept for comparison; they
  have NO range check and a rational witness could satisfy them.

### 1.2 Transcript-drawn projecting prime (the "fixed field" question)

> *The implementation does not currently sample a random fingerprinting
> prime and instead fixes a field?* — It used to pin the 256-bit secp256k1
> base prime everywhere (a leftover of the SHA+ECDSA demo). Now
> `ZincTypes::FIXED_PROJECTING_PRIME` defaults to `None`: step 1 draws the
> prime from the Fiat–Shamir transcript after the witness commitments, with
> the width set by `Fmod` (`Uint<2>` → 128 bits, `Uint<3>` → 192 bits).
> Only the SHA+ECDSA bundles pin secp256k1 (their EC constraints are
> identities modulo that prime). The Limber bench uses the transcript-drawn
> prime: 128-bit at the 100-bit target, 192-bit at 114/128 bits.

Supporting changes: the value-sized field's modulus slot can be
re-installed between proofs; prime candidates get their top bit forced so
the prime has the full width; the primality test is Baillie–PSW instead of
a single base-2 Miller–Rabin round (a grinding prover could otherwise steer
the transcript onto a base-2 pseudoprime).

### 1.3 Witness generation time

> *Do the prover numbers contain witness generation time?* — They did not.
> Every result line now reports witness generation on its own AND added to
> the prover median (`prove+witgen`). For these statements it is ~0.05–0.2 s.

### 1.4 Security target

> *Zinc+ is configured at 100 bits of security …* — Cargo features
> `sec-114` / `sec-128` raise the target. The number of Zip+ column
> openings follows `zip_plus::pcs::structs::num_column_openings` (the
> proximity parameter is the 1.5 Johnson bound `θ = 1 − ρ^{1/3}`, so each
> opening buys `log2(1/ρ)/3` bits: 100 / 114 / 128 openings at rate 1/8),
> and the projecting prime widens to 192 bits above 100 bits (the
> fingerprinting term is `≈ 2^13 / prime` and the LogUp term
> `≈ 2^22 / prime` for this statement, so 128 bits would cap the
> range-checked row near 106 bits).

## 2. Two soundness fixes found on the way

1. **Zip+ batched scalar openings only bound the sum of the batch.**
   `ZipPlus::prove_f` / `verify_with_alphas` fold a batch by summing the
   per-polynomial rows after weighting each polynomial's entries by its
   alphas. Polynomial lanes get fresh random alphas per polynomial, but
   scalar (int) lanes used `[1]` for every polynomial, so an opening of ≥2
   int columns bound only `Σ_j eval_j`, and the per-column lifted evals the
   CPR / multipoint chain relies on were individually unbound. Every
   `main-beta` statement with ≥2 int columns was affected (the Limber mock,
   and the SHA+ECDSA demo with its 8 int columns). Fixed in zip-plus
   (`sample_alphas_for_poly`: one random alpha per batched scalar
   polynomial), regression test
   `batched_verify_rejects_evaluation_mass_transfer`. Cost: none
   measurable (one extra 128-bit weight per column in the combined row).
2. **Lookup transcript gaps** (from the merged lookup branch): the
   non-parent bin lifts at `r_inner` and the lifts at the reducer's `r*`
   were never absorbed into the transcript before the reducer / opening
   challenges were drawn; both are now absorbed (mirrored in the verifier).
   Also fixed: the padded-leaf leaf identity and the single-chunk case.

## 3. How to run

```bash
# range-checked 16-bit-limb statement (the sound row), 100-bit target:
LIMB16=1 NVARS=13 REPS=3 RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features "simd unchecked iprs-rate-1-8" --bench limber_multiswap

# same at 128 bits (128 openings, 192-bit prime):
LIMB16=1 NVARS=13 REPS=3 RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features "simd unchecked iprs-rate-1-8 sec-128" --bench limber_multiswap

# fat-cell mock rows (no range check), as before:
FOLD=1 WIDE16=1 NVARS=11 REPS=5 RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features "simd unchecked iprs-rate-1-8" --bench limber_multiswap
WIDE8=1 NVARS=12 REPS=5 RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features "simd unchecked iprs-rate-1-8" --bench limber_multiswap
```

Add `parallel` to the feature list for the multi-threaded numbers. The
banner line prints the effective security target, opening count and prime
width.

## 4. Results

All rows: the same 8192-modmul statement (⊇ 6,209 constraints), rate 1/8,
single-threaded unless marked MT, medians (3–5 reps), `unchecked`, LTO,
`-C target-cpu=native`, Apple M4. "prove" excludes witness generation;
"prove+wg" includes it.

| Row | Shape | Range check | Prime | Openings | witness gen | prove | prove+wg | verify | proof raw | proof zstd |
|---|---|---|---|---|---|---|---|---|---|---|
| **limb16 (sound)** | 8192 rows × 512 int cols (16-bit limbs) | yes (`Word{16}` on all 512 cols) | FS 128-bit | 100 | 0.140 s | **5.306 s** | 5.446 s | **26.3 ms** | 2.33 MB | **881 KiB** |
| folded 4× wide16 | 2048 × 16 (Int<34>), 4n rows | no | FS 128-bit | 100 | 0.083 s | 0.895 s | 0.978 s | 73.8 ms | 1.38 MB | 808 KiB |
| unfolded wide8 | 4096 × 8 (Int<34>) | no | FS 128-bit | 100 | 0.088 s | 1.010 s | 1.099 s | 127 ms | 1.70 MB | 1.35 MB |
| unfolded wide16 | 2048 × 16 (Int<34>) | no | FS 128-bit | 100 | 0.083 s | 1.395 s | 1.478 s | 102 ms | 1.23 MB | 991 KiB |
| limb16 (sound), `sec-128` | as above | yes | FS 192-bit | 128 | 0.117 s | 7.353 s | 7.470 s | 26.8 ms | 3.11 MB | 1.08 MB |
| folded 4× wide16, `sec-128` | as above | no | FS 192-bit | 128 | 0.084 s | 0.934 s | 1.018 s | 62.6 ms | 1.45 MB | 850 KiB |
| limb16 (sound), MT (`parallel`, 10 cores) | as above | yes | FS 128-bit | 100 | 0.133 s | 2.205 s | 2.338 s | 18.2 ms | 2.33 MB | 884 KiB |
| folded 4× wide16, MT | as above | no | FS 128-bit | 100 | 0.083 s | 0.209 s | 0.292 s | 29.4 ms | 1.38 MB | 807 KiB |

For reference, the rows quoted on 2026-08-26 with the pinned 256-bit prime
and without the batch-binding fix: folded 4× wide16 0.852 s / 57.8 ms /
1.42 MB / 686 KiB. Prove and raw size are unchanged within noise; the zstd
size grew by ~120 KiB because the combined row now carries a random
128-bit weight per column (entropy that compressed away before) — that is
the cost of the fix in §2.1.

Limber's own headline rows (their machine, single-threaded): Limber-Hyrax
1.32 s / 39 ms / 170 KB; Limber-Brakedown 1.18 s / 45 ms / 5.3 MB raw.

### Per-step prover breakdown of the sound row (one run, `STEPS=1`)

| step | time | share |
|---|---|---|
| 0 Zip+ commit (512 int cols, `Cw = Int<2>`) | 2.382 s | 45.0 % |
| 1 prime projection | 0.060 s | 1.1 % |
| 2 ideal check | 0.189 s | 3.6 % |
| 3 eval projection | 0.027 s | 0.5 % |
| 4 CPR sumcheck | 0.423 s | 8.0 % |
| 4b lookup (GKR-LogUp over 4.2M cells) | 0.933 s | 17.6 % |
| 5 multipoint eval | 0.070 s | 1.3 % |
| 6 lift-and-project | 0.237 s | 4.5 % |
| 7 PCS open (int reducer + Zip+ open) | 0.977 s | 18.4 % |
| total | 5.298 s | |

(With the initial `Cw = Int<3>` codewords the same row took 6.886 s, of
which the commit was 4.46 s: narrowing the codeword from 192 to 128 bits
removed a third of the bytes hashed and 0.4 MB of raw proof.)

## 5. Reading the numbers

- **The sound row costs ~6× the unsound fat-cell rows on the prover, and
  the difference is mostly the commitment.** A 16-bit limb lives in a
  64-bit int cell and is encoded into a 128-bit IPRS codeword entry
  (`Cw = Int<2>`), so the 8 MB of witness bits become a ~540 MB codeword
  to hash, whereas a 2048-bit fat cell amortises the encoder's ~100-bit
  growth over 2048 bits (codeword ≈ 1.1× the data). The lookup itself
  (step 4b) is 0.9 s and the int reducer + opening 1.0 s; the ideal check
  and CPR of the 800-operation Horner constraint are cheap (0.6 s
  together).
- **The verifier is the best of any row** (26 ms): it re-encodes one
  8192-entry `Int<6>` combined row, checks 100 columns of 512 narrow
  entries, and the GKR verification is logarithmic; the 2^16-entry table
  side costs a batch inversion.
- **Proof size (2.33 MB raw / 881 KiB zstd)** is dominated by the
  multiplicity vector (2^16 field elements ≈ 1 MB raw, but it compresses
  ~10× since every entry is ≈ 64 ± noise), the 100 opened columns
  (100 × 512 × 16 B ≈ 0.8 MB) and the combined row (8192 × 48 B).
- **Against Limber** (single-threaded, their machine): the sound Zinc+ row
  proves ~4–4.5× slower than Limber-Brakedown / Limber-Hyrax, verifies
  ~1.5–1.7× faster than either, and its proof is ~2.3× smaller raw (6×
  zstd) than Limber-Brakedown but ~5× larger than Limber-Hyrax. The
  fat-cell rows that were quoted before should no longer be compared with
  Limber's range-checked numbers.
- **Cheap levers not taken here:** a compact (u32) encoding of the
  multiplicity vector (−0.8 MB raw, no zstd change); a narrower-cell IPRS
  lane (u32 cells) so the storage bloat per limb drops from 4× to 2× and
  the projected trace halves; parallelising the int reducer's `P(x)`
  build. The structural lever is committing the bits over F₂ and
  evaluating over the integers (the `f2-int` / F2Z line), which is exactly
  built for this kind of statement.

- **Security target.** Going from 100 to 128 bits (128 openings, 192-bit
  prime) costs the sound row +39 % prover time and +34 % raw proof
  (+28 opened columns, 3-limb field arithmetic) and the fat-cell row
  +4 % / +5 %; verification is unchanged. With the 128-bit prime the sound row's LogUp term
  (`≈ 2^22 / 2^128`) would already cap soundness near 106 bits, which is
  why the bench widens the prime above 100 bits.
- **Multi-threading** (10 cores) brings the sound row to 2.2 s prove /
  18 ms verify and the fat-cell folded row to 0.21 s / 29 ms; Limber's
  methodology is single-threaded, so the single-threaded rows are the
  ones to compare.

## 6. Artifacts

Branch `main-beta-limber-updates` (merged into `main-beta`), worktree
`.claude/worktrees/limber-updates`. Tests: `test_e2e_int_lookup16*`,
`test_e2e_mixed_bin_int_lookup`, `test_e2e_random_projecting_prime_128`,
`batched_verify_rejects_evaluation_mass_transfer`, plus the piop unit tests
of `gkr_logup::protocol` and `int_multipoint_reducer`.

## 7. Optimization log (2026-09-08, branch `main-beta-limber-updates`)

Goal: bring the sound `LIMB16=1 NVARS=13` row as close to Limber's
prover time as possible without touching soundness, the security
parameters, verifier time or proof size. Same machine and methodology as
§4 (Apple M4, single-threaded unless marked MT, `simd unchecked
iprs-rate-1-8`, LTO, `-C target-cpu=native`, medians of 3). The baseline
re-measured at the start of this pass: **prove 5.074 s / verify 21.1 ms /
2.33 MB raw / 876 KiB zstd** (commit 2.12 s, lookup 0.93 s, open 0.93 s,
CPR 0.41 s). A `sample` profile of the prover attributed the time as:
IPRS encode 2.1 s (crypto-bigint `Int<2>` limb loops), Merkle 0.4 s,
the Zip+ open's `b` inner product 0.6 s (a 384-bit → field `Uint::rem`
per cell), the GKR layer sumchecks 0.8 s, the CPR constraint closure
0.3 s.

### Shipped

- **Native `i64` cells / `i128` codeword entries for the limb lane**
  (bench-only: `RsaLimbZincTypes` now uses `Int = i64`, `CwR = i128`
  instead of `Int<1>` / `Int<2>`; same 64/128-bit widths, so the
  CHECKED-validated growth bound is unchanged, re-validated with a
  CHECKED full-size run). The IPRS base layer and butterflies run on
  machine integers (`mul`/`umulh`/`madd`) instead of crypto-bigint's
  generic limb loops. Encode 2.13 → 0.74 s, commit 2.12 → 1.08 s,
  **prove 5.074 → 4.102 s**; verify 21.0 ms and proof bytes identical.
  The `STEPS=1` mode now also prints the int lane's encode / Merkle
  split.

- **Zip+ opening without the per-cell wide-integer → field reduction**
  (`zip-plus`, `protocol`, `piop`). The profile's largest leaf was
  `crypto_bigint::Uint::rem` under `MBSInnerProduct::inner_product_field`
  in `ZipPlus::prove_f`: to send the row sum `b` the prover converted every
  alpha-weighted cell (`Int<6>`, 4.2M of them) into `F` with a 384-bit
  division, only to compute `Σ_j alpha_j · MLE[col_j](r*)` — a value it
  already holds (the `r*` evals it just absorbed). New
  `ZipPlus::prove_f_with_evals` (single-row scalar lanes only) computes
  `b = Σ_j F(alpha_j) · evals[j]` directly; the protocol's int-lane opens
  use it whenever `num_rows == 1`. In the same pass `prove_f` now
  accumulates the combined row one polynomial at a time (sequential adds
  into one `row_len` accumulator instead of materializing the whole batch's
  weighted rows and gathering them with a `row_len` stride), and the int
  reducer builds `P(x)` / `M(x)` column-by-column over `x` chunks instead of
  gathering 512 columns per `x`. Step 7: 0.982 → 0.092 s;
  **prove 4.102 → 2.814 s**; verify 21.1 ms; proof bytes identical (the
  transcript and every written value are unchanged — this is purely how
  the honest prover computes them). Tests: protocol / piop / zip-plus
  suites green.

- **Eq-factored GKR layer sumchecks** (`piop/src/lookup/gkr_logup/gkr.rs`,
  new `layer_sumcheck_prove`, used by both the batched and the single-tree
  GKR provers). Each of the 22 witness-tree layers ran the generic
  closure-based sumcheck driver on `[eq, pla, ql, pr, qr]` at degree 3:
  four `comb_fn` calls per hypercube pair (each `eq · (pla·qr + pr·ql) · δ⁰`),
  five arrays folded per round, and a 2^k clone of every layer into MLEs.
  The specialised prover keeps `eq` out of the fold
  (`eq(x, r) = E_{i-1} · eq(X, r_i) · eq(x', r_{>i})`), evaluates only
  the degree-2 inner sum at `X ∈ {0, 1, 2}` (three `h` evaluations per
  pair, `H(3)` by extrapolation), folds four arrays, and skips the `δ^ℓ`
  multiply (weighting per-tree sums once). It emits exactly the same
  round polynomials and transcript interaction, so the verifier is
  untouched. Step 4b: 0.821 → 0.456 s; **prove 2.814 → 2.631 s**; verify
  20.9 ms; proof bytes identical. Tests: piop + protocol suites green.

- **Step-6 int lifts on the projected column-major trace**
  (`protocol/src/prover.rs::step6_lift_and_project`). The lifted evals
  at `r_0` walked the row-major trace of heap-allocated `F_q[X]` cells
  column by column (two passes per column, one to find the degree) even
  though an int column's lift is a scalar; int columns now go through
  `compute_int_column_evals` on the already-projected column-major
  Montgomery MLEs (one contiguous pass per column), binary / arbitrary
  columns are unchanged. Step 6: 0.162 → 0.050 s (LTO off); proof
  bytes identical (the same degree-0 lifts are absorbed).

- **Blocked Merkle leaf gather** (`zip-plus/src/merkle.rs::hash_leaves`):
  leaves are gathered 64 columns at a time so each codeword row is read
  in contiguous runs instead of one 16-byte element per column with a
  1 MB stride, and one buffer serves 64 leaves instead of one allocation
  per leaf. Int-lane Merkle 0.39–0.41 → 0.34 s (LTO off). Modest: the
  remaining time is Blake3 over the 512 MB of codeword bytes
  (`blake3_hash_many_neon` ≈ 0.18 s) plus the per-leaf hasher overhead,
  so the next lever here is a narrower codeword serialization, not the
  gather.
