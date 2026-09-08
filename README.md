# Zinc+

WIP implementation of:
- A SNARK obtained via the Zinc+ framework (https://eprint.iacr.org/2026/109175).
- An arithmetization of a chain of 7xSHA-256 compressions followed by an ECDSA signature verification on the resulting hash.
See Sections 2.5 and 9 from https://eprint.iacr.org/2026/109175, and the documentation folder, for further details.

## Benchmarks

### Paper benchmarks:

To quickly reproduce the paper's benchmarks, use
```bash
RUSTFLAGS="-C target-cpu=native" \ 
  cargo bench --features "parallel simd unchecked iprs-rate-1-8" \ 
  --bench e2e -- "Folded 4x"
```

### Available benchmarks:

| Benchmark          | What it measures                                                                                                                       |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `zip_benches`      | PCS-level operations (encode, Merkle tree, commit, prove, verify) using **scalar** evaluations with IPRS codes. Uses `i32` evaluations |
| `zip_plus_benches` | Same PCS-level operations using **polynomial** evaluations (degree 32 & 64) with both RAA and IPRS codes. Uses `{0,1}^D` evaluations.  |
| `e2e`              | Full Zinc+ SNARK prove & verify on several test AIRs (NoMult, BinaryDecomposition, BigLinear, BigLinearPI) at varying sizes.           |
| `limber_multiswap` | The MultiSwap statement of Limber (ePrint 2026/1635) — `a·b = c + u·N` over a 2048-bit `N` — as fat 2048-bit int cells (`WIDE8`/`WIDE16`, `FOLD=1`) or as range-checked 16-bit limb columns (`LIMB16=1`, the sound row). See `docs/limber-2026-1635-*.md`. |

To run benchmarks, use
```bash
RUSTFLAGS="-C target-cpu=native" cargo bench \
  --features "simd parallel unchecked" \
  --bench BENCH_NAME
```

### Flags & features

| Flag / Feature         | What it does                                                                                                                                                     |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-C target-cpu=native` | Lets the compiler emit platform-specific instructions (NEON, AVX-512, etc.). Required for `simd`.                                                                |
| `simd`                 | Bit-packs binary polynomials into `u64`s and uses hand-written NEON / AVX-512 intrinsics for key operations (widening, inner products).                          |
| `parallel`             | Enables [rayon](https://docs.rs/rayon)-based multi-threaded execution across the whole stack (sumcheck, encoding, commitment, etc.).                             |
| `unchecked`            | Replaces `checked_add` / `checked_mul` with plain arithmetic, removing overflow guards. Only affects integer-typed computations; field arithmetic is unaffected. |
|iprs-rate-1-8| Uses IPRS codes of rate 1/8 (and, at the default 100-bit target, 100 spot proximity checks). Without this feature, the rate is 1/4 (150 checks). |
|`sec-114`, `sec-128`| Raise the security target of the tests/benches from 100 to 114 or 128 bits: the number of spot proximity checks follows `zip_plus::pcs::structs::num_column_openings` (100 / 114 / 128 at rate 1/8, 150 / 171 / 192 at rate 1/4), and benches drawing the projecting prime from the transcript widen it from 128 to 192 bits. |

### Projecting prime

Step 1 of the protocol reduces the integer trace modulo a prime `q`. By default `q` is drawn from the Fiat–Shamir transcript after the witness commitments (`ZincTypes::FIXED_PROJECTING_PRIME = None`; the width follows `Fmod`, e.g. `Uint<2>` for a 128-bit prime). The SHA+ECDSA arithmetization pins the secp256k1 base prime instead (`Some(&fixed_prime::SECP256K1_P_LE_BYTES)`) because its EC constraints are identities modulo that specific prime — see `protocol/src/fixed_prime.rs` for the soundness caveat of pinning.

### Integer range checks

Zip+'s extractor yields rational witness cells; integrality is enforced by lookup-constraining. A UAIR declares `LookupColumnSpec { column_index, table_type: LookupTableType::Word { width, chunk_width: None } }` on witness int columns and the protocol proves each such cell is an integer in `[0, 2^width)` (GKR-LogUp over the projected field, `piop/src/lookup/gkr_logup`). `BitPoly { width, chunk_width }` does the analogous check on binary-poly columns.

## License

Apache 2.0

## Would like to contribute?

see [Contributing](./CONTRIBUTING.md).

