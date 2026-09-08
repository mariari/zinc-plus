//! GKR-LogUp lookup with polynomial-valued chunk lifts.
//!
//! Modules:
//! * `gkr` — fraction tree primitives + layered GKR fractional sumcheck.
//! * `tables` — projected table generators + multiplicity helpers.
//! * `structs` — proof, error, and intermediate types.
//! * `protocol` — top-level prove/verify per lookup group with the
//!   chunks-in-clear polynomial-valued lift design (`BitPoly` tables over
//!   binary_poly columns), plus the `Word`-table variant over int columns
//!   (range checks on integer cells; `prove_group_int` / `verify_group_int`).

pub mod gkr;
pub mod protocol;
pub mod structs;
pub mod tables;

pub use protocol::{
    BinaryPolyLookupInstance, IntLookupInstance, combine_chunks, compute_binary_poly_lift,
    compute_binary_poly_lifts, compute_int_column_evals, int_table_index, lift_scalar,
    prove_group, prove_group_int, verify_group, verify_group_int,
};
pub use structs::{
    BatchedGkrFractionProof, BatchedGkrLayerProof, GkrFractionProof, GkrLayerProof,
    GkrLogupError, GkrLogupGroupMeta, GkrLogupGroupProof, GkrLogupGroupSubclaim,
    GkrLogupLookupProof, dump_size_breakdown,
};
