#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(step_trait)]
#![feature(iter_collect_into)]
#![feature(portable_simd)]
#![feature(slice_flatten)]
#![feature(const_ptr_read)]
#![feature(const_refs_to_cell)]

extern crate core;


mod vector_field;
mod vector_field_arr;
mod vector_field_slice;
mod set;
pub mod conv;
mod dot_arr;
pub mod top_k;
pub mod static_layout;
pub mod init;
pub mod dynamic_layout;
pub mod dot_slice;
mod vector_field_vec;
pub mod shaped_tensor_mad;
mod shape_arr;
pub mod dot_mad_arr;
pub mod dot_sparse_arr;
pub mod shape;
pub mod layout;
pub mod soft_wta;
pub mod init_rand;
pub mod conv_shape;
mod xyzw;
mod arr_concat;
pub mod vec_range;
mod norm;
pub mod from_usize;
mod tup_arr;
pub mod mat_slice;
pub mod mat_arr;
pub mod algebraic_inductive_inference;
pub mod bezier;
mod lin_trans;
pub mod collision;
pub mod lin_alg;
pub mod blas_safe;
pub mod mesh_primitives;
pub mod piecewise_linear;
mod mat_tri;

pub use statrs::*;
pub use levenshtein::*;
pub use mat_tri::*;
pub use levenshtein::*;
pub use arr_concat::*;
pub use shape_arr::*;
pub use tup_arr::*;
pub use norm::*;
pub use xyzw::*;
pub use lin_trans::*;
pub use vector_field::*;
pub use vector_field_arr::*;
pub use vector_field_vec::*;
pub use vector_field_slice::*;
pub use set::*;
pub use dot_arr::*;