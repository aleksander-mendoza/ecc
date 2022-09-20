#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(step_trait)]

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
pub mod shape_arr;
pub mod dot_mad_arr;
pub mod dot_sparse_arr;
pub mod shape;
pub mod layout;
pub mod blas;
pub mod soft_wta;
pub mod init_rand;
pub mod conv_shape;
pub mod xyzw;
pub mod arr_concat;
pub mod vec_range;
pub mod norm;
pub mod from_usize;
pub mod tup_arr;

pub use vector_field::*;
pub use vector_field_arr::*;
pub use vector_field_vec::*;
pub use vector_field_slice::*;
pub use set::*;
pub use dot_arr::*;