#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(step_trait)]
#![feature(iter_collect_into)]
#![feature(portable_simd)]
#![feature(slice_flatten)]
#![feature(const_ptr_read)]
#![feature(const_refs_to_cell)]
#![feature(new_uninit)]

extern crate core;


pub mod conv;
pub mod top_k;
pub mod soft_wta;
pub mod conv_shape;
pub mod vec_range;
mod from_usize;

pub use from_usize::*;

pub mod mat_slice;


pub mod bezier;
pub mod collision;
pub mod blas_safe;
pub mod mesh_primitives;
pub mod piecewise_linear;
pub mod line;

mod cayley;

pub use cayley::*;

mod mat_arr;

pub use mat_arr::*;
pub use statrs::*;

mod mat_tri;

pub use mat_tri::*;

mod shape_arr;

pub use shape_arr::*;

mod tup_arr;

pub use tup_arr::*;

mod mat;

pub use mat::*;

mod xyzw;

pub use xyzw::*;

mod lin_trans;

pub use lin_trans::*;

mod set;

pub use set::*;

mod quat;
pub mod histogram;
pub mod int_by_float;
mod pathfinding;
mod iter;

pub use iter::*;

mod num;
mod array;
mod array_ops;
pub mod rev_vec;
mod vec;

pub use vec::*;
pub use array_ops::*;

pub use array::*;

pub use num::*;

pub use pathfinding::*;

pub use quat::*;
