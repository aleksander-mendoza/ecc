// use std::iter::Sum;
// use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};
// use std::process::Output;
// use num_traits::{AsPrimitive, Float, FromPrimitive, One, Pow, Zero};
// use crate::*;
//
// pub trait Mat<const DIMS:usize>{
//     type Item;
//     fn shape(&self)->&[usize;DIMS];
//     unsafe fn get_unchkd(&mut self,i:usize)->Self::Item;
//
// }
// pub struct MatRef<'a, T,const DIMS:usize>{
//     shape:[usize;DIMS],
//     mat:&'a [T],
// }
//
// impl <T,const DIMS:usize> Mat<DIMS> for MatRef<T,DIMS>{
//     type Item = T;
//
//     fn shape(&self) -> &[usize; DIMS] {
//         &self.shape
//     }
//
//     unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
//         self.mat.get_unchecked(i)
//     }
// }
//
// pub fn mat<T,const DIMS:usize>(shape:[usize;DIMS], mat:&[T])->MatRef<T,DIMS>{
//     MatRef{ shape, mat}
// }
// /**short alias for `mat`*/
// pub fn m<T,const DIMS:usize>(shape:[usize;DIMS], mat:&[T])->MatRef<T,DIMS>{
//     mat(shape,mat)
// }
// pub struct MatFold<M,F,I,const DIMS:usize>{
//     shape:[usize;DIMS],
//     mat:M,
//     f:F,
//     i:I
// }
//
// impl <T,M:Mat<{ DIMS + 1 }>,I:Fn(usize)->T,F:Fn(T,M::Item)->T,const DIMS:usize> Mat<DIMS> for MatFold<M,F,I, { DIMS + 1}>{
//     type Item = T;
//
//     fn shape(&self) -> &[usize; DIMS] {
//         &self.shape
//     }
//
//     unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
//         let mut t = self.i(i);
//
//         self.mat.get_unchecked(i)
//     }
// }
//
// pub fn fold_m<T,A:Mat<D>,const D:usize>(a:A,init:impl Fn(usize)->T){
//
// }
// //
//
