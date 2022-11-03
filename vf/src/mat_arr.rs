use std::mem::MaybeUninit;
use std::ops::{Add, Mul, MulAssign};
use std::ptr;
use num_traits::{One, Zero};
use crate::*;
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a column*/
// pub fn fold_columns<A: Clone, D, const W: usize, const H: usize>(matrix: &[[D; W]; H], init: &mut [A; W], fold: impl Fn(A, &D) -> A) {
//     mat_slice::fold_columns(W, matrix.flatten(), init.as_mut_slice(), fold)
// }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a row*/
// pub fn fold_rows<A: Clone, D, const W: usize, const H: usize>(matrix: &[[D; W]; H], init: &mut [A; H], fold: impl Fn(A, &D) -> A) {
//     mat_slice::fold_rows(W, matrix.flatten(), init.as_mut_slice(), fold)
// }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a column*/
// pub fn fold_columns_mut<A: Clone, D, const W: usize, const H: usize>(matrix: &mut [[D; W]; H], init: &mut [A; W], fold: impl Fn(A, &mut D) -> A) {
//     mat_slice::fold_columns_mut(W, matrix.flatten_mut(), init.as_mut_slice(), fold)
// }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a row*/
// pub fn fold_rows_mut<A: Clone, D, const W: usize, const H: usize>(matrix: &mut [[D; W]; H], init: &mut [A; H], fold: impl Fn(A, &mut D) -> A) {
//     mat_slice::fold_rows_mut(W, matrix.flatten_mut(), init.as_mut_slice(), fold)
// }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function sums all elements in a column*/
// pub fn sum_columns<A: Clone + Add<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], sums: &mut [A; W]) {
//     fold_columns(matrix, sums, |a, b| a + b.clone())
// }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function sums all elements in a row*/
// pub fn sum_rows<A: Clone + Add<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], sums: &mut [A; H]) {
//     fold_rows(matrix, sums, |a, b| a + b.clone())
// }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies all elements in a column*/
// pub fn product_columns<A: Clone + Mul<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], products: &mut [A; W]) {
//     fold_columns(matrix, products, |a, b| a * b.clone())
// }
// // pub fn add_row<A,const W:usize,const H:usize>(mat:[[A;W];H], row:[A;W])->[[A;W];{H+1}]{
// //     _concat(mat,[row])
// // }
// //
// // pub fn add_column<A,const W:usize,const H:usize>(mat:[[A;W];H], column:[A;H])->[[A;{W+1}];H]{
// //     _zip_arr(mat,column,|row, col|_append(row,col))
// // }
//
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies all elements in a row*/
// pub fn product_rows<A: Clone + Mul<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], products: &mut [A; H]) {
//     fold_rows(matrix, products, |a, b| a * b.clone())
// }
// pub fn row_wise<A,B,C, const W: usize, const H: usize>(matrix: &[[A; W]; H], vec: &[B; W], mut zip: impl FnMut(&[A; W],&[B;W])->[C;W]) ->[[C;W];H]{
//     map_arr(matrix,|row|zip(row,vec))
// }
// pub fn row_wise_<A,B, const W: usize, const H: usize>(matrix: &mut [[A; W]; H], vec: &[B; W], mut zip: impl FnMut(&mut [A; W],&[B;W])){
//     for row in matrix{ zip(row,vec) }
// }
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies matrix with vector element-wise (row-wise)*/
// pub fn mul_row_wise_<A: Copy + MulAssign, const W: usize, const H: usize>(matrix: &mut [[A; W]; H], vec: &[A; W]) {
//     row_wise_(matrix, vec, |a, b|{a.mul_(b);})
// }
// /**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies matrix with vector element-wise (row-wise)*/
// pub fn mul_row_wise<A: Copy + Mul<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], vec: &[A; W]) -> [[A; W]; H]{
//     row_wise(matrix, vec, |a, b|mul1(a,b).into_arr())
// }




// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test1() {
//         for _ in 0..5 {
//             let mut arr1 = arange2::<u32, 4, 4>();
//             let arr2 = transpose(&arr1);
//             transpose_(&mut arr1);
//             assert_eq!(arr1, arr2);
//         }
//     }
// }