use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};
use std::process::Output;
use num_traits::{AsPrimitive, Float, FromPrimitive, One, Pow, Zero};
use crate::from_usize::FromUsize;
use crate::*;
use crate::levenshtein;



/**norm is the cardinality of a set (or length of a sequence)*/
pub fn card<D: FromUsize>(vec: &[D]) -> D {
    D::from_usize(vec.len())
}

pub fn l0<D: Zero + FromUsize>(vec: &[D], stride: usize) -> D {
    D::from_usize(vec.iter().step_by(stride).filter(|f| !f.is_zero()).count())
}

pub fn l1<E: Add<Output=E> + Zero, D>(vec: &[D], stride: usize) -> E where for<'a> &'a D: Norm<Output=E> {
    vec.iter().step_by(stride).map(|f| f.norm()).fold(E::zero(), |a, b| a + b)
}

pub fn l2<D: Copy + Add<Output=D> + Zero + Mul<Output=D>>(vec: &[D], stride: usize) -> D {
    vec.iter().step_by(stride).fold(D::zero(), |a, b| a + *b * *b)
}

pub fn l3<E: Add<Output=E> + Zero + Mul<Output=E> + Copy, D>(vec: &[D], stride: usize) -> E where for<'a> &'a D: Norm<Output=E> {
    vec.iter().step_by(stride).map(|f| f.norm()).fold(E::zero(), |a, b| a + b * b * b)
}

pub fn _normalise<D: Div<Output=D> + One+MulAssign +Add<Output=D> + Zero + Mul<Output=D>+ Copy, const DIM:usize>(mut q:[D;DIM])->[D;DIM]{
    normalise_(&mut q);
    q
}

pub fn normalise_<D:Div<Output=D> + One+MulAssign +Add<Output=D> + Zero + Mul<Output=D>+ Copy>(q:&mut [D])->&mut [D]{
    q.mul_scalar_(D::one()/l2(q, 1))
}

pub fn normalise<D:Div<Output=D> +  One+MulAssign +Add<Output=D> + Zero + Mul<Output=D>+ Copy, const DIM:usize>(q:&[D;DIM])-> [D;DIM]{
    q.mul_scalar(D::one()/l2(q, 1))
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function normalizes all columns*/
pub fn normalize_mat_columns<D: DivAssign + Copy>(width: usize, matrix: &mut [D], norm_with_stride: impl Fn(&[D], usize) -> D) {
    for j in 0..width {
        let modulo_equivalence_class = &mut matrix[j..];
        let n = norm_with_stride(modulo_equivalence_class, width);
        modulo_equivalence_class.iter_mut().step_by(width).for_each(|w| *w /= n);
    }
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function normalizes all rows*/
pub fn normalize_mat_rows<D: DivAssign + Copy>(width: usize, matrix: &mut [D], norm_with_stride: impl Fn(&[D], usize) -> D) {
    let mut from = 0;
    while from < matrix.len() {
        let to = from + width;
        let row = &mut matrix[from..to];
        row.div_scalar_(norm_with_stride(row, 1));
        from = to;
    }
}

pub fn l<D: Add<Output=D> + Zero + Mul<Output=D> + Copy + FromUsize>(n: usize) -> fn(&[D], usize) -> D where for<'a> &'a D: Norm<Output=D> {
    match n {
        0 => l0,
        1 => l1,
        2 => l2,
        3 => l3,
        _ => panic!("Unknown norm")
    }
}
