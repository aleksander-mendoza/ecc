use std::iter::Sum;
use std::ops::{Add, AddAssign, DivAssign, Mul};
use std::process::Output;
use num_traits::{AsPrimitive, FromPrimitive, Pow, Zero};
use crate::from_usize::FromUsize;
use crate::{VectorFieldDivAssign, VectorFieldZero};
use crate::levenshtein;

pub trait Norm {
    type Output;
    fn norm(self) -> Self::Output;
}
macro_rules! norm_abs {
    ($t:ident) => {
        impl Norm for $t{
            type Output = $t;

            fn norm(self) -> Self::Output {
                self.abs()
            }
        }
    };
}
/**identity function*/
macro_rules! norm_id {
    ($t:ident) => {
        impl Norm for $t{
            type Output = $t;

            fn norm(self) -> Self::Output {
                self
            }
        }
    };
}

norm_abs!(f32);
norm_abs!(f64);
norm_id!(bool);
norm_id!(usize);
norm_id!(u8);
norm_id!(u16);
norm_id!(u32);
norm_id!(u64);
norm_abs!(isize);
norm_abs!(i8);
norm_abs!(i16);
norm_abs!(i32);
norm_abs!(i64);

impl Norm for (f32, f32) {
    type Output = f32;

    fn norm(self) -> Self::Output {
        let (a, b) = self;
        f32::sqrt(a * a + b * b)
    }
}

impl Norm for (f64, f64) {
    type Output = f64;

    fn norm(self) -> Self::Output {
        let (a, b) = self;
        f64::sqrt(a * a + b * b)
    }
}

impl Norm for &str {
    type Output = usize;

    fn norm(self) -> Self::Output {
        self.len()
    }
}

impl Norm for &[f32] {
    type Output = f32;

    fn norm(self) -> Self::Output {
        l2(self, 1)
    }
}

impl Norm for &[f64] {
    type Output = f64;

    fn norm(self) -> Self::Output {
        l2(self, 1)
    }
}


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

pub fn ln<D: Add<Output=D> + Zero + Mul<Output=D> + Copy + FromUsize>(n: usize) -> fn(&[D], usize) -> D where for<'a> &'a D: Norm<Output=D> {
    match n {
        0 => l0,
        1 => l1,
        2 => l2,
        3 => l3,
        _ => panic!("Unknown norm")
    }
}

pub trait Dist {
    type Output;
    fn dist(self, other: Self) -> Self::Output;
}
/**Use this macro only for primitive types*/
macro_rules! dist_metric_induced_by_norm {
    ($t:ident) => {
        impl Dist for $t{
            type Output = $t;

            fn dist(self,other:Self) -> Self::Output {
                if self > other{ // works even with unsigned primitives
                    self - other
                }else{
                    other - self
                }
            }
        }
    };
}
impl Dist for bool {
    type Output = bool;

    fn dist(self, other: Self) -> Self::Output {
        self == other // this is discrete metric
    }
}
dist_metric_induced_by_norm!(f32);
dist_metric_induced_by_norm!(f64);
dist_metric_induced_by_norm!(usize);
dist_metric_induced_by_norm!(u8);
dist_metric_induced_by_norm!(u16);
dist_metric_induced_by_norm!(u32);
dist_metric_induced_by_norm!(u64);
dist_metric_induced_by_norm!(isize);
dist_metric_induced_by_norm!(i8);
dist_metric_induced_by_norm!(i16);
dist_metric_induced_by_norm!(i32);
dist_metric_induced_by_norm!(i64);

impl<X: Zero + AddAssign, const DIM: usize> Dist for &[X; DIM] where for<'a> &'a X: Dist<Output=X> {
    type Output = X;

    fn dist(self, other: Self) -> Self::Output {
        let mut sum = X::zero();
        for i in 0..DIM {
            sum += self[i].dist(&other[i])
        }
        sum
    }
}

impl<X: Zero + AddAssign> Dist for &[X] where for<'a> &'a X: Dist<Output=X> {
    type Output = X;

    fn dist(self, other: Self) -> Self::Output {
        assert_eq!(self.len(), other.len());
        let mut sum = X::zero();
        for i in 0..self.len() {
            sum += self[i].dist(&other[i])
        }
        sum
    }
}


impl Dist for &str {
    type Output = usize;

    fn dist(self, other: Self) -> Self::Output {
        levenshtein(self, other)
    }
}