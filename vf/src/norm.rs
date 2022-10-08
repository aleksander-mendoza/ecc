use std::ops::{Add, DivAssign, Mul};
use std::process::Output;
use num_traits::{AsPrimitive, FromPrimitive, Pow, Zero};
use crate::from_usize::FromUsize;
use crate::{VectorFieldDivAssign, VectorFieldZero};

pub trait Norm{
    type Output;
    fn norm(&self)->Self::Output;
}
macro_rules! norm_abs {
    ($t:ident) => {
        impl Norm for $t{
            type Output = $t;

            fn norm(&self) -> Self::Output {
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

            fn norm(&self) -> Self::Output {
                *self
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

/**Complex norm*/
impl Norm for (f32,f32){
    type Output = f32;

    fn norm(&self) -> Self::Output {
        let (a,b) = self.clone();
        f32::sqrt(a*a+b*b)
    }
}
impl Norm for str{
    type Output = usize;

    fn norm(&self) -> Self::Output {
        self.len()
    }
}
impl Norm for [f32]{
    type Output = f32;

    fn norm(&self) -> Self::Output {
        l2(self, 1)
    }
}
impl Norm for [f64]{
    type Output = f64;

    fn norm(&self) -> Self::Output {
        l2(self, 1)
    }
}


/**norm is the cardinality of a set (or length of a sequence)*/
pub fn card<D:FromUsize>(vec:&[D]) ->D{
    D::from_usize(vec.len())
}

pub fn l0<D:Zero+FromUsize>(vec:&[D], stride:usize) ->D{
    D::from_usize(vec.iter().step_by(stride).filter(|f|!f.is_zero()).count())
}
pub fn l1<E:Add<Output=E>+Zero, D:Norm<Output=E>>(vec:&[D], stride:usize) ->E{
    vec.iter().step_by(stride).map(|f|f.norm()).fold(E::zero(),|a,b|a+b)
}

pub fn l2<D:Copy+Add<Output=D>+Zero + Mul<Output=D>>(vec:&[D], stride:usize) ->D{
    vec.iter().step_by(stride).fold(D::zero(),|a,b|a+*b* *b)
}

pub fn l3<E:Add<Output=E>+Zero+Mul<Output=E>+Copy, D:Norm<Output=E>>(vec:&[D], stride:usize) ->E{
    vec.iter().step_by(stride).map(|f|f.norm()).fold(E::zero(),|a,b|a+b* b* b)
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function normalizes all columns*/
pub fn normalize_mat_columns<D: DivAssign+Copy>(width:usize, matrix: &mut [D], norm_with_stride:impl Fn(&[D], usize)->D){
    for j in 0..width{
        let modulo_equivalence_class = &mut matrix[j..];
        let n = norm_with_stride(modulo_equivalence_class, width);
        modulo_equivalence_class.iter_mut().step_by(width).for_each(|w|*w /= n);
    }
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function normalizes all rows*/
pub fn normalize_mat_rows<D: DivAssign+Copy>(width:usize, matrix: &mut [D], norm_with_stride:impl Fn(&[D], usize)->D){
    let mut from = 0;
    while from < matrix.len(){
        let to = from + width;
        let row = &mut matrix[from..to];
        row.div_scalar_(norm_with_stride(row, 1));
        from = to;
    }
}
pub fn ln<D:Norm<Output=D>+Add<Output=D>+Zero+Mul<Output=D>+Copy+FromUsize>(n:usize)->fn(&[D],usize)->D{
    match n {
        0 => l0,
        1 => l1,
        2 => l2,
        3 => l3,
        _ => panic!("Unknown norm")
    }
}