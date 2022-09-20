use std::ops::{Add, Mul};
use std::process::Output;
use num_traits::{AsPrimitive, FromPrimitive, Pow, Zero};
use crate::from_usize::FromUsize;
use crate::VectorFieldZero;

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
        l2(self)
    }
}
impl Norm for [f64]{
    type Output = f64;

    fn norm(&self) -> Self::Output {
        l2(self)
    }
}


/**norm is the cardinality of a set (or length of a sequence)*/
pub fn card<D:FromUsize>(vec:&[D]) ->D{
    D::from_usize(vec.len())
}
pub fn l1<E:Add<Output=E>+Zero, D:Norm<Output=E>>(vec:&[D]) ->E{
    vec.iter().map(|f|f.norm()).fold(E::zero(),|a,b|a+b)
}

pub fn l2<D:Copy+Add<Output=D>+Zero + Mul<Output=D>>(vec:&[D]) ->D{
    vec.iter().fold(D::zero(),|a,b|a+*b* *b)
}

pub fn l3<E:Add<Output=E>+Zero+Mul<Output=E>+Copy, D:Norm<Output=E>>(vec:&[D]) ->E{
    vec.iter().map(|f|f.norm()).fold(E::zero(),|a,b|a+b* b* b)
}


