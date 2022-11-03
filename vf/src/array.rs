use std::array::IntoIter;
use std::cmp::Ordering;
use std::iter::{Chain, Cloned, Copied, Cycle, Enumerate, Filter, FilterMap, FlatMap, Flatten, Fuse, Inspect, Map, MapWhile, Peekable, Product, Rev, Scan, Skip, SkipWhile, StepBy, Sum, Take, TakeWhile, Zip};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use num_traits::{AsPrimitive, Float, MulAdd, MulAddAssign, One, Zero};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use crate::*;

pub fn empty<T: Copy, const DIM: usize>() -> [T; DIM] {
    unsafe { MaybeUninit::array_assume_init(MaybeUninit::uninit_array()) }
}

pub fn cut_out<T: Copy, const DIM: usize>(z: &[T; DIM], idx: usize) -> [T; DIM - 1] {
    let mut s: [T; DIM - 1] = empty();
    s[..idx].copy_from_slice(&z[..idx]);
    s[idx..].copy_from_slice(&z[idx + 1..]);
    s
}

pub fn insert<T: Copy, const DIM: usize>(z: &[T; DIM], idx: usize, elem: T) -> [T; DIM + 1] {
    let mut s: [T; DIM + 1] = empty();
    s[..idx].copy_from_slice(&z[..idx]);
    s[idx] = elem;
    s[idx + 1..].copy_from_slice(&z[idx..]);
    s
}


pub fn shape1<T, const DIM0: usize>(_mat: &[T; DIM0]) -> [usize; 1] {
    [DIM0]
}

pub fn shape2<T, const DIM0: usize, const DIM1: usize>(_mat: &[[T; DIM0]; DIM1]) -> [usize; 2] {
    [DIM1, DIM0]
}

pub fn shape3<T, const DIM0: usize, const DIM1: usize, const DIM2: usize>(_mat: &[[[T; DIM0]; DIM1]; DIM2]) -> [usize; 3] {
    [DIM2, DIM1, DIM0]
}

pub fn shape4<T, const DIM0: usize, const DIM1: usize, const DIM2: usize, const DIM3: usize>(_mat: &[[[[T; DIM0]; DIM1]; DIM2]; DIM3]) -> [usize; 4] {
    [DIM3, DIM2, DIM1, DIM0]
}

pub fn shape5<T, const DIM0: usize, const DIM1: usize, const DIM2: usize, const DIM3: usize, const DIM4: usize>(_mat: &[[[[[T; DIM0]; DIM1]; DIM2]; DIM3]; DIM4]) -> [usize; 5] {
    [DIM4, DIM3, DIM2, DIM1, DIM0]
}

pub fn col_vec<T: Sized, const DIM0: usize>(mat: [T; DIM0]) -> [[T; 1]; DIM0] {
    mat.map(|x| [x])
}

pub fn row_vec<T: Sized, const DIM0: usize>(mat: [T; DIM0]) -> [[T; DIM0]; 1] {
    [mat]
}

pub fn unsqueeze2_1<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; 1]; DIM0] {
    col_vec(mat)
}

pub fn unsqueeze2_0<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; DIM0]; 1] {
    row_vec(mat)
}

pub fn squeeze2_1<T, const DIM1: usize>(mat: [[T; 1]; DIM1]) -> [T; DIM1] {
    mat.map(|[x]| x)
}

pub fn squeeze2_0<T, const DIM0: usize>(mat: [[T; DIM0]; 1]) -> [T; DIM0] {
    let [x] = mat;
    x
}

pub fn unsqueeze3_0<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; DIM0]; DIM1]; 1] {
    [mat]
}

pub fn unsqueeze3_1<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; DIM0]; 1]; DIM1] {
    unsqueeze2_1(mat)
}

pub fn unsqueeze3_2<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; 1]; DIM0]; DIM1] {
    mat.map(|x| unsqueeze2_1(x))
}


pub fn squeeze3_0<T, const DIM0: usize, const DIM1: usize>(mat: [[[T; DIM0]; DIM1]; 1]) -> [[T; DIM0]; DIM1] {
    let [x] = mat;
    x
}

pub fn squeeze3_1<T, const DIM0: usize, const DIM2: usize>(mat: [[[T; DIM0]; 1]; DIM2]) -> [[T; DIM0]; DIM2] {
    squeeze2_1(mat)
}

pub fn squeeze3_2<T, const DIM2: usize, const DIM1: usize>(mat: [[[T; 1]; DIM1]; DIM2]) -> [[T; DIM1]; DIM2] {
    mat.map(|x| squeeze2_1(x))
}


pub fn swap2<A, const W: usize, const H: usize>(mat: &mut [[A; W]; H], pos1: &[usize; 2], pos2: &[usize; 2]) {
    let shape = shape2(mat);
    assert!(all1(lt1(pos1, &shape)), "position1 {:?} out of bounds {:?}", pos1, shape);
    assert!(all1(lt1(pos2, &shape)), "position2 {:?} out of bounds {:?}", pos2, shape);
    unsafe {
        swap2_unchecked(mat, pos1, pos2)
    }
}

pub unsafe fn swap2_unchecked<A, const W: usize, const H: usize>(mat: &mut [[A; W]; H], pos1: &[usize; 2], pos2: &[usize; 2]) {
    let shape = shape2(mat);
    let i1 = idx(shape, c(pos1));
    let i2 = idx(shape, c(pos2));
    let p = mat.as_mut_ptr() as *mut A;
    unsafe {
        std::ptr::swap(p.add(i1), p.add(i2))
    }
}

pub fn swap3<A, const W: usize, const H: usize, const D: usize>(mat: &mut [[[A; W]; H]; D], pos1: &[usize; 3], pos2: &[usize; 3]) {
    let shape = shape3(mat);
    assert!(all1(lt1(pos1, &shape)), "position1 {:?} out of bounds {:?}", pos1, shape);
    assert!(all1(lt1(pos2, &shape)), "position2 {:?} out of bounds {:?}", pos2, shape);
    unsafe {
        swap3_unchecked(mat, pos1, pos2);
    }
}

pub unsafe fn swap3_unchecked<A, const W: usize, const H: usize, const D: usize>(mat: &mut [[[A; W]; H]; D], pos1: &[usize; 3], pos2: &[usize; 3]) {
    let shape = shape3(mat);
    let i1 = idx(shape, c(pos1));
    let i2 = idx(shape, c(pos2));
    let p = mat.as_mut_ptr() as *mut A;
    unsafe {
        std::ptr::swap(p.add(i1), p.add(i2))
    }
}

pub fn transpose_<A, const W: usize>(mat: &mut [[A; W]; W]) {
    for i in 0..W {
        for j in (i + 1)..W {
            unsafe { swap2_unchecked(mat, &[i, j], &[j, i]) };
        }
    }
}


pub fn mat3_to_mat4_fill<A: Clone>(mat: [[A; 3]; 3], fill_value: A, diagonal_fill_value: A) -> [[A; 4]; 4] {
    let [r0, r1, r2] = mat;
    [
        xyz_w4(r0, fill_value.clone()),
        xyz_w4(r1, fill_value.clone()),
        xyz_w4(r2, fill_value.clone()),
        [fill_value.clone(), fill_value.clone(), fill_value.clone(), diagonal_fill_value.clone()]
    ]
}

/**Same as mat3_to_mat4 but fill_value is 0 and diagonal_fill_value is 1*/
pub fn mat3_to_mat4<A: Zero + One>(mat: [[A; 3]; 3]) -> [[A; 4]; 4] {
    let [r0, r1, r2] = mat;
    [
        xyz_w4(r0, A::zero()),
        xyz_w4(r1, A::zero()),
        xyz_w4(r2, A::zero()),
        [A::zero(), A::zero(), A::zero(), A::one()]
    ]
}

pub fn mat2_add_column<A>(mat: [[A; 2]; 2], column: [A; 2]) -> [[A; 3]; 2] {
    let [r0, r1] = mat;
    let [c0, c1] = column;
    [
        xy_z3(r0, c0),
        xy_z3(r1, c1),
    ]
}

pub fn mat3_add_column<A>(mat: [[A; 3]; 3], column: [A; 3]) -> [[A; 4]; 3] {
    let [r0, r1, r2] = mat;
    let [c0, c1, c2] = column;
    [
        xyz_w4(r0, c0),
        xyz_w4(r1, c1),
        xyz_w4(r2, c2),
    ]
}

pub fn mat3x2_add_row<A>(mat: [[A; 3]; 2], row: [A; 3]) -> [[A; 3]; 3] {
    xy_z3(mat, row)
}

pub fn mat4x3_add_row<A>(mat: [[A; 4]; 3], row: [A; 4]) -> [[A; 4]; 4] {
    xyz_w4(mat, row)
}

pub fn mat3_add_row<A>(mat: [[A; 3]; 3], row: [A; 3]) -> [[A; 3]; 4] {
    xyz_w4(mat, row)
}

pub fn mat2_to_mat3<A: Clone>(mat: [[A; 2]; 2], fill_value: A) -> [[A; 3]; 3] {
    let [r0, r1] = mat;
    [
        xy_z3(r0, fill_value.clone()),
        xy_z3(r1, fill_value.clone()),
        [fill_value.clone(), fill_value.clone(), fill_value.clone()]
    ]
}

pub struct ArrayIter<'a, A: Array<DIM>, const DIM: usize> {
    i: usize,
    e: usize,
    a: &'a mut A,
}

impl<'a, A: Array<DIM>, const DIM: usize> ExactSizeIterator for ArrayIter<'a, A, DIM> {}

impl<'a, A: Array<DIM>, const DIM: usize> Iterator for ArrayIter<'a, A, DIM> {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.e {
            self.i = i + 1;
            Some(unsafe { self.a.get_unchkd(i) })
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.e - self.i;
        (l, Some(l))
    }
    fn count(self) -> usize where Self: Sized {
        self.e - self.i
    }
}

impl<'a, A: Array<DIM>, const DIM: usize> DoubleEndedIterator for ArrayIter<'a, A, DIM> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let e = self.e;
        if self.i < e {
            self.e = e - 1;
            Some(unsafe { self.a.get_unchkd(self.e) })
        } else {
            None
        }
    }
}



pub struct ArrayIntoIter<A: Array<DIM>, const DIM: usize> {
    i: usize,
    e: usize,
    a: A,
}

impl<A: Array<DIM>, const DIM: usize> ExactSizeIterator for ArrayIntoIter<A, DIM> {}

impl<A: Array<DIM>, const DIM: usize> Iterator for ArrayIntoIter<A, DIM> {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.e {
            self.i = i + 1;
            Some(unsafe { self.a.get_unchkd(i) })
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.e - self.i;
        (l, Some(l))
    }
    fn count(self) -> usize where Self: Sized {
        self.e - self.i
    }
}

impl<A: Array<DIM>, const DIM: usize> DoubleEndedIterator for ArrayIntoIter<A, DIM> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let e = self.e;
        if self.i < e {
            self.e = e - 1;
            Some(unsafe { self.a.get_unchkd(self.e) })
        } else {
            None
        }
    }
}

pub trait Array<const DIM: usize>: Sized {
    type Item;
    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item;
    fn into_iter(self) -> ArrayIntoIter<Self, DIM> {
        ArrayIntoIter { i: 0, a: self, e: DIM }
    }
    fn to_iter(&mut self) -> ArrayIter<Self, DIM> {
        ArrayIter { i: 0, a: self, e: DIM }
    }
    fn get_chkd(&mut self, i: usize) -> Self::Item {
        assert!(i < DIM, "Index {} out of bounds {}", i, DIM);
        unsafe { self.get_unchkd(i) }
    }
    fn len(&self) -> usize {
        DIM
    }
    fn into_arr(mut self) -> [Self::Item; DIM] {
        array_init::array_init(|i| unsafe { self.get_unchkd(i) })
    }
    /**Visits every element of the array, but doesn't do anything with it. This method
     is  useful only in the case when `get_unchkd` performs some mutable operations. */
    fn done(&mut self){
        for i in 0..DIM{
            unsafe{self.get_unchkd(i)};
        }
    }
}

impl<'a, T: Copy, const DIM: usize> Array<DIM> for [T; DIM] {
    type Item = T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        *self.get_unchecked(i)
    }
}
impl<'a, T: Sized, const DIM: usize> Array<DIM> for &'a [T; DIM] {
    type Item = &'a T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.get_unchecked(i)
    }
}

pub struct AddArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
macro_rules! impl_clone {
    ($t:ident $(,$vars:ident)* ; $($types:ident)* ; $($DIMS:ident)*) => {
        impl <$($types:Copy,)* $(const $DIMS:usize,)*> Copy for $t<$($types,)* $($DIMS,)*>{

        }
        impl <$($types:Copy,)* $(const $DIMS:usize,)*> Clone for $t<$($types,)* $($DIMS,)*>{
            fn clone(&self) -> Self {
                Self{$($vars:self.$vars.clone(),)*}
            }
        }
    };
}
macro_rules! impl_add {
    ($t:ident $(,$types:ident)*;$($DIMS:ident)* , $DIM:expr) => {
        impl <$($types,)* C:Array<$DIM> $(,const $DIMS:usize)*> Add<C> for $t<$($types,)* $($DIMS,)*> where Self:Array<$DIM>, <Self as Array<$DIM>>::Item:Add<C::Item> {
            type Output = AddArray<Self,C,$DIM>;

            fn add(self, rhs: C) -> Self::Output {
                add1(self,rhs)
            }
        }
    };
}
macro_rules! impl_sub {
    ($t:ident $(,$types:ident)*;$($DIMS:ident)* , $DIM:expr) => {
        impl <$($types,)* C:Array<$DIM> $(,const $DIMS:usize)*> Sub<C> for $t<$($types,)*  $($DIMS,)*> where Self:Array<$DIM>, <Self as Array<$DIM>>::Item:Sub<C::Item> {
            type Output = SubArray<Self,C,$DIM>;

            fn sub(self, rhs: C) -> Self::Output {
                sub1(self,rhs)
            }
        }
    };
}
macro_rules! impl_mul {
    ($t:ident $(,$types:ident)*;$($DIMS:ident)* , $DIM:expr) => {
        impl <$($types,)* C:Array<$DIM> $(,const $DIMS:usize)*> Mul<C> for $t<$($types,)*  $($DIMS,)*> where Self:Array<$DIM>, <Self as Array<$DIM>>::Item:Mul<C::Item> {
            type Output = MulArray<Self,C,$DIM>;

            fn mul(self, rhs: C) -> Self::Output {
                mul1(self,rhs)
            }
        }
    };
}
macro_rules! impl_div {
    ($t:ident $(,$types:ident)*;$($DIMS:ident)* , $DIM:expr) => {
        impl <$($types,)* C:Array<$DIM> $(,const $DIMS:usize)*> Div<C> for $t<$($types,)*  $($DIMS,)*> where Self:Array<$DIM>, <Self as Array<$DIM>>::Item:Div<C::Item> {
            type Output = DivArray<Self,C,$DIM>;

            fn div(self, rhs: C) -> Self::Output {
                div1(self,rhs)
            }
        }
    };
}
macro_rules! impl_rem {
    ($t:ident $(,$types:ident)*;$($DIMS:ident)* , $DIM:expr) => {
        impl <$($types,)* C:Array<$DIM> $(,const $DIMS:usize)*> Rem<C> for $t<$($types,)*  $($DIMS,)*> where Self:Array<$DIM>, <Self as Array<$DIM>>::Item:Rem<C::Item> {
            type Output = RemArray<Self,C,$DIM>;

            fn rem(self, rhs: C) -> Self::Output {
                rem1(self,rhs)
            }
        }
    };
}
macro_rules! impl_neg {
    ($t:ident $(,$types:ident)*;$($DIMS:ident)* , $DIM:expr) => {
        impl <$($types,)* $(const $DIMS:usize,)*> Neg for $t<$($types,)*  $($DIMS,)*> where Self:Array<$DIM>, <Self as Array<$DIM>>::Item:Neg{
            type Output = NegArray<Self,$DIM>;

            fn neg(self) -> Self::Output {
                neg1(self)
            }
        }
    };
}
macro_rules! impl_all_with_dims {
    ($t:ident $(,$types:ident)* ;$($DIMS:ident)* , $DIM:expr) => {
        impl_add!($t $(,$types)* ;$($DIMS)* , $DIM);
        impl_sub!($t $(,$types)* ;$($DIMS)* , $DIM);
        impl_mul!($t $(,$types)* ;$($DIMS)* , $DIM);
        impl_div!($t $(,$types)* ;$($DIMS)* , $DIM);
        impl_rem!($t $(,$types)* ;$($DIMS)* , $DIM);
        impl_neg!($t $(,$types)* ;$($DIMS)* , $DIM);
    };
}
macro_rules! impl_all {
    ($t:ident $(,$types:ident)*) => {
        impl_all_with_dims!($t $(,$types)* ; DIM, DIM);
    };
}
impl_all!(AddArray,A,B);
impl_clone!(AddArray,a,b; A B ; DIM);
impl<A: Array<DIM>, B: Array<DIM>, const DIM: usize> Array<DIM> for AddArray<A, B, DIM> where A::Item: Add<B::Item> {
    type Item = <A::Item as Add<B::Item>>::Output;
    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i) + self.b.get_unchkd(i)
    }
}

pub fn add1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> AddArray<A, B, DIM> where A::Item: Add<B::Item> {
    AddArray { a, b }
}


pub struct MulAddArray<A, B, C, const DIM: usize, > {
    a: A,
    b: B,
    c: C,
}
impl_clone!(MulAddArray,a,b,c ; A B C; DIM);
impl_all!(MulAddArray,A,B,D);

impl<B: Array<DIM>, A: Array<DIM>, C: Array<DIM>, const DIM: usize> Array<DIM> for MulAddArray<A, B, C, DIM> where A::Item: MulAdd<B::Item, C::Item> {
    type Item = <A::Item as MulAdd<B::Item, C::Item>>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        let c = self.c.get_unchkd(i);
        a.mul_add(b, c)
    }
}

pub fn mul_add1<A: Array<DIM>, B: Array<DIM>, C: Array<DIM>, const DIM: usize>(a: A, b: B, c: C) -> MulAddArray<A, B, C, DIM> where A::Item: MulAdd<B::Item, C::Item> {
    MulAddArray { a, b, c }
}


pub struct SubArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(SubArray,a,b ; A B ; DIM);
impl_all!(SubArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for SubArray<A, B, DIM> where A::Item: Sub<B::Item> {
    type Item = <A::Item as Sub<B::Item>>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a - b
    }
}

pub fn sub1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> SubArray<A, B, DIM> where A::Item: Sub<B::Item> {
    SubArray { a, b }
}


pub struct MulArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(MulArray,a,b ; A B ; DIM);
impl_all!(MulArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for MulArray<A, B, DIM> where A::Item: Mul<B::Item> {
    type Item = <A::Item as Mul<B::Item>>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a * b
    }
}

pub fn mul1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> MulArray<A, B, DIM> where A::Item: Mul<B::Item> {
    MulArray { a, b }
}

pub struct DivArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(DivArray,a,b ; A B ; DIM);
impl_all!(DivArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for DivArray<A, B, DIM> where A::Item: Div<B::Item> {
    type Item = <A::Item as Div<B::Item>>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a / b
    }
}

pub fn div1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> DivArray<A, B, DIM> where A::Item: Div<B::Item> {
    DivArray { a, b }
}

pub struct RemArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(RemArray,a,b ; A B ; DIM);
impl_all!(RemArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for RemArray<A, B, DIM> where A::Item: Rem<B::Item> {
    type Item = <A::Item as Rem<B::Item>>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a % b
    }
}

pub fn rem1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> RemArray<A, B, DIM> where A::Item: Rem<B::Item> {
    RemArray { a, b }
}


pub struct GtArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(GtArray,a,b ; A B ; DIM);
impl_all!(GtArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for GtArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a > b
    }
}

pub fn gt1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> GtArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    GtArray { a, b }
}


pub struct GeArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(GeArray,a,b ; A B ; DIM);
impl_all!(GeArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for GeArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a >= b
    }
}

pub fn ge1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> GeArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    GeArray { a, b }
}


pub struct LtArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(LtArray,a,b ; A B ; DIM);
impl_all!(LtArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for LtArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a < b
    }
}

pub fn lt1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> LtArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    LtArray { a, b }
}


pub struct LeArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(LeArray,a,b ; A B ; DIM);
impl_all!(LeArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for LeArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a <= b
    }
}

pub fn le1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> LeArray<A, B, DIM> where A::Item: PartialOrd<B::Item> {
    LeArray { a, b }
}


pub struct MaxArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(MaxArray,a,b ; A B ; DIM);
impl_all!(MaxArray,A,B);
impl<I, B: Array<DIM, Item=I>, A: Array<DIM, Item=I>, const DIM: usize> Array<DIM> for MaxArray<A, B, DIM> where I: PartialOrd<I> {
    type Item = I;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        if a > b { a } else { b }
    }
}

pub fn max1<I, B: Array<DIM, Item=I>, A: Array<DIM, Item=I>, const DIM: usize>(a: A, b: B) -> MaxArray<A, B, DIM> where I: PartialOrd<I> {
    MaxArray { a, b }
}


pub struct MinArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(MinArray,a,b ; A B ; DIM);
impl_all!(MinArray,A,B);
impl<I, B: Array<DIM, Item=I>, A: Array<DIM, Item=I>, const DIM: usize> Array<DIM> for MinArray<A, B, DIM> where I: PartialOrd<I> {
    type Item = I;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        if a < b { a } else { b }
    }
}

pub fn min1<I, B: Array<DIM, Item=I>, A: Array<DIM, Item=I>, const DIM: usize>(a: A, b: B) -> MinArray<A, B, DIM> where I: PartialOrd<I> {
    MinArray { a, b }
}


pub struct EqArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(EqArray,a,b ; A B ; DIM);
impl_all!(EqArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for EqArray<A, B, DIM> where A::Item: PartialEq<B::Item> {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a == b
    }
}

pub fn eq1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> EqArray<A, B, DIM> where A::Item: PartialEq<B::Item> {
    EqArray { a, b }
}

pub struct NeArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(NeArray,a,b ; A B ; DIM);
impl_all!(NeArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for NeArray<A, B, DIM> where A::Item: PartialEq<B::Item> {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a != b
    }
}

pub fn ne1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> NeArray<A, B, DIM> where A::Item: PartialEq<B::Item> {
    NeArray { a, b }
}


pub struct NegArray<A, const DIM: usize> {
    a: A,
}

impl_clone!(NegArray,a ; A; DIM);
impl_all!(NegArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for NegArray<A, DIM> where A::Item: Neg {
    type Item = <A::Item as Neg>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        -self.a.get_unchkd(i)
    }
}

pub fn neg1<A: Array<DIM>, const DIM: usize>(a: A) -> NegArray<A, DIM> where A::Item: Neg {
    NegArray { a }
}

pub struct FullArray<T, const DIM: usize> {
    a: T,
}
impl_clone!(FullArray,a ; A ; DIM);
impl_all!(FullArray,A);
impl<T: Clone, const DIM: usize> Array<DIM> for FullArray<T, DIM> {
    type Item = T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.clone()
    }
}

pub fn full1<T: Clone, const DIM: usize>(a: T) -> FullArray<T, DIM> {
    FullArray { a }
}

pub fn zeroes1<T: Clone + Zero, const DIM: usize>() -> FullArray<T, DIM> {
    full1(T::zero())
}

pub fn ones1<T: Clone + One, const DIM: usize>() -> FullArray<T, DIM> {
    full1(T::one())
}

#[derive(Copy, Clone)]
pub struct RangeArray<const DIM: usize>(usize);
impl_all!(RangeArray);
impl<const DIM: usize> Array<DIM> for RangeArray<DIM> {
    type Item = usize;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.0 + i
    }
}

pub fn arange1<const DIM: usize>(offset: usize) -> RangeArray<DIM> {
    RangeArray(offset)
}

//
// pub struct DeepRangeArray<A,const DIM1: usize,const DIM2: usize>{
//     offset:usize,
//     a:A
// }
// impl_clone!(DeepRangeArray, offset, a ; A ; DIM1 DIM2);
// impl_all_with_dims!(DeepRangeArray, A ; DIM1 DIM2 , DIM1);
// impl<A:Array<DIM2,Item=usize>,const DIM1: usize,const DIM2: usize> Array<DIM1> for RangeArray<A,DIM1,DIM2> {
//     type Item = usize;
//
//     unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
//         self.offset*DIM1 + self.a.get_unchkd(i)
//     }
// }
//
// pub fn arange2<const DIM: usize>() -> RangeArray<FullArray<> DIM> {
//     RangeArray(PhantomData)
// }


pub struct ClonedArray<A, const DIM: usize>(A);

impl<A: Copy, const DIM: usize> Copy for ClonedArray<A, DIM> {}

impl<A: Clone, const DIM: usize> Clone for ClonedArray<A, DIM> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl_all!(ClonedArray,A);
impl<'a, T: Clone + 'a, A: Array<DIM, Item=&'a T>, const DIM: usize> Array<DIM> for ClonedArray<A, DIM> {
    type Item = T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.0.get_unchkd(i).clone()
    }
}

pub fn cloned1<'a, T: Clone + 'a, A: Array<DIM, Item=&'a T>, const DIM: usize>(a: A) -> ClonedArray<A, DIM> {
    ClonedArray(a)
}

/**Just a short alias for cloned1*/
pub fn c1<'a, T: Clone + 'a, A: Array<DIM, Item=&'a T>, const DIM: usize>(a: A) -> ClonedArray<A, DIM> {
    cloned1(a)
}


pub struct MapArray<A, F, const DIM: usize> {
    a: A,
    f: F,
}
impl_clone!(MapArray,a,f ;A F ; DIM);
impl_all!(MapArray,A,F);
impl<B, A: Array<DIM>, F: Fn(A::Item) -> B, const DIM: usize> Array<DIM> for MapArray<A, F, DIM> {
    type Item = B;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        (self.f)(self.a.get_unchkd(i))
    }
}

pub fn map1<B, A: Array<DIM>, F: Fn(A::Item) -> B, const DIM: usize>(a: A, f: F) -> MapArray<A, F, DIM> {
    MapArray { a, f }
}


pub struct RandArray<T, R, const DIM: usize> {
    rng: R,
    _p: PhantomData<T>,
}
impl_clone!(RandArray,rng,_p ; T R ; DIM);
impl_all!(RandArray,T,R);
impl<T, R: Rng, const DIM: usize> Array<DIM> for RandArray<T, R, DIM> where Standard: Distribution<T> {
    type Item = T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.rng.gen()
    }
}

pub fn rand1<T, R: Rng, const DIM: usize>(rng: R) -> RandArray<T, R, DIM> where Standard: Distribution<T> {
    RandArray { rng, _p: PhantomData }
}

pub fn rnd1<T, const DIM: usize>() -> RandArray<T, rand::rngs::ThreadRng, DIM> where Standard: Distribution<T> {
    rand1(rand::thread_rng())
}

pub fn rand_arr<T, R: Rng, const DIM: usize>(rng: R) -> [T; DIM] where Standard: Distribution<T> {
    rand1(rng).into_arr()
}

pub fn rnd_arr<T, const DIM: usize>() -> [T; DIM] where Standard: Distribution<T> {
    rnd1().into_arr()
}


pub struct AsPrimitiveArray<A, B, const DIM: usize> {
    a: A,
    b: PhantomData<B>,
}
impl_clone!(AsPrimitiveArray,a,b ; A B ; DIM);
impl_all!(AsPrimitiveArray,A,B);
impl<B: Copy + 'static, A: Array<DIM>, const DIM: usize> Array<DIM> for AsPrimitiveArray<A, B, DIM> where A::Item: AsPrimitive<B> {
    type Item = B;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).as_()
    }
}

pub fn as1<B: Copy + 'static, A: Array<DIM>, const DIM: usize>(a: A) -> AsPrimitiveArray<A, B, DIM> where A::Item: AsPrimitive<B> {
    AsPrimitiveArray { a, b: PhantomData }
}


pub struct IfArray<A, B, C, const DIM: usize> {
    a: A,
    b: B,
    c: C,
}
impl_clone!(IfArray,a,b,c; A B C; DIM);
impl_all!(IfArray,A,B,D);

impl<B: Array<DIM>, A: Array<DIM, Item=bool>, C: Array<DIM, Item=B::Item>, const DIM: usize> Array<DIM> for IfArray<A, B, C, DIM> {
    type Item = B::Item;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        if self.a.get_unchkd(i) {
            self.b.get_unchkd(i)
        } else {
            self.c.get_unchkd(i)
        }
    }
}

pub fn if1<A: Array<DIM, Item=bool>, B: Array<DIM>, C: Array<DIM, Item=B::Item>, const DIM: usize>(condition: A, if_true: B, if_false: C) -> IfArray<A, B, C, DIM> {
    IfArray { a: condition, b: if_true, c: if_false }
}

pub fn add1_<'a, S: AddAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(a:&mut A, b: B){
    add_(a.to_iter(),b.into_iter())
}
pub fn _add1<'a, S: AddAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(mut a: A, b: B) -> A{
    add1_(&mut a,b);
    a
}
pub fn mul_add1_<'a, S: MulAddAssign<B::Item, C::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>, C: Array<DIM>,const DIM:usize>( a: &mut A, b: B, c:C)  {
    mul_add_(a.to_iter(),b.into_iter(), c.into_iter())
}
pub fn _mul_add1<'a, S: MulAddAssign<B::Item, C::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>, C: Array<DIM>,const DIM:usize>(mut a: A, b: B, c:C) -> A {
    mul_add1_(&mut a,b,c);
    a
}
pub fn sub1_<'a, S: SubAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>( a: &mut A, b: B){
    sub_(a.to_iter(),b.into_iter())
}
pub fn _sub1<'a, S: SubAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(mut a: A, b: B) -> A {
    sub1_(&mut a,b);
    a
}

pub fn mul1_<'a, S: MulAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(a: &mut A, b: B){
    mul_(a.to_iter(),b.into_iter())
}
pub fn _mul1<'a, S: MulAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(mut a: A, b: B) -> A{
    mul1_(&mut a,b);
    a
}


pub fn div1_<'a, S: DivAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(a: &mut A, b: B){
    div_(a.to_iter(),b.into_iter())
}
pub fn _div1<'a, S: DivAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(mut a: A, b: B)->A{
    div1_(&mut a,b);
    a
}

pub fn rem1_<'a, S: RemAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(a: &mut A, b: B){
    rem_(a.to_iter(),b.into_iter())
}
pub fn _rem1<'a, S: RemAssign<B::Item> + 'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM>,const DIM:usize>(mut a: A, b: B)->A{
    rem1_(&mut a,b);
    a
}

pub fn neg1_<'a, S: NegAssign + 'a, A: Array<DIM,Item=&'a mut S>,const DIM:usize>(a: &mut A){
    neg_(a.to_iter())
}

pub fn _neg1<'a, S: NegAssign + 'a, A: Array<DIM,Item=&'a mut S>,const DIM:usize>(mut a: A) -> A{
    neg1_(&mut a);
    a
}

pub fn assign1_<'a, S:'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM,Item=S>,const DIM:usize>(a: &mut A, b: B){
    assign_(a.to_iter(),b.into_iter())
}
pub fn _assign1<'a, S:'a, A: Array<DIM,Item=&'a mut S>, B: Array<DIM,Item=S>,const DIM:usize>(mut a: A, b: B) -> A{
    assign1_(&mut a,b);
    a
}

pub fn map1_<'a, S:'a, A: Array<DIM,Item=&'a mut S>,const DIM:usize>(a: &mut A, f: impl FnMut(&'a mut S)){
    a.to_iter().for_each(f)
}


pub struct SquareArray<A, const DIM: usize> {
    a: A,
}
impl_clone!(SquareArray,a; A; DIM);
impl_all!(SquareArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for SquareArray<A, DIM> where A::Item: Mul + Clone {
    type Item = <A::Item as Mul>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        a.clone() * a
    }
}

pub fn square1<A: Array<DIM>, const DIM: usize>(a: A) -> SquareArray<A, DIM> where A::Item: Mul + Clone {
    SquareArray { a }
}

pub struct CubeArray<A, const DIM: usize> {
    a: A,
}
impl_clone!(CubeArray,a; A; DIM);
impl_all!(CubeArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for CubeArray<A, DIM> where A::Item: Mul<Output=A::Item> + Clone {
    type Item = <A::Item as Mul>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        a.clone() * a.clone() * a
    }
}

pub fn cube1<A: Array<DIM>, const DIM: usize>(a: A) -> CubeArray<A, DIM> where A::Item: Mul<Output=A::Item> + Clone {
    CubeArray { a }
}


pub struct PowiArray<A, const DIM: usize> {
    a: A,
    pow: i32,
}

impl<A: Copy, const DIM: usize> Copy for PowiArray<A, DIM> {}

impl<A: Clone, const DIM: usize> Clone for PowiArray<A, DIM> {
    fn clone(&self) -> Self {
        Self { a: self.a.clone(), pow: self.pow }
    }
}
impl_all!(PowiArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for PowiArray<A, DIM> where A::Item: Float {
    type Item = A::Item;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).powi(self.pow)
    }
}

pub fn powi1<A: Array<DIM>, const DIM: usize>(a: A, pow: i32) -> PowiArray<A, DIM> where A::Item: Float {
    PowiArray { a, pow }
}

pub struct PowfArray<T, A, const DIM: usize> {
    a: A,
    pow: T,
}
impl_clone!(PowfArray,a,pow ; A B ; DIM);
impl_all!(PowfArray,T,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for PowfArray<A::Item, A, DIM> where A::Item: Float {
    type Item = A::Item;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).powf(self.pow)
    }
}

pub fn powf1<A: Array<DIM>, const DIM: usize>(a: A, pow: A::Item) -> PowfArray<A::Item, A, DIM> where A::Item: Float {
    PowfArray { a, pow }
}


pub struct SqrtArray<A, const DIM: usize> {
    a: A,
}
impl_clone!(SqrtArray,a; A; DIM);
impl_all!(SqrtArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for SqrtArray<A, DIM> where A::Item: Float {
    type Item = A::Item;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).sqrt()
    }
}

pub fn sqrt1<A: Array<DIM>, const DIM: usize>(a: A) -> SqrtArray<A, DIM> where A::Item: Float {
    SqrtArray { a }
}


pub struct AbsArray<A, const DIM: usize> {
    a: A,
}
impl_clone!(AbsArray,a; A; DIM);
impl_all!(AbsArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for AbsArray<A, DIM> where A::Item: Abs {
    type Item = <A::Item as Abs>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).abs()
    }
}

pub fn abs1<A: Array<DIM>, const DIM: usize>(a: A) -> AbsArray<A, DIM> where A::Item: Abs {
    AbsArray { a }
}


pub struct IsZeroArray<A, const DIM: usize> {
    a: A,
}
impl_clone!(IsZeroArray,a; A; DIM);
impl_all!(IsZeroArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for IsZeroArray<A, DIM> where A::Item: Zero {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).is_zero()
    }
}

pub fn is_zero1<A: Array<DIM>, const DIM: usize>(a: A) -> IsZeroArray<A, DIM> where A::Item: Zero {
    IsZeroArray { a }
}


pub struct IsOneArray<A, const DIM: usize> {
    a: A,
}
impl_clone!(IsOneArray,a; A; DIM);
impl_all!(IsOneArray,A);
impl<A: Array<DIM>, const DIM: usize> Array<DIM> for IsOneArray<A, DIM> where A::Item: One + PartialEq {
    type Item = bool;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        self.a.get_unchkd(i).is_one()
    }
}


pub fn is_one1<A: Array<DIM>, const DIM: usize>(a: A) -> IsOneArray<A, DIM> where A::Item: One + PartialEq {
    IsOneArray { a }
}


pub struct DistArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(DistArray,a,b; A B ; DIM);
impl_all!(DistArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for DistArray<A, B, DIM> where A::Item: Dist<B::Item> {
    type Item = <A::Item as Dist<B::Item>>::Output;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        a.dist(b)
    }
}

/**elementwise absolute difference |a-b|. Also works with unsigned types!*/
pub fn dist1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> DistArray<A, B, DIM> where A::Item: Dist<B::Item> {
    DistArray { a, b }
}


pub struct ZipArray<A, B, const DIM: usize> {
    a: A,
    b: B,
}
impl_clone!(ZipArray,a,b; A B ; DIM);
impl_all!(ZipArray,A,B);
impl<B: Array<DIM>, A: Array<DIM>, const DIM: usize> Array<DIM> for ZipArray<A, B, DIM> {
    type Item = (A::Item, B::Item);

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a = self.a.get_unchkd(i);
        let b = self.b.get_unchkd(i);
        (a, b)
    }
}

/**elementwise absolute difference |a-b|. Also works with unsigned types!*/
pub fn zip1<A: Array<DIM>, B: Array<DIM>, const DIM: usize>(a: A, b: B) -> ZipArray<A, B, DIM> {
    ZipArray { a, b }
}


pub struct ConcatArray<A, B, const DIM_A: usize, const DIM_B: usize> {
    a: A,
    b: B,
}

// impl_clone!(ConcatArray,a,b);
// impl_all_with_dims!(ConcatArray,A,B ; DIM_A DIM_B, {DIM_A+DIM_B});
impl<A: Array<DIM_A>, B: Array<DIM_B, Item=A::Item>, const DIM_A: usize, const DIM_B: usize> Array<{ DIM_A + DIM_B }> for ConcatArray<A, B, DIM_A, DIM_B> {
    type Item = A::Item;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        if i < DIM_A { self.a.get_unchkd(i) } else { self.b.get_unchkd(i) }
    }
}

/**elementwise absolute difference |a-b|. Also works with unsigned types!*/
pub fn concat1<A: Array<DIM_A>, B: Array<DIM_B, Item=A::Item>, const DIM_A: usize, const DIM_B: usize>(a: A, b: B) -> ConcatArray<A, B, DIM_A, DIM_B> {
    ConcatArray { a, b }
}


pub fn all1<A: Array<DIM, Item=bool>, const DIM: usize>(a: A) -> bool {
    a.into_iter().all(|b| b)
}

pub fn any1<A: Array<DIM, Item=bool>, const DIM: usize>(a: A) -> bool {
    a.into_iter().any(|b| b)
}

pub fn count1<A: Array<DIM, Item=bool>, const DIM: usize>(a: A) -> usize {
    a.into_iter().filter(|&a| a).count()
}

pub fn sum1<D: Zero + Add<I, Output = D>, I, A: Array<DIM, Item=I>, const DIM: usize>(a: A) -> D {
    sum(a.into_iter())
}

pub fn prod1<D: One + Mul<I, Output = D>, I, A: Array<DIM, Item=I>, const DIM: usize>(a: A) -> D {
    prod(a.into_iter())
}

pub fn dot1<T: Zero, I2, I1: MulAdd<I2, T, Output=T>, const DIM: usize>(lhs: impl Array<DIM, Item=I1>, rhs: impl Array<DIM, Item=I2>) -> T {
    dot(lhs.into_iter(), rhs.into_iter())
}


pub struct TakeArray<A, B, const DIM_A: usize, const DIM_B: usize> {
    a: A,
    b: B,
}
impl_clone!(TakeArray, a, b; A B; DIM_A DIM_B);
impl_all_with_dims!(TakeArray, A,B ; DIM_A DIM_B , DIM_B);
// impl <A,B, C:Array<DIM> ,const DIM_A:usize,const DIM_B:usize,const DIM:usize> Add<C> for ConcatArray<A, B, DIM_A, DIM_B>  where Self:Array<DIM>, <Self as Array<DIM>>::Item:Add<C::Item> {
//     type Output = AddArray<Self,C,DIM>;
//
//     fn add(self, rhs: C) -> Self::Output {
//         add1(self,rhs)
//     }
// }
impl<A: Array<DIM_A>, B: Array<DIM_B>, const DIM_A: usize, const DIM_B: usize> Array<DIM_B> for TakeArray<A, B, DIM_A, DIM_B> where B::Item: AsPrimitive<usize> {
    type Item = A::Item;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        let a_idx: usize = self.b.get_unchkd(i).as_();
        assert!(a_idx < DIM_A, "Index {} is out of bounds {}", a_idx, DIM_A);
        self.a.get_unchkd(a_idx)
    }
}

pub fn take1<A: Array<DIM_A>, B: Array<DIM_B>, const DIM_A: usize, const DIM_B: usize>(a: A, indices: B) -> TakeArray<A, B, DIM_A, DIM_B> where B::Item: AsPrimitive<usize> {
    TakeArray { a, b: indices }
}


pub fn pos1<T: RemDivAssign<A::Item>, A: Array<DIM>, const DIM: usize>(shape: A, index: T) -> [<T as Rem<A::Item>>::Output; DIM] {
    array_init::from_iter(pos(shape.into_iter(), index)).unwrap()
}

pub fn idx1<I1,I2,T: MulAdd<I1,I2,Output=T> + Zero, const DIM: usize>(shape: impl Array<DIM, Item=I1>, position: impl Array<DIM, Item=I2>) -> T {
    idx(shape.into_iter(), position.into_iter())
}

pub fn sparse_dot1<I: AsPrimitive<usize>, T: Add<Output=T> + Zero, const DIM: usize>(lhs: impl IntoIterator<Item=I>, mut rhs: impl Array<DIM, Item=T>) -> T {
    sum(lhs.into_iter().map(|i| rhs.get_chkd(i.as_())))
}


pub struct OneHot<T, const DIM: usize> {
    i: usize,
    one: T,
    zero: T,
}
impl_clone!(OneHot, i, one, zero; T; DIM);
impl_all!(OneHot,T);
impl<T: Clone, const DIM: usize> Array<DIM> for OneHot<T, DIM> {
    type Item = T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        if i == self.i { self.one.clone() } else { self.zero.clone() }
    }
}

pub fn one_hot1<T: Clone, const DIM: usize>(index: usize, one: T, zero: T) -> OneHot<T, DIM> {
    OneHot { i: index, one, zero }
}


pub struct OneHotMask<A, T, const DIM: usize> {
    i: usize,
    one: A,
    zero: T,
}
impl_clone!(OneHotMask, i, one, zero; A T; DIM);
impl_all!(OneHotMask,A, T);
impl<T: Clone, A: Array<DIM, Item=T>, const DIM: usize> Array<DIM> for OneHotMask<A, T, DIM> {
    type Item = T;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        if i == self.i { unsafe { self.one.get_unchkd(i) } } else { self.zero.clone() }
    }
}

pub fn one_hot_mask1<T: Clone, A: Array<DIM, Item=T>, const DIM: usize>(index: usize, one: A, zero: T) -> OneHotMask<A, T, DIM> {
    OneHotMask { i: index, one, zero }
}


pub struct IdMatArray<T, const DIM1: usize, const DIM2: usize> {
    one: T,
    zero: T,
}
impl_clone!(IdMatArray, one, zero; T; DIM1 DIM2);
impl_all_with_dims!(IdMatArray,T ;  DIM1 DIM2 , DIM2);
impl<T: Clone, const DIM1: usize, const DIM2: usize> Array<DIM1> for IdMatArray<T, DIM1, DIM2> {
    type Item = OneHot<T, DIM2>;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        one_hot1(i, self.one.clone(), self.zero.clone())
    }
}

/**Identity matrix*/
pub fn ident1<T: Clone, const DIM1: usize, const DIM2: usize>(one: T, zero: T) -> IdMatArray<T, DIM1, DIM2> {
    IdMatArray { one, zero }
}

/**Identity matrix*/
pub fn id1<T: Clone + One + Zero, const DIM1: usize, const DIM2: usize>() -> IdMatArray<T, DIM1, DIM2> {
    ident1(T::one(), T::zero())
}


pub struct DiagMatArray<A, T, const DIM: usize> {
    diag: A,
    zero: T,
}
impl_clone!(DiagMatArray, diag, zero; A T; DIM);
impl_all!(DiagMatArray,A,T);
impl<T: Clone, A: Array<DIM, Item=T> + Clone, const DIM: usize> Array<DIM> for DiagMatArray<A, T, DIM> {
    type Item = OneHotMask<A, T, DIM>;

    unsafe fn get_unchkd(&mut self, i: usize) -> Self::Item {
        one_hot_mask1(i, self.diag.clone(), self.zero.clone())
    }
}

/**Diagonal matrix*/
pub fn diagonal1<T: Clone, A: Array<DIM, Item=T> + Clone, const DIM: usize>(diag: A, zero: T) -> DiagMatArray<A, T, DIM> {
    DiagMatArray { diag, zero }
}

/**Diagonal matrix*/
pub fn diag1<T: Clone + Zero, A: Array<DIM, Item=T> + Clone, const DIM: usize>(diag: A) -> DiagMatArray<A, T, DIM> {
    diagonal1(diag, T::zero())
}


pub struct TransposedColumnArray<A, const DIM1: usize, const DIM2: usize> {
    col: usize,
    // DIM1 are rows, DIM2 are columns
    mat: A,
}
impl_clone!(TransposedColumnArray, col, mat; A ; DIM1 DIM2);
impl_all_with_dims!(TransposedColumnArray,A ; DIM1 DIM2 , DIM1);
impl<B: Array<DIM2>, A: Array<DIM1, Item=B>, const DIM1: usize, const DIM2: usize> Array<DIM1> for TransposedColumnArray<A, DIM1, DIM2> {
    type Item = B::Item;

    unsafe fn get_unchkd(&mut self, row: usize) -> Self::Item {
        self.mat.get_unchkd(row).get_unchkd(self.col)
    }
}


pub struct TransposedArray<A, const DIM1: usize, const DIM2: usize> {
    mat: A,// DIM1 are rows, DIM2 are columns
}
impl_clone!(TransposedArray, mat; A ; DIM1 DIM2);
impl_all_with_dims!(TransposedArray,A ; DIM1 DIM2 , DIM2);
impl<B: Array<DIM2> + Clone, A: Array<DIM1, Item=B> + Clone, const DIM1: usize, const DIM2: usize> Array<DIM2> for TransposedArray<A, DIM1, DIM2> {
    type Item = TransposedColumnArray<A, DIM1, DIM2>;

    unsafe fn get_unchkd(&mut self, col: usize) -> Self::Item {
        TransposedColumnArray { col, mat: self.mat.clone() }
    }
}

/**transposed matrix*/
pub fn transpose1<B: Array<DIM2> + Clone, A: Array<DIM1, Item=B> + Clone, const DIM1: usize, const DIM2: usize>(mat: A) -> TransposedArray<A, DIM1, DIM2> {
    TransposedArray { mat }
}

/**transposed matrix*/
pub fn t1<B: Array<DIM2> + Clone, A: Array<DIM1, Item=B> + Clone, const DIM1: usize, const DIM2: usize>(mat: A) -> TransposedArray<A, DIM1, DIM2> {
    transpose1(mat)
}


/*
pub fn dot3<T: Copy + Zero, const X: usize, const Y: usize, const Z: usize, const W: usize>(lhs: &[[[T; Z]; Y]; W], rhs: &[[[T; X]; Z]; W]) -> [[[T; X]; Y]; W] where for<'a> &'a T: MulAdd<Output=T> {
    let mut o: [[[T; X]; Y]; W] = empty();
    for w in 0..W {
        for x in 0..X {
            for y in 0..Y {
                o[w][y][x] = (0..Z).fold(T::zero(), |sum, z| lhs[w][y][z].mul_add(rhs[w][z][x], sum));
            }
        }
    }
    o
}

pub fn dot2<T: Copy + Zero, const X: usize, const Y: usize, const Z: usize>(lhs: &[[T; Z]; Y], rhs: &[[T; X]; Z]) -> [[T; X]; Y] {
    let mut o: [[T; X]; Y] = empty();
    for x in 0..X {
        for y in 0..Y {
            o[y][x] = (0..Z).fold(T::zero(), |sum, z| lhs[y][z].mul_add(rhs[z][x], sum));
        }
    }
    o
}

pub fn dot1<T: Copy + Zero, const X: usize, const Z: usize>(lhs: &[T; Z], rhs: &[[T; X]; Z]) -> [T; X] where for<'a> &'a T: MulAdd<Output=T> {
    let mut o = empty();
    for x in 0..X {
        o[x] = (0..Z).fold(T::zero(), |sum, z| lhs[z].mul_add(rhs[z][x], sum));
    }
    o
}

pub fn dot0<T: Zero, const Z: usize>(lhs: &[T; Z], rhs: &[T; Z]) -> T where for<'a> &'a T: MulAdd<Output=T> {
    dot(lhs, rhs)
}


macro_rules! impl_dot1 {
    ($t:ident) => {
        impl <const X:usize> Dot for &[$t;X]{
            type Output = $t;

            fn dot(self, other: Self) -> Self::Output {
                dot1(self,other)
            }
        }
    };
}
impl_dot1!(f32);
impl_dot1!(f64);
impl_dot1!(usize);
impl_dot1!(u8);
impl_dot1!(u16);
impl_dot1!(u32);
impl_dot1!(u64);
impl_dot1!(isize);
impl_dot1!(i8);
impl_dot1!(i16);
impl_dot1!(i32);
impl_dot1!(i64);

macro_rules! impl_dot1 {
    ($t:ident) => {
        impl <const X:usize, const Z:usize> Dot<&[[$t; X]; Z]> for &[$t;Z] {
            type Output = [$t; X];

            fn dot(self, other: &[[$t; X]; Z]) -> Self::Output {
                dot1(self,other)
            }
        }
    };
}
impl_dot1!(f32);
impl_dot1!(f64);
impl_dot1!(usize);
impl_dot1!(u8);
impl_dot1!(u16);
impl_dot1!(u32);
impl_dot1!(u64);
impl_dot1!(isize);
impl_dot1!(i8);
impl_dot1!(i16);
impl_dot1!(i32);
impl_dot1!(i64);

macro_rules! impl_dot2 {
    ($t:ident) => {
        impl <const X:usize, const Y:usize, const Z:usize> Dot<&[[$t; X]; Z]> for &[[$t; Z]; Y] {
            type Output = [[$t; X]; Y];

            fn dot(self, other: &[[$t; X]; Z]) -> Self::Output {
                dot2(self,other)
            }
        }
    };
}
impl_dot2!(f32);
impl_dot2!(f64);
impl_dot2!(usize);
impl_dot2!(u8);
impl_dot2!(u16);
impl_dot2!(u32);
impl_dot2!(u64);
impl_dot2!(isize);
impl_dot2!(i8);
impl_dot2!(i16);
impl_dot2!(i32);
impl_dot2!(i64);


macro_rules! impl_dot3 {
    ($t:ident) => {
        impl <const X:usize, const Y:usize, const Z:usize, const W:usize> Dot<&[[[$t; X]; Z]; W]> for &[[[$t; Z]; Y]; W]  {
            type Output = [[[$t; X]; Y]; W];

            fn dot(self, other: &[[[$t; X]; Z]; W]) -> Self::Output {
                dot3(self,other)
            }
        }
    };
}
impl_dot3!(f32);
impl_dot3!(f64);
impl_dot3!(usize);
impl_dot3!(u8);
impl_dot3!(u16);
impl_dot3!(u32);
impl_dot3!(u64);
impl_dot3!(isize);
impl_dot3!(i8);
impl_dot3!(i16);
impl_dot3!(i32);
impl_dot3!(i64);



pub fn sparse_dot2<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const X: usize, const Z: usize>(lhs: &[I], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o: [T; X] = empty();
    for x in 0..X {
        o[x] = lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()][x])
    }
    o
}

pub fn sparse_dot1<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const Z: usize>(lhs: &[I], rhs: &[T; Z]) -> T {
    lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()])
}
*/


#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test3() {
    //     let shape = shape3(&mat);
    //     let mut index = 0;
    //     for i in 0..4 {
    //         for j in 0..3 {
    //             for k in 0..5 {
    //                 let pos = [i, j, k];
    //                 let idx2 = idx(shape, &pos);
    //                 assert_eq!(index, idx2);
    //                 assert!(pos.lt(&shape));
    //                 index += 1;
    //             }
    //         }
    //     }
    // }
}