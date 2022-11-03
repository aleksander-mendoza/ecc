use std::collections::VecDeque;
use std::iter::{Cloned, Sum};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::path::Iter;
use num_traits::{AsPrimitive, Float, MulAdd, MulAddAssign, One, Zero};
use rand::distributions::Standard;
use rand::prelude::{Distribution, ThreadRng};
use rand::Rng;
use crate::*;

pub trait DoneIter: Iterator + Sized {
    fn done(self) {
        self.for_each(|_| ())
    }
}

pub struct AddIter<A: Iterator, B: Iterator> where A::Item: Add<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for AddIter<A, B> where A::Item: Add<B::Item> {
    type Item = <A::Item as Add<B::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a + b)
    }
}

pub fn add<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> AddIter<A, B> where A::Item: Add<B::Item> {
    AddIter { a: a.into_iter(), b: b.into_iter() }
}

pub trait Addition<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> Addition<B> for A where A::Item: Add<B::Item> {
    type Output = AddIter<A, B>;

    fn add(self, rhs: B) -> Self::Output {
        add(self, rhs)
    }
}


pub struct MulAddIter<A: Iterator, B: Iterator, C: Iterator> where A::Item: MulAdd<B::Item, C::Item> {
    a: A,
    b: B,
    c: C,
}

impl<B: Iterator, A: Iterator, C: Iterator> Iterator for MulAddIter<A, B, C> where A::Item: MulAdd<B::Item, C::Item> {
    type Item = <A::Item as MulAdd<B::Item, C::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        let c = self.c.next()?;
        Some(a.mul_add(b, c))
    }
}

pub fn mul_add<A: Iterator, B: Iterator, C: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>, c: impl IntoIterator<IntoIter=C>) -> MulAddIter<A, B, C> where A::Item: MulAdd<B::Item, C::Item> {
    MulAddIter { a: a.into_iter(), b: b.into_iter(), c: c.into_iter() }
}

pub trait MultiplyAdd<Mul = Self, Add = Self> {
    type Output;
    fn mul_add(self, mul: Mul, add: Add) -> Self::Output;
}

impl<A: Iterator, B: Iterator, C: Iterator> MultiplyAdd<B, C> for A where A::Item: MulAdd<B::Item, C::Item> {
    type Output = MulAddIter<A, B, C>;

    fn mul_add(self, mul: B, add: C) -> Self::Output {
        mul_add(self, mul, add)
    }
}


pub struct SubIter<A: Iterator, B: Iterator> where A::Item: Sub<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for SubIter<A, B> where A::Item: Sub<B::Item> {
    type Item = <A::Item as Sub<B::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a - b)
    }
}

pub fn sub<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> SubIter<A, B> where A::Item: Sub<B::Item> {
    SubIter { a: a.into_iter(), b: b.into_iter() }
}


pub trait Subtract<Rhs = Self> {
    type Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> Subtract<B> for A where A::Item: Sub<B::Item> {
    type Output = SubIter<A, B>;

    fn sub(self, rhs: B) -> Self::Output {
        sub(self, rhs)
    }
}

pub struct MulIter<A: Iterator, B: Iterator> where A::Item: Mul<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for MulIter<A, B> where A::Item: Mul<B::Item> {
    type Item = <A::Item as Mul<B::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a * b)
    }
}

pub fn mul<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> MulIter<A, B> where A::Item: Mul<B::Item> {
    MulIter { a: a.into_iter(), b: b.into_iter() }
}

pub trait Multiply<Rhs = Self> {
    type Output;
    fn mul(self, rhs: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> Multiply<B> for A where A::Item: Mul<B::Item> {
    type Output = MulIter<A, B>;

    fn mul(self, rhs: B) -> Self::Output {
        mul(self, rhs)
    }
}


pub struct DivIter<A: Iterator, B: Iterator> where A::Item: Div<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for DivIter<A, B> where A::Item: Div<B::Item> {
    type Item = <A::Item as Div<B::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a / b)
    }
}

pub fn div<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> DivIter<A, B> where A::Item: Div<B::Item> {
    DivIter { a: a.into_iter(), b: b.into_iter() }
}

pub trait Divide<Rhs = Self> {
    type Output;
    fn div(self, rhs: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> Divide<B> for A where A::Item: Div<B::Item> {
    type Output = DivIter<A, B>;

    fn div(self, rhs: B) -> Self::Output {
        div(self, rhs)
    }
}

pub struct RemIter<A: Iterator, B: Iterator> where A::Item: Rem<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for RemIter<A, B> where A::Item: Rem<B::Item> {
    type Item = <A::Item as Rem<B::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a % b)
    }
}

pub fn rem<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> RemIter<A, B> where A::Item: Rem<B::Item> {
    RemIter { a: a.into_iter(), b: b.into_iter() }
}

pub trait Reminder<Rhs = Self> {
    type Output;
    fn rem(self, rhs: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> Reminder<B> for A where A::Item: Rem<B::Item> {
    type Output = RemIter<A, B>;

    fn rem(self, rhs: B) -> Self::Output {
        rem(self, rhs)
    }
}


pub struct GtIter<A: Iterator, B: Iterator> where A::Item: PartialOrd<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for GtIter<A, B> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a > b)
    }
}

pub fn gt<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> GtIter<A, B> where A::Item: PartialOrd<B::Item> {
    GtIter { a: a.into_iter(), b: b.into_iter() }
}


pub struct GeIter<A: Iterator, B: Iterator> where A::Item: PartialOrd<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for GeIter<A, B> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a >= b)
    }
}

pub fn ge<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> GeIter<A, B> where A::Item: PartialOrd<B::Item> {
    GeIter { a: a.into_iter(), b: b.into_iter() }
}


pub struct LtIter<A: Iterator, B: Iterator> where A::Item: PartialOrd<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for LtIter<A, B> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a < b)
    }
}

pub fn lt<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> LtIter<A, B> where A::Item: PartialOrd<B::Item> {
    LtIter { a: a.into_iter(), b: b.into_iter() }
}


pub struct LeIter<A: Iterator, B: Iterator> where A::Item: PartialOrd<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for LeIter<A, B> where A::Item: PartialOrd<B::Item> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a <= b)
    }
}

pub fn le<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> LeIter<A, B> where A::Item: PartialOrd<B::Item> {
    LeIter { a: a.into_iter(), b: b.into_iter() }
}

pub trait PartialOrder<Rhs = Self> {
    type OutputLt;
    type OutputGt;
    type OutputGe;
    type OutputLe;
    fn lt(self, rhs: Rhs) -> Self::OutputLt;
    fn le(self, rhs: Rhs) -> Self::OutputLe;
    fn ge(self, rhs: Rhs) -> Self::OutputGe;
    fn gt(self, rhs: Rhs) -> Self::OutputGt;
}

impl<A: Iterator, B: Iterator> PartialOrder<B> for A where A::Item: PartialOrd<B::Item> {
    type OutputLt = LtIter<A, B>;
    type OutputGt = GtIter<A, B>;
    type OutputGe = GeIter<A, B>;
    type OutputLe = LeIter<A, B>;

    fn lt(self, rhs: B) -> Self::OutputLt {
        lt(self, rhs)
    }

    fn le(self, rhs: B) -> Self::OutputLe {
        le(self, rhs)
    }

    fn ge(self, rhs: B) -> Self::OutputGe {
        ge(self, rhs)
    }

    fn gt(self, rhs: B) -> Self::OutputGt {
        gt(self, rhs)
    }
}


pub struct EqIter<A: Iterator, B: Iterator> where A::Item: PartialEq<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for EqIter<A, B> where A::Item: PartialEq<B::Item> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a == b)
    }
}

pub fn eq<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> EqIter<A, B> where A::Item: PartialEq<B::Item> {
    EqIter { a: a.into_iter(), b: b.into_iter() }
}


pub trait PartialEquality<Rhs = Self> {
    type Output;
    fn eq(self, rhs: Rhs) -> Self::Output;
}


impl<A: Iterator, B: Iterator> PartialEquality<B> for A where A::Item: PartialOrd<B::Item> {
    type Output = EqIter<A, B>;
    fn eq(self, rhs: B) -> Self::Output {
        eq(self, rhs)
    }
}

pub struct NeIter<A: Iterator, B: Iterator> where A::Item: PartialEq<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for NeIter<A, B> where A::Item: PartialEq<B::Item> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a != b)
    }
}

pub fn ne<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> NeIter<A, B> where A::Item: PartialEq<B::Item> {
    NeIter { a: a.into_iter(), b: b.into_iter() }
}


pub trait PartialNotEquality<Rhs = Self> {
    type Output;
    fn ne(self, rhs: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> PartialNotEquality<B> for A where A::Item: PartialOrd<B::Item> {
    type Output = NeIter<A, B>;
    fn ne(self, rhs: B) -> Self::Output {
        ne(self, rhs)
    }
}

pub struct NegIter<A: Iterator> where A::Item: Neg {
    a: A,
}

impl<A: Iterator> Iterator for NegIter<A> where A::Item: Neg {
    type Item = <A::Item as Neg>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| -a)
    }
}

pub fn neg<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> NegIter<A> where A::Item: Neg {
    NegIter { a: a.into_iter() }
}


pub trait PartialNegation {
    type Output;
    fn neg(self) -> Self::Output;
}


impl<A: Iterator> PartialNegation for A where A::Item: Neg {
    type Output = NegIter<A>;
    fn neg(self) -> Self::Output {
        neg(self)
    }
}


pub struct FullIter<T: Clone>(T);

impl<T: Clone> Iterator for FullIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.clone())
    }
}

pub fn full<T: Clone>(a: T) -> FullIter<T> {
    FullIter(a)
}

pub fn zeroes<T: Clone + Zero>() -> FullIter<T> {
    full(T::zero())
}

pub fn ones<T: Clone + One>() -> FullIter<T> {
    full(T::one())
}


pub fn add_<'a, S: AddAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) {
    a.into_iter().zip(b).for_each(|(a, b)| *a += b)
}


pub trait AdditionAssign<Rhs = Self> {
    fn add_(self, rhs: Rhs);
}


impl<'a, S: AddAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator> AdditionAssign<B> for A {
    fn add_(self, rhs: B) {
        add_(self, rhs)
    }
}

pub fn mul_add_<'a, S: MulAddAssign<B::Item, C::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator, C: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>, c: impl IntoIterator<IntoIter=C>) {
    a.into_iter().zip(b).zip(c).for_each(|((a, b), c)| a.mul_add_assign(b, c))
}


pub trait MultiplyAddAssign<Mul = Self, Add = Self> {
    fn mul_add_(self, mul: Mul, add: Add);
}


impl<'a, S: MulAddAssign<B::Item, C::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator, C: Iterator> MultiplyAddAssign<B, C> for A {
    fn mul_add_(self, mul: B, add: C) {
        mul_add_(self, mul, add)
    }
}

pub fn sub_<'a, S: SubAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) {
    a.into_iter().zip(b).for_each(|(a, b)| *a -= b)
}


pub trait SubtractAssign<Rhs = Self> {
    fn sub_(self, rhs: Rhs);
}

impl<'a, S: SubAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator> SubtractAssign<B> for A {
    fn sub_(self, rhs: B) {
        sub_(self, rhs)
    }
}

pub fn mul_<'a, S: MulAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) {
    a.into_iter().zip(b).for_each(|(a, b)| *a *= b)
}


pub trait MultiplyAssign<Rhs = Self> {
    fn mul_(self, rhs: Rhs);
}


impl<'a, S: MulAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator> MultiplyAssign<B> for A {
    fn mul_(self, rhs: B) {
        mul_(self, rhs)
    }
}

pub fn div_<'a, S: DivAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) {
    a.into_iter().zip(b).for_each(|(a, b)| *a /= b)
}


pub trait DivideAssign<Rhs = Self> {
    fn div_(self, rhs: Rhs);
}


impl<'a, S: DivAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator> DivideAssign<B> for A {
    fn div_(self, rhs: B) {
        div_(self, rhs)
    }
}

pub fn rem_<'a, S: RemAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>){
    a.into_iter().zip(b).for_each(|(a, b)| *a %= b)
}


pub trait ReminderAssign<Rhs = Self> {
    fn rem_(self, rhs: Rhs);
}


impl<'a, S: RemAssign<B::Item> + 'a, A: Iterator<Item=&'a mut S>, B: Iterator> ReminderAssign<B> for A {
    fn rem_(self, rhs: B) {
        rem_(self, rhs)
    }
}

pub fn assign_<'a, S: 'a, A: Iterator<Item=&'a mut S>, B: Iterator<Item=S>>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) {
    a.into_iter().zip(b).for_each(|(a,b)|*a=b)
}


pub trait Assignment<Rhs = Self> {
    fn assign_(self, rhs: Rhs) ;
}


impl<'a, S: 'a, A: Iterator<Item=&'a mut S>, B: Iterator<Item=S>> Assignment<B> for A {
    fn assign_(self, rhs: B){
        assign_(self, rhs)
    }
}


pub fn neg_<'a, S: 'a+NegAssign, A: Iterator<Item=&'a mut S>>(a: impl IntoIterator<IntoIter=A>) {
    a.into_iter().for_each(NegAssign::neg_assign)
}

pub trait NegateAssign{
    fn neg_(self);
}

impl<'a, S: 'a+NegAssign, A: Iterator<Item=&'a mut S>> NegateAssign for A {
    fn neg_(self){
        neg_(self)
    }
}

pub struct SquareIter<A: Iterator> where A::Item: Mul + Clone {
    a: A,
}

impl<A: Iterator> Iterator for SquareIter<A> where A::Item: Mul + Clone {
    type Item = <A::Item as Mul>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| a.clone() * a)
    }
}

pub fn square<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> SquareIter<A> where A::Item: Mul + Clone {
    SquareIter { a: a.into_iter() }
}


pub trait Square {
    type Output;
    fn square(self) -> Self::Output;
}


impl<A: Iterator> Square for A where A::Item: Mul + Clone {
    type Output = SquareIter<A>;
    fn square(self) -> Self::Output {
        square(self)
    }
}


pub struct CubeIter<A: Iterator> where A::Item: Mul<Output=A::Item> + Clone {
    a: A,
}

impl<A: Iterator> Iterator for CubeIter<A> where A::Item: Mul<Output=A::Item> + Clone {
    type Item = <A::Item as Mul>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| a.clone() * a.clone() * a)
    }
}

pub fn cube<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> CubeIter<A> where A::Item: Mul<Output=A::Item> + Clone {
    CubeIter { a: a.into_iter() }
}

pub trait Cube {
    type Output;
    fn cube(self) -> Self::Output;
}


impl<A: Iterator> Cube for A where A::Item: Mul<Output=A::Item> + Clone {
    type Output = CubeIter<A>;
    fn cube(self) -> Self::Output {
        cube(self)
    }
}


pub struct PowiIter<A: Iterator> where A::Item: Float {
    a: A,
    pow: i32,
}

impl<A: Iterator> Iterator for PowiIter<A> where A::Item: Float {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let pow = self.pow;
        self.a.next().map(|a| a.powi(pow))
    }
}

pub fn powi<A: Iterator>(a: impl IntoIterator<IntoIter=A>, pow: i32) -> PowiIter<A> where A::Item: Float {
    PowiIter { a: a.into_iter(), pow }
}

pub trait Powi {
    type Output;
    fn powi(self, pow: i32) -> Self::Output;
}


impl<A: Iterator> Powi for A where A::Item: Float {
    type Output = PowiIter<A>;
    fn powi(self, pow: i32) -> Self::Output {
        powi(self, pow)
    }
}


pub struct PowfIter<A: Iterator> where A::Item: Float {
    a: A,
    pow: A::Item,
}

impl<A: Iterator> Iterator for PowfIter<A> where A::Item: Float {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let pow = self.pow;
        self.a.next().map(|a| a.powf(pow))
    }
}

pub fn powf<A: Iterator>(a: impl IntoIterator<IntoIter=A>, pow: A::Item) -> PowfIter<A> where A::Item: Float {
    PowfIter { a: a.into_iter(), pow }
}

pub trait Powf<D = Self> {
    type Output;
    fn powf(self, pow: D) -> Self::Output;
}


impl<A: Iterator> Powf<A::Item> for A where A::Item: Float {
    type Output = PowfIter<A>;
    fn powf(self, pow: A::Item) -> Self::Output {
        powf(self, pow)
    }
}

pub struct SqrtIter<A: Iterator> where A::Item: Float {
    a: A,
}

impl<A: Iterator> Iterator for SqrtIter<A> where A::Item: Float {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| a.sqrt())
    }
}

pub fn sqrt<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> SqrtIter<A> where A::Item: Float {
    SqrtIter { a: a.into_iter() }
}

pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}


impl<A: Iterator> Sqrt for A where A::Item: Float {
    type Output = SqrtIter<A>;
    fn sqrt(self) -> Self::Output {
        sqrt(self)
    }
}

pub struct AbsIter<A: Iterator> where A::Item: Abs {
    a: A,
}

impl<A: Iterator> Iterator for AbsIter<A> where A::Item: Abs {
    type Item = <A::Item as Abs>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| a.abs())
    }
}


pub fn abs<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> AbsIter<A> where A::Item: Abs {
    AbsIter { a: a.into_iter() }
}


pub trait Absolute {
    type Output;
    fn abs(self) -> Self::Output;
}

impl<A: Iterator> Absolute for A where A::Item: Abs {
    type Output = AbsIter<A>;
    fn abs(self) -> Self::Output {
        abs(self)
    }
}

pub struct IsZeroIter<A: Iterator> where A::Item: Zero {
    a: A,
}

impl<A: Iterator> Iterator for IsZeroIter<A> where A::Item: Zero {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| a.is_zero())
    }
}

pub fn is_zero<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> IsZeroIter<A> where A::Item: Zero {
    IsZeroIter { a: a.into_iter() }
}

pub trait IsZero {
    type Output;
    fn is_zero(self) -> Self::Output;
}


impl<A: Iterator> IsZero for A where A::Item: Float {
    type Output = IsZeroIter<A>;
    fn is_zero(self) -> Self::Output {
        is_zero(self)
    }
}

pub struct DistIter<A: Iterator, B: Iterator> where A::Item: Dist<B::Item> {
    a: A,
    b: B,
}

impl<B: Iterator, A: Iterator> Iterator for DistIter<A, B> where A::Item: Dist<B::Item> {
    type Item = <A::Item as Dist<B::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some(a.dist(b))
    }
}

/**elementwise absolute difference |a-b|. Also works with unsigned types!*/
pub fn dist<A: Iterator, B: Iterator>(a: impl IntoIterator<IntoIter=A>, b: impl IntoIterator<IntoIter=B>) -> DistIter<A, B> where A::Item: Dist<B::Item> {
    DistIter { a: a.into_iter(), b: b.into_iter() }
}


pub trait Distance<Rhs = Self> {
    type Output;
    fn dist(self, other: Rhs) -> Self::Output;
}

impl<A: Iterator, B: Iterator> Distance<B> for A where A::Item: Dist<B::Item> {
    type Output = DistIter<A, B>;
    fn dist(self, other: B) -> Self::Output {
        dist(self, other)
    }
}


pub fn all<A: Iterator<Item=bool>>(a: impl IntoIterator<IntoIter=A>) -> bool {
    a.into_iter().all(|b| b)
}

pub fn any<A: Iterator<Item=bool>>(a: impl IntoIterator<IntoIter=A>) -> bool {
    a.into_iter().any(|b| b)
}

pub fn count<A: Iterator<Item=bool>>(a: impl IntoIterator<IntoIter=A>) -> usize {
    a.into_iter().filter(|&a| a).count()
}


pub struct IsOneIter<A: Iterator> where A::Item: One + PartialEq {
    a: A,
}

impl<A: Iterator> Iterator for IsOneIter<A> where A::Item: One + PartialEq {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().map(|a| a.is_one())
    }
}


pub fn is_one<A: Iterator>(a: impl IntoIterator<IntoIter=A>) -> IsOneIter<A> where A::Item: One + PartialEq {
    IsOneIter { a: a.into_iter() }
}


pub trait IsOne {
    type Output;
    fn is_one(self) -> Self::Output;
}


impl<A: Iterator> IsOne for A where A::Item: One + PartialEq {
    type Output = IsOneIter<A>;
    fn is_one(self) -> Self::Output {
        is_one(self)
    }
}

/**cloned iterator*/
pub fn c<'a, T: Clone + 'a, I: Iterator<Item=&'a T>>(iter: impl IntoIterator<IntoIter=I>) -> Cloned<I> {
    iter.into_iter().cloned()
}

pub struct RandIter<T, R: Rng> {
    rng: R,
    _p: PhantomData<T>,
}

impl<T, R: Rng> Iterator for RandIter<T, R> where Standard: Distribution<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.rng.gen())
    }
}

pub fn rand<T, R: Rng>(rng: R) -> RandIter<T, R> where Standard: Distribution<T> {
    RandIter { rng, _p: PhantomData }
}

/**same as `rand(rand::thread_rng())`*/
pub fn rnd<T>() -> RandIter<T, ThreadRng> where Standard: Distribution<T> {
    rand(rand::thread_rng())
}

pub fn rand_vec<T, R: Rng>(rng: R, len: usize) -> Vec<T> where Standard: Distribution<T> {
    rand(rng).take(len).collect()
}

pub fn rnd_vec<T>(len: usize) -> Vec<T> where Standard: Distribution<T> {
    rnd().take(len).collect()
}

pub fn prod<D: One + Mul<I, Output=D>, I>(a: impl IntoIterator<Item=I> ) -> D {
    a.into_iter().fold(D::one(), |a, b| a * b)
}

pub fn sum<D: Zero + Add<I, Output=D>, I>(a: impl IntoIterator<Item=I> ) -> D {
    a.into_iter().fold(D::zero(), |a, b| a + b)
}


pub struct PosIter<T, I: DoubleEndedIterator> {
    index: T,
    shape: I,
}

impl<T: RemDivAssign<I::Item>, I: DoubleEndedIterator> Iterator for PosIter<T, I> {
    type Item = <T as Rem<I::Item>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.shape.next_back().map(|dim_size|self.index.rem_div_assign(dim_size))
    }
}

/**Position in a tensor of a given shape that corresponds to a specific index*/
pub fn pos<T: RemDivAssign<I::Item>, I: DoubleEndedIterator>(shape: impl IntoIterator<IntoIter=I>, index: T) -> PosIter<T, I> {
    PosIter { index, shape: shape.into_iter() }
}

/**Index into a tensor of a given shape that corresponds to a specific position*/
pub fn idx<I1, I2, T: MulAdd<I1, I2, Output=T> + Zero>(shape: impl IntoIterator<Item=I1>, position: impl IntoIterator<Item=I2>) -> T {
    shape.into_iter().zip(position.into_iter()).fold(T::zero(), |idx, (dim_size, pos_in_dim)| idx.mul_add(dim_size, pos_in_dim))
}

pub struct FoldMap<T, I: Iterator, F: Fn(I::Item)> {
    iter: I,
    acc: T,
    f: F,
}

pub fn fold_map<D: Add<Output=D> + Zero>(a: impl IntoIterator<Item=D>) -> D {
    a.into_iter().fold(D::zero(), |a, b| a + b)
}

pub fn norm_l0<D: Zero + FromUsize>(a: impl IntoIterator<Item=D>) -> D {
    D::from_usize(is_zero(a).filter(|&a| a).count())
}

pub fn norm_l1<D: Abs<Output=D> + Sum>(a: impl IntoIterator<Item=D>) -> D {
    abs(a).sum()
}

pub fn norm_l2<D: Float + Sum>(a: impl IntoIterator<Item=D>) -> D {
    D::sqrt(square(a).sum::<D>())
}

pub fn norm_l3<D: Float + Sum>(a: impl IntoIterator<Item=D>) -> D {
    D::cbrt(cube(a).sum::<D>())
}

pub fn norm_ln<D: Float + Sum + 'static>(a: impl IntoIterator<Item=D>, n: i32) -> D where i32: AsPrimitive<D> {
    D::powf(powi(a, n).sum::<D>(), D::one() / n.as_())
}


pub fn dist_l1<A: Dist<B>, B>(a: impl IntoIterator<Item=A>, b: impl IntoIterator<Item=B>) -> A::Output where A::Output: Sum {
    dist(a, b).sum()
}

pub fn dist_l2<D: Float + Sum>(a: impl IntoIterator<Item=D>, b: impl IntoIterator<Item=D>) -> D {
    norm_l2(sub(a, b))
}

pub fn dist_l3<D: Float + Sum>(a: impl IntoIterator<Item=D>, b: impl IntoIterator<Item=D>) -> D {
    norm_l3(sub(a, b))
}

pub fn dist_ln<D: Float + Sum + 'static>(a: impl IntoIterator<Item=D>, b: impl IntoIterator<Item=D>, n: i32) -> D where i32: AsPrimitive<D> {
    norm_ln(sub(a, b), n)
}

pub fn dot<T: Zero, I2, I1: MulAdd<I2, T, Output=T>>(a: impl IntoIterator<Item=I1>, b: impl IntoIterator<Item=I2>) -> T {
    a.into_iter().zip(b).fold(T::zero(), |sum, (a, b)| a.mul_add(b, sum))
}


pub struct TakeIter<'a, A: Iterator, T> where A::Item: AsPrimitive<usize> {
    indices: A,
    a: &'a [T],
}

impl<'a, A: Iterator, T> Iterator for TakeIter<'a, A, T> where A::Item: AsPrimitive<usize> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|i| &self.a[i.as_()])
    }
}

pub fn take<T, A: Iterator>(a: &[T], indices: impl IntoIterator<IntoIter=A>) -> TakeIter<A, T> where A::Item: AsPrimitive<usize> {
    TakeIter { a, indices: indices.into_iter() }
}



#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;


    #[test]
    fn test1() {
        let s = [4, 3];
        assert_eq!(s.idx([0, 0]), 0i32);
        assert_eq!(s.idx([0, 1]), 1i32);
        assert_eq!(s.idx([0, 2]), 2i32);
        assert_eq!(s.pos(0), [0, 0]);
        assert_eq!(s.pos(1), [0, 1]);
        assert_eq!(s.pos(2), [0, 2]);
        assert_eq!(s.pos(3), [1, 0]);
        assert_eq!(s.pos(4), [1, 1]);
        assert_eq!(s.pos(5), [1, 2]);
        for i in 0..(3 * 4) {
            let p = s.pos(i);
            assert_eq!(s.idx(p), i, "{}=={:?}", i, p);
        }
    }

    #[test]
    fn test2() {
        let s = [3, 4];
        for x in 0..3 {
            for y in 0..4 {
                assert_eq!(s.pos(s.idx([x, y])), [x, y]);
            }
        }
    }

    #[test]
    fn test3() {
        let s = [6, 4, 3];
        assert_eq!(s.idx([2, 0, 0]), 24);
        assert_eq!(s.idx([3, 0, 1]), 37);
        assert_eq!(s.idx([4, 0, 2]), 50);
        assert_eq!(s.pos(0), [0, 0, 0]);
        assert_eq!(s.pos(1), [0, 0, 1]);
        assert_eq!(s.pos(2), [0, 0, 2]);
        assert_eq!(s.pos(3), [0, 1, 0]);
        assert_eq!(s.pos(4), [0, 1, 1]);
        assert_eq!(s.pos(5), [0, 1, 2]);
        for i in 0..s.prod() {
            let p = s.pos(i);
            assert_eq!(s.idx(p), i, "{}=={:?}", i, p);
        }
    }

    #[test]
    fn test4() {
        let s = [6u32, 3, 4];
        for x in 0..s[2] {
            for y in 0..s[1] {
                for z in 0..s[0] {
                    assert_eq!(s.pos(s.idx([z, y, x])), [z, y, x]);
                }
            }
        }
    }
}