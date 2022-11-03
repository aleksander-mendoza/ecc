use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use num_traits::{AsPrimitive, Float, MulAdd, MulAddAssign, One, Zero};
use crate::*;

pub trait ArrayOp<const DIM: usize>: Array<DIM> {
    fn as_<B: Copy + 'static>(self) -> [B; DIM] where Self::Item: AsPrimitive<B> {
        as1(self).into_arr()
    }
    fn add<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Add<A::Item>>::Output; DIM] where Self::Item: Add<A::Item> {
        add1(self, rhs).into_arr()
    }
    fn add_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Add<D>>::Output; DIM] where Self::Item: Add<D> {
        add1(self, full1(rhs)).into_arr()
    }
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn mul_add<B: Array<DIM>, C: Array<DIM>>(self, middle: B, rhs: C) -> [<Self::Item as MulAdd<B::Item, C::Item>>::Output; DIM] where Self::Item: MulAdd<B::Item, C::Item> {
        mul_add1(self, middle, rhs).into_arr()
    }
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn mul_scalar_add<B: Clone, C: Array<DIM>>(self, middle: B, rhs: C) -> [<Self::Item as MulAdd<B, C::Item>>::Output; DIM] where Self::Item: MulAdd<B, C::Item> {
        mul_add1(self, full1(middle), rhs).into_arr()
    }
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn mul_add_scalar<C: Clone, B: Array<DIM>>(self, middle: B, rhs: C) -> [<Self::Item as MulAdd<B::Item, C>>::Output; DIM] where Self::Item: MulAdd<B::Item, C> {
        mul_add1(self, middle, full1(rhs)).into_arr()
    }
    fn sum<T: Zero + Add<Self::Item, Output=T>>(self) -> T where Self::Item: Add<T, Output=T> {
        sum1(self)
    }
    fn prod<T: One + Mul<Self::Item, Output=T>>(self) -> T where Self::Item: Mul<T, Output=T> {
        prod1(self)
    }
    fn max<B: Array<DIM, Item=Self::Item>>(self, rhs: B) -> [Self::Item; DIM] where Self::Item: PartialOrd {
        max1(self, rhs).into_arr()
    }
    fn min<B: Array<DIM, Item=Self::Item>>(self, rhs: B) -> [Self::Item; DIM] where Self::Item: PartialOrd {
        min1(self, rhs).into_arr()
    }
    fn max_scalar(self, rhs: Self::Item) -> [Self::Item; DIM] where Self::Item: PartialOrd + Clone {
        max1(self, full1(rhs)).into_arr()
    }
    fn min_scalar(self, rhs: Self::Item) -> [Self::Item; DIM] where Self::Item: PartialOrd + Clone {
        min1(self, full1(rhs)).into_arr()
    }
    fn all(self, predicate: impl Fn(Self::Item) -> bool) -> bool {
        all1(map1(self, predicate))
    }
    fn any(self, predicate: impl Fn(Self::Item) -> bool) -> bool {
        any1(map1(self, predicate))
    }
    fn all_le<B: Array<DIM>>(self, rhs: B) -> bool where Self::Item: PartialOrd<B::Item> {
        all1(le1(self, rhs))
    }
    fn all_le_scalar<T: Clone>(self, rhs: T) -> bool where Self::Item: PartialOrd<T> {
        all1(le1(self, full1(rhs)))
    }
    fn all_lt<B: Array<DIM>>(self, rhs: B) -> bool where Self::Item: PartialOrd<B::Item> {
        all1(lt1(self, rhs))
    }
    fn all_lt_scalar<T: Clone>(self, rhs: T) -> bool where Self::Item: PartialOrd<T> {
        all1(lt1(self, full1(rhs)))
    }
    fn all_gt<B: Array<DIM>>(self, rhs: B) -> bool where Self::Item: PartialOrd<B::Item> {
        all1(gt1(self, rhs))
    }
    fn all_gt_scalar<T: Clone>(self, rhs: T) -> bool where Self::Item: PartialOrd<T> {
        all1(gt1(self, full1(rhs)))
    }
    fn all_ge<B: Array<DIM>>(self, rhs: B) -> bool where Self::Item: PartialOrd<B::Item> {
        all1(ge1(self, rhs))
    }
    fn all_ge_scalar<T: Clone>(self, rhs: T) -> bool where Self::Item: PartialOrd<T> {
        all1(ge1(self, full1(rhs)))
    }
    fn all_eq<B: Array<DIM>>(self, rhs: B) -> bool where Self::Item: PartialEq<B::Item> {
        all1(eq1(self, rhs))
    }
    fn all_eq_scalar<T: Clone>(self, rhs: T) -> bool where Self::Item: PartialEq<T> {
        all1(eq1(self, full1(rhs)))
    }
    fn all_ne<B: Array<DIM>>(self, rhs: B) -> bool where Self::Item: PartialEq<B::Item> {
        all1(ne1(self, rhs))
    }
    fn all_ne_scalar<T: Clone>(self, rhs: T) -> bool where Self::Item: PartialEq<T> {
        all1(ne1(self, full1(rhs)))
    }
    fn abs(self) -> [<Self::Item as Abs>::Output; DIM] where Self::Item: Abs {
        abs1(self).into_arr()
    }
    fn neg(self) -> [<Self::Item as Neg>::Output; DIM] where Self::Item: Neg {
        neg1(self).into_arr()
    }
    fn sub<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Sub<A::Item>>::Output; DIM] where Self::Item: Sub<A::Item> {
        sub1(self, rhs).into_arr()
    }
    fn sub_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Sub<D>>::Output; DIM] where Self::Item: Sub<D> {
        sub1(self, full1(rhs)).into_arr()
    }
    fn div<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Div<A::Item>>::Output; DIM] where Self::Item: Div<A::Item> {
        div1(self, rhs).into_arr()
    }
    fn div_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Div<D>>::Output; DIM] where Self::Item: Div<D> {
        div1(self, full1(rhs)).into_arr()
    }
    fn mul<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Mul<A::Item>>::Output; DIM] where Self::Item: Mul<A::Item> {
        mul1(self, rhs).into_arr()
    }
    fn mul_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Mul<D>>::Output; DIM] where Self::Item: Mul<D> {
        mul1(self, full1(rhs)).into_arr()
    }
    fn rem<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Rem<A::Item>>::Output; DIM] where Self::Item: Rem<A::Item> {
        rem1(self, rhs).into_arr()
    }
    fn rem_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Rem<D>>::Output; DIM] where Self::Item: Rem<D> {
        rem1(self, full1(rhs)).into_arr()
    }
    fn index<T: MulAdd<Self::Item, B::Item, Output=T> + Zero, B: Array<DIM>>(self, rhs: B) -> T {
        idx1(self, rhs)
    }
    fn idx<B: Array<DIM>>(self, rhs: B) -> Self::Item where Self::Item: MulAdd<Self::Item, B::Item, Output=Self::Item> + Zero {
        idx1(self, rhs)
    }
    fn pos<T: RemDivAssign<Self::Item> + Div<Self::Item, Output=T>>(self, index: T) -> [<T as Rem<Self::Item>>::Output; DIM] {
        pos1(self, index)
    }
}

pub trait ArrayCopyOp<'a, const DIM: usize>: Array<DIM, Item=&'a Self::C> {
    type C: Copy + 'a;
    fn cloned(self) -> ClonedArray<Self, DIM> where Self::Item: Clone {
        cloned1(self)
    }
    fn c(self) -> ClonedArray<Self, DIM> where Self::Item: Clone {
        c1(self)
    }
}

pub trait ArrayMutOp<'a, const DIM: usize>: Array<DIM, Item=&'a mut Self::C> {
    type C: 'a;
    fn assign_<A: Array<DIM,Item=Self::C>>(&mut self, rhs: A) {
        assign1_(self, rhs)
    }
    fn add_<A: Array<DIM>>(&mut self, rhs: A) where Self::C: AddAssign<A::Item> {
        add1_(self, rhs)
    }
    fn add_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: AddAssign<D> {
        add1_(self, full1(rhs))
    }
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn mul_add_<B: Array<DIM>, C: Array<DIM>>(&mut self, middle: B, rhs: C) where Self::C: MulAddAssign<B::Item, C::Item> {
        mul_add1_(self, middle, rhs)
    }
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn mul_add_scalar_<B: Array<DIM>, C:Clone>(&mut self, middle: B, rhs: C) where Self::C: MulAddAssign<B::Item, C> {
        mul_add1_(self, middle, full1(rhs))
    }
    fn div_<A: Array<DIM>>(&mut self, rhs: A) where Self::C: DivAssign<A::Item> {
        div1_(self, rhs)
    }
    fn div_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: DivAssign<D> {
        div1_(self, full1(rhs))
    }
    fn neg_(&mut self) where Self::C: Neg<Output=Self::C>+Clone {
        map1_(self,|a|*a=-a.clone())
    }
    /*
    fn max<B: Array<DIM, Item=Self::Item>>(self, rhs: B) -> [Self::Item; DIM] where Self::Item: PartialOrd {
        max1(self, rhs).into_arr()
    }
    fn min<B: Array<DIM, Item=Self::Item>>(self, rhs: B) -> [Self::Item; DIM] where Self::Item: PartialOrd {
        min1(self, rhs).into_arr()
    }
    fn max_scalar(self, rhs: Self::Item) -> [Self::Item; DIM] where Self::Item: PartialOrd + Clone {
        max1(self, full1(rhs)).into_arr()
    }
    fn min_scalar(self, rhs: Self::Item) -> [Self::Item; DIM] where Self::Item: PartialOrd + Clone {
        min1(self, full1(rhs)).into_arr()
    }
    fn abs(self) -> [<Self::Item as Abs>::Output; DIM] where Self::Item: Abs {
        abs1(self).into_arr()
    }
    fn neg(self) -> [<Self::Item as Neg>::Output; DIM] where Self::Item: Neg {
        neg1(self).into_arr()
    }
    fn sub<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Sub<A::Item>>::Output; DIM] where Self::Item: Sub<A::Item> {
        sub1(self, rhs).into_arr()
    }
    fn sub_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Sub<D>>::Output; DIM] where Self::Item: Sub<D> {
        sub1(self, full1(rhs)).into_arr()
    }
    fn div<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Div<A::Item>>::Output; DIM] where Self::Item: Div<A::Item> {
        div1(self, rhs).into_arr()
    }
    fn div_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Div<D>>::Output; DIM] where Self::Item: Div<D> {
        div1(self, full1(rhs)).into_arr()
    }
    fn mul<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Mul<A::Item>>::Output; DIM] where Self::Item: Mul<A::Item> {
        mul1(self, rhs).into_arr()
    }
    fn mul_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Mul<D>>::Output; DIM] where Self::Item: Mul<D> {
        mul1(self, full1(rhs)).into_arr()
    }
    fn rem<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Rem<A::Item>>::Output; DIM] where Self::Item: Rem<A::Item> {
        rem1(self, rhs).into_arr()
    }
    fn rem_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Rem<D>>::Output; DIM] where Self::Item: Rem<D> {
        rem1(self, full1(rhs)).into_arr()
    }
     */
}

pub trait ArrayOwnedOp<const DIM: usize>: Sized {
    type C;
    fn assign_<A: Array<DIM,Item=Self::C>>(&mut self, rhs: A);
    // fn add_<A: Array<DIM>>(&mut self, rhs: A) where Self::C: AddAssign<A::Item>;
    fn add_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: AddAssign<D>;
    // /**`self*middle + rhs` where * is element-wise multiplication*/
    // fn mul_add_<B: Array<DIM>, C: Array<DIM>>(&mut self, middle: B, rhs: C) where Self::C: MulAddAssign<B::Item, C::Item>;
    fn _add<A: Array<DIM>>(self, rhs: A) -> Self where Self::C: AddAssign<A::Item>;
    fn _add_scalar<D: Clone>(self, rhs: D) -> Self where Self::C: AddAssign<D>;
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn _mul_add<B: Array<DIM>, C: Array<DIM>>(self, middle: B, rhs: C) -> Self where Self::C: MulAddAssign<B::Item, C::Item>;
    fn mul_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: MulAssign<D>;
    fn div_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: DivAssign<D>;
    fn neg_(&mut self) where Self::C: NegAssign;
    fn _neg(self) -> Self where Self::C: NegAssign;
    /*
    fn max<B: Array<DIM, Item=Self::Item>>(self, rhs: B) -> [Self::Item; DIM] where Self::Item: PartialOrd {
        max1(self, rhs).into_arr()
    }
    fn min<B: Array<DIM, Item=Self::Item>>(self, rhs: B) -> [Self::Item; DIM] where Self::Item: PartialOrd {
        min1(self, rhs).into_arr()
    }
    fn max_scalar(self, rhs: Self::Item) -> [Self::Item; DIM] where Self::Item: PartialOrd + Clone {
        max1(self, full1(rhs)).into_arr()
    }
    fn min_scalar(self, rhs: Self::Item) -> [Self::Item; DIM] where Self::Item: PartialOrd + Clone {
        min1(self, full1(rhs)).into_arr()
    }
    fn abs(self) -> [<Self::Item as Abs>::Output; DIM] where Self::Item: Abs {
        abs1(self).into_arr()
    }
    fn neg(self) -> [<Self::Item as Neg>::Output; DIM] where Self::Item: Neg {
        neg1(self).into_arr()
    }
    fn sub<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Sub<A::Item>>::Output; DIM] where Self::Item: Sub<A::Item> {
        sub1(self, rhs).into_arr()
    }
    fn sub_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Sub<D>>::Output; DIM] where Self::Item: Sub<D> {
        sub1(self, full1(rhs)).into_arr()
    }
    fn div<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Div<A::Item>>::Output; DIM] where Self::Item: Div<A::Item> {
        div1(self, rhs).into_arr()
    }
    fn div_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Div<D>>::Output; DIM] where Self::Item: Div<D> {
        div1(self, full1(rhs)).into_arr()
    }
    fn mul<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Mul<A::Item>>::Output; DIM] where Self::Item: Mul<A::Item> {
        mul1(self, rhs).into_arr()
    }
    fn mul_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Mul<D>>::Output; DIM] where Self::Item: Mul<D> {
        mul1(self, full1(rhs)).into_arr()
    }
    fn rem<A: Array<DIM>>(self, rhs: A) -> [<Self::Item as Rem<A::Item>>::Output; DIM] where Self::Item: Rem<A::Item> {
        rem1(self, rhs).into_arr()
    }
    fn rem_scalar<D: Clone>(self, rhs: D) -> [<Self::Item as Rem<D>>::Output; DIM] where Self::Item: Rem<D> {
        rem1(self, full1(rhs)).into_arr()
    }
     */
}

impl<T: Array<DIM>, const DIM: usize> ArrayOp<DIM> for T {}

impl<T, const DIM: usize> ArrayOwnedOp<DIM> for [T; DIM] {
    type C = T;
    fn assign_<A: Array<DIM,Item=T>>(&mut self, rhs: A) {
        assign1_(&mut self, rhs)
    }
    // fn add_<A: Array<DIM>>(&mut self, rhs: A) where Self::C: AddAssign<A::Item> {
    //     add1_(self, rhs)
    // }
    fn add_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: AddAssign<D> {
        add1_(&mut self, full1(rhs))
    }
    // /**`self*middle + rhs` where * is element-wise multiplication*/
    // fn mul_add_<B: Array<DIM>, C: Array<DIM>>(&mut self, middle: B, rhs: C) where Self::C: MulAddAssign<B::Item, C::Item> {
    //     mul_add1_(self, middle, rhs)
    // }
    fn _add<A: Array<DIM>>(mut self, rhs: A) -> Self where Self::C: AddAssign<A::Item> {
        add1_(&mut &mut self, rhs);
        self
    }
    fn _add_scalar<D: Clone>(mut self, rhs: D) -> Self where Self::C: AddAssign<D> {
        add1_(&mut &mut self, full1(rhs));
        self
    }
    /**`self*middle + rhs` where * is element-wise multiplication*/
    fn _mul_add<B: Array<DIM>, C: Array<DIM>>(mut self, middle: B, rhs: C) -> Self where Self::C: MulAddAssign<B::Item, C::Item> {
        mul_add1_(&mut &mut self, middle, rhs);
        self
    }
    fn mul_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: MulAssign<D> {
        mul1_(&mut self, full1(rhs))
    }
    fn div_scalar_<D: Clone>(&mut self, rhs: D) where Self::C: DivAssign<D> {
        div1_(&mut self, full1(rhs))
    }
    fn neg_(&mut self) where Self::C: NegAssign {
        neg1_(&mut self)
    }
    fn _neg(self) -> Self where Self::C: NegAssign {
        neg1_(&mut &mut self);
        self
    }
}

impl<'a, C: 'a, T: Array<DIM, Item=&'a mut C>, const DIM: usize> ArrayMutOp<'a, DIM> for T {
    type C = C;
}

impl<'a, C: 'a + Copy, T: Array<DIM, Item=&'a C>, const DIM: usize> ArrayCopyOp<'a, DIM> for T {
    type C = C;
}