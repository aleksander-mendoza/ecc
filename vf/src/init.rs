use std::mem::MaybeUninit;

pub trait UninitEmpty{
    unsafe fn empty_uninit()->Self;
}
pub trait InitEmpty{
    /**SAFETY: T must be Copy. This way it does not need ot be dropped. If T is not Copy then use unsafe method UnInitEmpty::empty_uninit*/
    fn empty()->Self;
}
pub trait InitEmptyWithCapacity{
    type C;
    /**SAFETY: T must be Copy. This way it does not need ot be dropped. If T is not Copy then use unsafe method UninitEmptyWithCapacity::empty_uninit*/
    fn empty(capacity:Self::C)->Self;
}
pub trait UninitEmptyWithCapacity{
    type C;
    unsafe fn empty_uninit(capacity:Self::C)->Self;
}
pub trait InitWith<T>{
    fn init_with(f:impl FnMut(usize)->T)->Self;
}
pub trait InitFilled<T:Clone>{
    fn init_filled(f:T)->Self;
}
pub trait InitWithCapacity<T>{
    type C;
    fn init_with(capacity:Self::C, f:impl FnMut(usize)->T)->Self;
}
pub trait InitFilledCapacity<T:Clone>{
    type C;
    fn init_filled(capacity:Self::C, f:T)->Self;
}
pub trait InitFold<T:Clone>{
    fn init_fold(start:T, f:impl FnMut(T,usize)->T)->Self;
}
pub trait InitFoldWithCapacity<T:Clone>{
    type C;
    fn init_fold(capacity:Self::C, start:T, f:impl FnMut(T, usize)->T)->Self;
}
pub trait InitRFold<T:Clone>{
    fn init_rfold(end:T, f:impl FnMut(T,usize)->T) ->Self;
}
pub trait InitRFoldWithCapacity<T:Clone>{
    type C;
    fn init_rfold(capacity:Self::C, end:T, f:impl FnMut(T, usize)->T) ->Self;
}
impl <T:Copy,const DIM:usize> InitEmpty for [T;DIM]{
    /**SAFETY: T must be Copy. This way it does not need ot be dropped. If T is not Copy then use unsafe method UnInitEmpty::empty_uninit*/
    fn empty() -> Self {
        empty()
    }
}
impl <T:Copy,const DIM:usize> UninitEmpty for [T;DIM]{
    unsafe fn empty_uninit() -> Self {
        empty_uninit()
    }
}
impl <T:Copy> InitEmptyWithCapacity for Vec<T>{
    type C = usize;
    /**SAFETY: T must be Copy. This way it does not need ot be dropped. If T is not Copy then use unsafe method UninitEmptyWithCapacity::empty_uninit*/
    fn empty(capacity: usize) -> Self {
        unsafe{Vec::empty_uninit(capacity)}
    }
}
impl <T> UninitEmptyWithCapacity for Vec<T>{
    type C = usize;
    unsafe fn empty_uninit(capacity: usize) -> Self {
        let mut v = Vec::with_capacity(capacity);
        v.set_len(capacity);
        v
    }
}
impl <T,const DIM:usize> InitWith<T> for [T;DIM]{
    fn init_with(mut f:impl FnMut(usize)->T)->Self{
        let mut e:[MaybeUninit<T>;DIM] = MaybeUninit::uninit_array();
        e.iter_mut().enumerate().for_each(|(i,e)|{e.write(f(i));});
        unsafe{MaybeUninit::array_assume_init(e)}
    }
}
impl <T:Copy,const DIM:usize> InitFilled<T> for [T;DIM]{
    fn init_filled(t:T)->Self{
        [t;DIM]
    }
}
impl <T> InitWithCapacity<T> for Vec<T>{
    type C = usize;
    fn init_with(capacity: usize, mut f: impl FnMut(usize) -> T) -> Self {
        (0..capacity).map(f).collect()
    }
}
impl <T:Clone> InitFilledCapacity<T> for Vec<T>{
    type C = usize;
    fn init_filled(capacity:usize, f:T)->Self{
        vec![f;capacity]
    }
}
impl <T:Clone,const DIM:usize> InitFold<T> for [T;DIM]{
    fn init_fold(start: T, mut f: impl FnMut(T, usize) -> T) -> Self {
        init_fold(start,f)
    }
}
impl <T:Clone> InitFoldWithCapacity<T> for Vec<T>{
    type C = usize;
    fn init_fold(capacity: usize, mut start: T, mut f: impl FnMut(T, usize) -> T) -> Self {
        let mut v = Vec::with_capacity(capacity);
        for i in 0..capacity{
            v.push(start.clone());
            start=f(start,i);
        }
        v
    }
}
impl <T:Clone,const DIM:usize> InitRFold<T> for [T;DIM]{
    fn init_rfold(mut end: T, mut f: impl FnMut(T, usize) -> T) -> Self {
        init_rfold(end, f)
    }
}
impl <T:Clone> InitRFoldWithCapacity<T> for Vec<T>{
    type C = usize;
    fn init_rfold(capacity: usize, mut end: T, mut f: impl FnMut(T, usize) -> T) -> Self {
        let mut v = unsafe{Vec::empty_uninit(capacity)};
        for i in (0..capacity).rev(){
            unsafe{std::ptr::write(&mut v[i] as *mut T, end.clone());}
            end=f(end,i);
        }
        v
    }
}
pub unsafe fn empty_uninit<T, const DIM:usize>()->[T;DIM]{
    MaybeUninit::array_assume_init(MaybeUninit::uninit_array())
}
pub fn empty<T:Copy, const DIM:usize>()->[T;DIM]{
    unsafe{MaybeUninit::array_assume_init(MaybeUninit::uninit_array())}
}
pub fn init_fold<T:Clone, const DIM:usize>(start: T,mut  f: impl FnMut(T, usize) -> T)->[T;DIM]{
    let mut e:[MaybeUninit<T>;DIM] = MaybeUninit::uninit_array();
    e.iter_mut().enumerate().fold(start,|acc,(i,e)|{
        let a = f(acc,i);
        e.write(a.clone());
        a
    });
    unsafe{MaybeUninit::array_assume_init(e)}
}
pub fn init_rfold<T:Clone, const DIM:usize>(end: T,mut  f: impl FnMut(T, usize) -> T) ->[T;DIM]{
    let mut e:[MaybeUninit<T>;DIM] = MaybeUninit::uninit_array();
    e.iter_mut().enumerate().rev().fold(end,|acc,(i,e)|{
        let a = f(acc,i);
        e.write(a.clone());
        a
    });
    unsafe{MaybeUninit::array_assume_init(e)}
}

#[must_use]
#[inline(always)]
pub const fn uninit_array3<X,const N: usize, const M: usize, const L: usize>() -> [[[MaybeUninit<X>; N]; M]; L] {
    // SAFETY: An uninitialized `[MaybeUninit<_>; LEN]` is valid.
    unsafe { MaybeUninit::<[[[MaybeUninit<X>; N];M];L]>::uninit().assume_init() }
}
#[must_use]
#[inline(always)]
pub const fn uninit_array2<X,const N: usize, const M: usize>() -> [[MaybeUninit<X>; N]; M] {
    // SAFETY: An uninitialized `[MaybeUninit<_>; LEN]` is valid.
    unsafe { MaybeUninit::<[[MaybeUninit<X>; N];M]>::uninit().assume_init() }
}

#[inline(always)]
#[track_caller]
pub const unsafe fn array_assume_init2<X,const N: usize,const M: usize>(array: [[MaybeUninit<X>; N];M]) -> [[X; N]; M] {
    // SAFETY:
    // * The caller guarantees that all elements of the array are initialized
    // * `MaybeUninit<T>` and T are guaranteed to have the same layout
    // * `MaybeUninit` does not drop, so there are no double-frees
    // And thus the conversion is safe
    let ret = (&array as *const _ as *const [[X; N]; M]).read();

    // FIXME: required to avoid `~const Destruct` bound
    std::mem::forget(array);
    ret
}
#[inline(always)]
#[track_caller]
pub const unsafe fn array_assume_init3<X,const N: usize,const M: usize,const L: usize>(array: [[[MaybeUninit<X>; N];M];L]) -> [[[X; N]; M];L] {
    // SAFETY:
    // * The caller guarantees that all elements of the array are initialized
    // * `MaybeUninit<T>` and T are guaranteed to have the same layout
    // * `MaybeUninit` does not drop, so there are no double-frees
    // And thus the conversion is safe
    let ret = (&array as *const _ as *const [[[X; N]; M]; L]).read();

    // FIXME: required to avoid `~const Destruct` bound
    std::mem::forget(array);
    ret
}