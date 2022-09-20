use std::mem::MaybeUninit;
use crate::init::empty;

pub fn concat<X:Copy,const L1:usize, const L2:usize>(a:&[X;L1], b:&[X;L2]) ->[X;{L1+L2}]{
    let mut arr:[X;{L1+L2}] = empty();
    arr[..L1].copy_from_slice(a.as_slice());
    arr[L1..].copy_from_slice(b.as_slice());
    arr
}
