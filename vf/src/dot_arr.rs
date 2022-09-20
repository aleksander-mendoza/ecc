use std::ops::{Add, Mul};
use crate::init::empty;
use num_traits::{MulAdd, Zero};


pub fn dot3<T: Mul<Output = T> + Add<Output = T> + Copy + Zero , const X: usize, const Y: usize, const Z: usize, const W: usize>(lhs: &[[[T; Z]; Y]; W], rhs: &[[[T; X]; Z]; W]) -> [[[T; X]; Y]; W] {
    let mut o: [[[T; X]; Y]; W]= empty();
    for w in 0..W {
        for x in 0..X {
            for y in 0..Y {
                o[w][y][x] = (0..Z).fold(T::zero(), |sum, z| sum + lhs[w][y][z] * rhs[w][z][x]);
            }
        }
    }
    o
}

pub fn dot2<T: Mul<Output = T> + Add<Output = T> + Copy + Zero, const X: usize, const Y: usize, const Z: usize>(lhs: &[[T; Z]; Y], rhs: &[[T; X]; Z]) -> [[T; X]; Y] {
    let mut o: [[T; X]; Y] = empty();
    for x in 0..X {
        for y in 0..Y {
            o[y][x] = (0..Z).fold(T::zero(), |sum, z| sum + lhs[y][z] * rhs[z][x]);
        }
    }
    o
}

pub fn dot1<T: Mul<Output = T> + Add<Output = T> + Copy + Zero, const X: usize, const Z: usize>(lhs: &[T; Z], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o: [T; X]  = empty();
    for x in 0..X {
        o[x] = (0..Z).fold(T::zero(), |sum, z| sum + lhs[z] * rhs[z][x]);
    }
    o
}

pub fn dot0<T: Mul<Output = T> + Add<Output = T> + Copy + Zero, const Z: usize>(lhs: &[T; Z], rhs: &[T; Z]) -> T {
    (0..Z).fold(T::zero(), |sum, z| sum + lhs[z] * rhs[z])
}
