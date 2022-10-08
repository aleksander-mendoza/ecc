use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd;
use std::simd::{f32x16, Simd, SimdElement};
use num_traits::{AsPrimitive, Float, NumAssign, One, Zero};
use crate::init::InitRFoldWithCapacity;
use crate::{VectorField, VectorFieldAdd, VectorFieldAddAssign, VectorFieldAddOwned, VectorFieldInitZero, VectorFieldMul};

pub type Bezier<F: Float, const DIM: usize> = [[F; DIM]];

/**De Casteljau's algorithm. See here
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#General_definition and here
https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
Returns $\sum_{i=0}^{n-1} \binom{n-1}{i} (1 - t)^{n-1-i} t^i P_i$ where P_i is the point at bezier_curve[i].
 If you can't see the LaTeX equation copy-paste it to https://latex.codecogs.com/eqneditor/editor.php*/
pub fn pos<F: NumAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t: F) -> [F; DIM] where usize: AsPrimitive<F> {
    let n = bezier_curve.len();
    /**b_i = (1-t)^{n-1-i}*/
    let b = Vec::init_rfold(
        /*0 \le i < */n,
        /*b_{n-1}=*/F::one(),
        /** b_{i-1} = (1-t)^{n-1-i+1} = (1-t)^{n-1-i}(1-t) = b_i (1-t) */
        |/*b_i=*/b, i| /*b_{i-1=}*/b * (F::one() - t),
    );
    let mut v = [F::zero(); DIM];
    /**c_i = \frac{(n-1)!}{(n-1-i)!} */
    let mut c = 1; // c_0 = 1
    /**i!*/
    let mut i_factorial = 1;
    /**t^i*/
    let mut t_power_i = F::one();
    for i in 0..n {
        /**b_i = (1-t)^{n-1-i}*/
        let b = b[i];
        /**P_i*/
        let p = &bezier_curve[i];
        /**\binom{n-1}{i} = \frac{(n-1)!}{i! (n-1-i)!} = \frac{c_i}{i!}*/
        let n_minus_1_choose_i = c / i_factorial;
        /**\binom{n-1}{i} (1 - t)^{n-1-i} t^i */
        let coefficient = n_minus_1_choose_i.as_() * t_power_i * b;
        v.add_(&p.mul_scalar(coefficient));
        t_power_i *= t;
        i_factorial *= i;
        /*c_{i+1} = \frac{(n-1)!}{(n-1-i-1)!} = \frac{(n-1)!(n-1-i)}{(n-1-i-1)!(n-1-i)} = \frac{(n-1)!(n-1-i)}{(n-1-i)!} = c_i (n-1-i)*/
        c *= (n - 1 - i);
    }
    v
}