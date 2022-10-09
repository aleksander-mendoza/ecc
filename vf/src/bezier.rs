use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::process::Output;
use std::simd;
use std::simd::{f32x16, Simd, SimdElement};
use flo_curves::bezier::chord_length;
use num_traits::{AsPrimitive, Float, MulAdd, MulAddAssign, NumAssign, One, Zero};
use crate::init::InitRFoldWithCapacity;
use crate::{Dist, piecewise_linear, tri_len, VectorField, VectorFieldAdd, VectorFieldMulAssign, VectorFieldAddAssign, VectorFieldAddOwned, VectorFieldInitZero, VectorFieldMul, VectorFieldMulAdd};

pub type Bezier<F: Float, const DIM: usize> = [[F; DIM]];

/**Returns position of point that lies at distance t on the curve (where t is normalized to range between 0 and 1).
It uses Bernstein basis polynomials to efficiently compute the position. See here
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


/**de Casteljau's algorithm using dynamic programming. https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Recursive_definition .

 This function returns the entire triangular table as on [this picture](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/b-sub-chart.jpg).
Table T where T[i]*/
pub fn de_casteljau<F: NumAssign + MulAdd<Output=F> + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t: F) -> Vec<[F; DIM]> {
    let mut out = Vec::with_capacity(tri_len(bezier_curve.len()));
    let mut prev = &bezier_curve[0];
    let d = F::one() - t;
    for next in &bezier_curve[1..] {
        out.push(prev.linear_comb(d, next, t));
        prev = next;
    }
    let mut from = 1;
    let mut to = out.len();
    while from < to {
        for i in from..to {
            let comb = out[i - 1].linear_comb(d, &out[i], t);
            out.push(comb);
        }
        from = to + 1;
        to = out.len();
    }
    out
}

/**same as de_casteljau but t==0.5,  which allows for some optimisations*/
pub fn de_casteljau_in_half<F: NumAssign + MulAdd<Output=F> + MulAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> Vec<[F; DIM]> {
    let mut out = Vec::with_capacity(tri_len(bezier_curve.len()));
    let mut prev = &bezier_curve[0];
    let half = F::one() / (F::one() + F::one());
    for next in &bezier_curve[1..] {
        out.push(prev.add_mul_scalar(next, half));
        prev = next;
    }
    let mut from = 1;
    let mut to = out.len();
    while from < to {
        for i in from..to {
            let mut comb = out[i - 1].add_mul_scalar(&out[i], half);
            out.push(comb);
        }
        from = to + 1;
        to = out.len();
    }
    out
}

/**Straight-line length between endpoints*/
fn cord_length<F: Float + Copy + NumAssign, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> F where for<'a> &'a F: Dist<Output=F> {
    bezier_curve[0].dist(&bezier_curve[bezier_curve.len() - 1])
}

/** Takes table obtained from de_casteljau or de_casteljau_in_half. Produces two curves that subdivide original one.
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html */
pub fn de_casteljau_table_to_sub_curves<F, const DIM: usize>(table: Vec<[F;DIM]>) -> (Vec<[F; DIM]>, Vec<[F; DIM]>){
    table
}

pub fn subdivide<F: NumAssign + MulAdd<Output=F> + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t: F) -> Vec<[F; DIM]> {

}
/**Straight-line length between endpoints*/
pub fn subdivide_in_half<F: NumAssign + MulAdd<Output=F> + MulAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> Vec<[F; DIM]> {
    de_casteljau_in_half(bezier_curve)
}
// ///
// /// Computes the length of a section of a bezier curve
// ///
// fn curve_length<F: NumAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> f64
//     where for<'a> &'a F: Dist<Output=F>
// {
//
//     // This algorithm is described in Graphics Gems V IV.7
//
//     // The MIN_ERROR guards against cases where the length of a section fails to converge for some reason
//     const MIN_ERROR: f64 = 1e-12;
//
//     // Algorithm is recursive, but we use a vec as a stack to avoid overflowing (and to make the number of iterations easy to count)
//     let mut waiting = vec![(section, max_error)];
//     let mut total_length = 0.0;
//
//     while let Some((section, max_error)) = waiting.pop() {
//         // Estimate the error for the length of the curve
//         let polygon_length = piecewise_linear::curve_length(&section);
//         let chord_length = chord_length(&section);
//
//         let error = (polygon_length - chord_length) * (polygon_length - chord_length);
//
//         // If the error is low enough, return the estimated length
//         if error < max_error || max_error <= MIN_ERROR {
//             total_length += (2.0 * chord_length + 2.0 * polygon_length) / 4.0;
//         } else {
//             // Subdivide the curve (each half has half the error tolerance)
//             let left = section.subsection(0.0, 0.5);
//             let right = section.subsection(0.5, 1.0);
//             let subsection_error = max_error / 2.0;
//
//             waiting.push((left, subsection_error));
//             waiting.push((right, subsection_error));
//         }
//     }
//
//     total_length
// }