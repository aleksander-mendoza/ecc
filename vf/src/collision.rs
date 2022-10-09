use std::ops::Mul;
use num_traits::{Float, MulAdd};
use crate::{AffTrans, VectorFieldSub};
use crate::dot_arr::dot0;

pub type Line<F, const DIM: usize> = [[F; DIM]; 2];

/**Finds two points on lines a and b such that the distance between them is minimized. Those points must lie on a third that is
 * perpendicular to both of the lines a and b.*/
pub fn line_to_line<F: Float, const DIM: usize>(a: &Line<F, DIM>, b: &Line<F, DIM>) -> [F; 2] where for<'a> &'a F:Mul<Output=F>{
    let qp = b[0].sub(&a[0]);
    let u = &a[1];
    let v = &b[1];
    if u.eq(v) {
        // similar to https://math.stackexchange.com/questions/1347604/find-3d-distance-between-two-parallel-lines-in-simple-way
        // but we ue cos(theta) to compute s, while t=0 is assumed , thus is becomes just
        // https://en.wikipedia.org/wiki/Vector_projection#Scalar_projection_2
        let s = dot0(u, &qp);
        let t = F::zero();
        [s, t]
    } else {
        // https://math.stackexchange.com/questions/1033419/line-perpendicular-to-two-other-lines-data-sufficiency
        let uu = dot0(u,u);
        let vv = dot0(v,v);
        let uv = dot0(u, v);

        let uqp = dot0(u, &qp);
        let vqp = dot0(v, &qp);
        // s * uu - t * uv =  uqp
        // s * uv - t * vv =  vqp
        //
        // - s * uv + t * uv^2/uu =  - uqp * uv / uu
        // s * uv - t * vv =  vqp
        //
        // t * uv^2/uu - t * vv =  - uqp * uv / uu + vqp
        //
        // t * (uv^2/uu - vv) =  - uqp * uv / uu + vqp
        //
        // t  =  - (uqp * uv / uu + vqp) / (uv^2/uu - vv)
        //
        // t  =  - (uqp * uv + vqp * uu) / (uv^2 - vv * uu)
        //

        let t = -(uqp * uv + vqp * uu) / (uv * uv - vv * uu);
        // s * uu - t * uv =  uqp
        //s = (uqp + t * uv) / uu
        let s = (uqp + t * uv) / uu;
        [s, t]
    }
}

pub fn line_segment_to_line_segment() {}

pub fn cylinder_to_cylinder() {}


/**Vector holding length of x,y,z radii.*/
pub type Ellipsoid = [f32; 3];

pub fn ellipsoid_to_plane(ell1: Ellipsoid, tran1: AffTrans<f32, 3>, ell2: Ellipsoid, tran2: AffTrans<f32, 3>) {}

/** https://matthias-research.github.io/pages/publications/orientedParticles.pdf */
pub fn ellipsoid_to_ellipsoid(ell1: Ellipsoid, tran1: AffTrans<f32, 3>, ell2: Ellipsoid, tran2: AffTrans<f32, 3>) {}