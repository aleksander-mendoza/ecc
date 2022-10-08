use crate::AffTrans;

/**Vector holding length of x,y,z radii.*/
pub type Ellipsoid = [f32;3];

/** https://matthias-research.github.io/pages/publications/orientedParticles.pdf */
pub fn ellipsoid_ellipsoid(ell1:Ellipsoid,tran1:AffTrans<f32,3>,ell2:Ellipsoid,tran2:AffTrans<f32,3>) {

}