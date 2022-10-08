use std::ops::{Add, AddAssign, MulAssign, Neg, Sub};
use std::process::Output;
use num_traits::Num;
use crate::mat_arr::{mat2_add_column, mat3_add_column, mat3x2_add_row, mat4x3_add_row, mul_row_wise_};
use crate::VectorFieldAddAssign;

pub type Translation<S, const DIM: usize> = [S; DIM];
pub type Scaling<S, const DIM: usize> = [S; DIM];
/**When DIM=2, it's an array [[S;2];1] holding one normal vector of length 2, which determines where the X axis lies.
The Y axis can then be obtained by rotating X axis 90 degrees clockwise.
 When DIM=3 it's an array [[S;3];2] holding two normal vector of length 2, which determine where the X and Y axes lie.
The Z axis can then be obtained by taking cross product of X and Y.
 And so on for DIM>3*/
pub type AlignmentAxis<S, const DIM: usize> = [[S; DIM]; { DIM - 1 }];
pub type EulerRotation<S, const DIM: usize> = [S; DIM];
pub type Quaternion<S> = [S; 4];


/**clockwise rotation*/
pub fn rot90deg_cw<S: Neg<Output=S>>(v: [S; 2]) -> [S; 2] {
    let [x, y] = v;
    [y, -x]
}

/**counter clockwise rotation*/
pub fn rot90deg_ccw<S: Neg<Output=S>>(v: [S; 2]) -> [S; 2] {
    let [x, y] = v;
    [-y, x]
}

/**clockwise rotation in radians*/
pub fn rot2d(radians: f32, v: [f32; 2]) -> [f32; 2] {
    let s = radians.sin();
    let c = radians.cos();
    let [x, y] = v;
    [c * x - s * x, s * x + c * y]
}

/**cross product of two vectors*/
pub fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}


/**Given a normal vector of X axis, produces a 2x2 rotation matrix R such that [1,0] M = X_axis
 and [0, 1] M = Y_axis*/
pub fn rot_to_mat2d(x_axis: [f32; 2]) -> [[f32; 2]; 2] {
    let y_axis = rot90deg_cw(x_axis);
    [x_axis, y_axis]
}

/**Given a 2 normal vectors, one of X axis and other of Y , produces a 3x3 rotation matrix R such that [1,0,0] M = X_axis
 , [0, 1, 0] M = Y_axis and [0, 0, 1] M = Z_axis*/
pub fn rot_to_mat3d(xy_axes: [[f32; 3]; 2]) -> [[f32; 3]; 3] {
    let [x_axis, y_axis] = xy_axes;
    let z_axis = cross(&x_axis, &y_axis);
    [x_axis, y_axis, z_axis]
}

pub trait AffineTransformation<S, const DIM: usize>: Clone {
    fn compose_(&mut self, other: &Self) -> &mut Self;
    fn compose(&self, other: &Self) -> Self {
        let mut s = self.clone();
        s.compose_(other);
        s
    }
    fn inverse_(&mut self) -> &mut Self;
    fn inverse(&self) -> Self {
        let mut s = self.clone();
        s.inverse_();
        s
    }
    fn scale_(&mut self, scaling: &Scaling<S, DIM>) -> &mut Self;
    fn scale(&self, scaling: &Scaling<S, DIM>) -> Self {
        let mut s = self.clone();
        s.scale_(scaling);
        s
    }
    fn rotate_(&mut self, rot: &EulerRotation<S, DIM>) -> &mut Self;
    fn rotate(&self, rot: &EulerRotation<S, DIM>) -> Self {
        let mut s = self.clone();
        s.rotate_(rot);
        s
    }
    fn translate_(&mut self, translation: &Translation<S, DIM>) -> &mut Self;
    fn translate(&self, translation: &Translation<S, DIM>) -> Self {
        let mut s = self.clone();
        s.translate_(translation);
        s
    }
}

/**A linear transformation resulting from first applying scaling, then rotation to align with axis, then translation*/
#[derive(Clone, Debug)]
pub struct AffTrans<S, const DIM: usize> where [(); { DIM - 1 }]: Sized {
    /**This holds both scaling and rotation information. Each vector's direction represents rotation axis,
             and it's length represents scaling. The last axis must be perpendicular to all other, which means
             that there are only DIM - 1 degrees of freedom. Therefore we do not store the last vector here.
             It can be computer as needed. Once computed, we obtain an orthogonal axis.*/
    axis: AlignmentAxis<S, DIM>,
    translation: Translation<S, DIM>,
}

impl<S: Copy+MulAssign+AddAssign, const DIM: usize> AffineTransformation<S, DIM> for AffTrans<S, DIM> where [(); { DIM - 1 }]: Sized {
    fn compose_(&mut self, other: &Self) -> &mut Self {
        todo!()
    }

    fn inverse_(&mut self) -> &mut Self {
        todo!()
    }

    fn inverse(&self) -> Self {
        todo!()
    }

    fn scale_(&mut self, scaling: &Scaling<S, DIM>) -> &mut Self {
        mul_row_wise_(&mut self.axis, scaling);
        self
    }
    fn rotate_(&mut self, rot: &EulerRotation<S, DIM>) -> &mut Self {
        // self.axis
        self
    }

    fn translate_(&mut self, translation: &Translation<S, DIM>) -> &mut Self {
        self.translation.add_(translation);
        self
    }
}

impl AffTrans<f32, 2> {
    pub fn mat3(&self) -> [[f32; 3]; 3] {
        let mut mat = rot_to_mat2d(self.axis[0]);
        let mat = mat2_add_column(mat, self.translation);
        mat3x2_add_row(mat, [0., 0., 1.])
    }
}

impl AffTrans<f32, 3> {
    pub fn mat4(&self) -> [[f32; 4]; 4] {
        let mut mat = rot_to_mat3d(self.axis);
        let mat = mat3_add_column(mat, self.translation);
        mat4x3_add_row(mat, [0., 0., 0., 1.])
    }
}