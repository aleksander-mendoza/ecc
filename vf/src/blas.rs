// use blas;
//
// trait Blas: Sized {
//     /**dot product, */
//     unsafe fn dot(x: &[Self], stride_x: i32, y: &[Self], stride_y: i32) -> Self;
//     /**add vectors, y = αx + y*/
//     unsafe fn axpy_(a: Self, x: &[Self], stride_x: i32, y: &mut [Self], stride_y: i32);
// }
//
// impl Blas for f32 {
//     unsafe fn dot(x: &[Self], stride_x: i32, y: &[Self], stride_y: i32) -> Self {
//         blas::sdot(x.len() as i32, x, stride_x, y, stride_y)
//     }
//
//     unsafe fn axpy_(a: Self, x: &[Self], stride_x: i32, y: &mut [Self], stride_y: i32) {
//         blas::saxpy(x.len() as i32, a, x, stride_x, y, stride_y)
//     }
// }
//
// impl Blas for f64 {
//     unsafe fn dot(x: &[Self], stride_x: i32, y: &[Self], stride_y: i32) -> Self {
//         blas::ddot(x.len() as i32, x, stride_x, y, stride_y)
//     }
//     unsafe fn axpy_(a: Self, x: &[Self], stride_x: i32, y: &mut [Self], stride_y: i32) {
//         blas::daxpy(x.len() as i32, a, x, stride_x, y, stride_y)
//     }
// }