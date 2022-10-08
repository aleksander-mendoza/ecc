#![feature(option_result_contains)]

mod slice_box;
mod util;

use std::ops::Range;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray6, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyNativeType};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use pyo3::types::PyList;
use vf::soft_wta::*;
use vf::{ArrayCast, conv, VecCast, VectorField, VectorFieldOne};
use vf::tup_arr::{arr2, arr3, slice_as_arr, tup2, tup3, tup4, tup6};
use crate::util::{arrX, py_any_as_numpy};


#[pyfunction]
#[text_signature = "(input_size,stride,kernel)"]
pub fn conv_out_size(input_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let input_size = arrX(py, &input_size, 1, 1, 1)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let out_size = conv::out_size(&input_size, &stride, &kernel);
    Ok(out_size.to_vec())
}

#[pyfunction]
#[text_signature = "(output_size,stride,kernel)"]
pub fn conv_in_size(output_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let output_size = arrX(py, &output_size, 1, 1, 1)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let in_size = conv::in_size(&output_size, &stride, &kernel);
    Ok(in_size.to_vec())
}

#[pyfunction]
#[text_signature = "(output_position,output_size,stride,kernel)"]
pub fn conv_in_range_with_custom_size(output_pos: PyObject, output_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let output_pos = arrX(py, &output_pos, 0, 0, 0)?;
    let output_size = arrX(py, &output_size, 1, 1, 1)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let in_range = conv::in_range_with_custom_size(&output_pos, &output_size, &stride, &kernel);
    Ok((in_range.start.to_vec(), in_range.end.to_vec()))
}

#[pyfunction]
#[text_signature = "(output_position,stride,kernel)"]
pub fn conv_in_range(output_pos: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let output_pos = arrX(py, &output_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let in_range = conv::in_range(&output_pos, &stride, &kernel);
    Ok((in_range.start.to_vec(), in_range.end.to_vec()))
}

#[pyfunction]
#[text_signature = "(input_position,stride,kernel)"]
pub fn conv_out_range(input_pos: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let input_pos = arrX(py, &input_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let out_range = conv::out_range_clipped(&input_pos, &stride, &kernel);
    Ok((out_range.start.to_vec(), out_range.end.to_vec()))
}

#[pyfunction]
#[text_signature = "(input_position,stride,kernel,max_bounds)"]
pub fn conv_out_range_clipped_both_sides(input_pos: PyObject, stride: PyObject, kernel: PyObject, max_bounds: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let input_pos = arrX(py, &input_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let max_bounds = arrX(py, &max_bounds, 0, 0, 0)?;
    let out_range = conv::out_range_clipped_both_sides(&input_pos, &stride, &kernel, &max_bounds);
    Ok((out_range.start.to_vec(), out_range.end.to_vec()))
}

#[pyfunction]
#[text_signature = "(output_position,stride)"]
pub fn conv_in_range_begin(output_pos: PyObject, stride: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let output_pos = arrX(py, &output_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let begin = conv::in_range_begin(&output_pos, &stride);
    Ok(begin.to_vec())
}

#[pyfunction]
#[text_signature = "(input_size,output_size,kernel)"]
pub fn conv_stride(input_size: PyObject, output_size: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let input_size = arrX(py, &input_size, 1, 1, 1)?;
    let output_size = arrX(py, &output_size, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let stride = conv::stride(&input_size, &output_size, &kernel);
    Ok(stride.to_vec())
}

#[pyfunction]
#[text_signature = "(strides,kernels)"]
pub fn conv_compose_array(strides: Vec<PyObject>, kernels: Vec<PyObject>) -> PyResult<(Vec<u32>, Vec<u32>)> {
    assert_eq!(strides.len(), kernels.len());
    let (mut kernel, mut stride) = ([1; 3], [1; 3]);
    let gil = Python::acquire_gil();
    let py = gil.python();
    for (s, k) in strides.into_iter().zip(kernels.into_iter()) {
        let s = arrX(py, &s, 1, 1, 1)?;
        let k = arrX(py, &k, 1, 1, 1)?;
        (stride, kernel) = conv::compose(&stride, &kernel, &s, &k);
    }
    Ok((stride.to_vec(), kernel.to_vec()))
}

#[pyfunction]
#[text_signature = "(stride1,kernel1,stride2,kernel2)"]
pub fn conv_compose(stride1: PyObject, kernel1: PyObject, stride2: PyObject, kernel2: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let stride1 = arrX(py, &stride1, 1, 1, 1)?;
    let kernel1 = arrX(py, &kernel1, 1, 1, 1)?;
    let stride2 = arrX(py, &stride2, 1, 1, 1)?;
    let kernel2 = arrX(py, &kernel2, 1, 1, 1)?;
    let (stride, kernel) = conv::compose(&stride1, &kernel1, &stride2, &kernel2);
    Ok((stride.to_vec(), kernel.to_vec()))
}


#[pyfunction]
#[text_signature = "(v,s)"]
/// u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
pub fn soft_wta_v<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = top_v_slice(unsafe { v.as_slice()? }, unsafe { s.as_slice()? });
    let w = winners.into_pyarray(v.py());
    Ok(w)
}

#[pyfunction]
#[text_signature = "(u,s)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn soft_wta_u<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = top_u_slice(unsafe { u.as_slice()? }, unsafe { s.as_slice()? });
    let w = winners.into_pyarray(u.py());
    Ok(w)
}

#[pyfunction]
#[text_signature = "(u,s)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn multiplicative_soft_wta_u<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = multiplicative_top_u_slice(unsafe { u.as_slice()? }, unsafe { s.as_slice()? });
    let w = winners.into_pyarray(u.py());
    Ok(w)
}

#[pyfunction]
#[text_signature = "(u,s,y)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn soft_wta_u_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(top_u_slice_(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(v,s,y)"]
/// u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
pub fn soft_wta_v_<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(top_v_slice_(unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}


#[pyfunction]
#[text_signature = "(u,s,y)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of u is [height, width, channels, channels],
/// shape of y is [height, width, channels].
pub fn soft_wta_u_conv_<'py>(u: &'py PyArray4<f32>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(top_u_conv_(slice_as_arr(y.shape()),unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(v,s,y)"]
/// v is row-major. Element v[j0,j1,k,j]==1 means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of v is [height, width, channels, channels],
/// shape of y is [height, width, channels].
pub fn soft_wta_v_conv_<'py>(v: &'py PyArray4<bool>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(top_v_conv_(slice_as_arr(y.shape()),unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}


#[pyfunction]
#[text_signature = "(u,s,y)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of u is [channels, channels],
/// shape of y is [height, width, channels].
pub fn soft_wta_u_repeated_conv_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(top_u_repeated_conv_(slice_as_arr(y.shape()),unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(v,s,y)"]
/// v is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of v is [channels, channels],
/// shape of y is [height, width, channels].
pub fn soft_wta_v_repeated_conv_<'py>(v: &'py PyArray2<bool>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(top_v_repeated_conv_(slice_as_arr(y.shape()),unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(u,s,y)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn multiplicative_soft_wta_u_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(multiplicative_top_u_slice_(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(u,s,y)"]
/// Shape of s is [height, width, channels], shape of u is [height, width, channels, channels],
///  shape of y is [height, width, channels].
/// u is row-major. Element u[j0,j1,k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn multiplicative_soft_wta_u_conv_<'py>(u: &'py PyArray4<f32>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(multiplicative_top_u_conv_(slice_as_arr(y.shape()),unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(u,s,y)"]
/// Shape of s is [height, width, channels], shape of u is [channels, channels],
///  shape of y is [height, width, channels].
/// u is row-major. Element u[j0,j1,k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn multiplicative_soft_wta_u_repeated_conv_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(multiplicative_top_u_repeated_conv_(slice_as_arr(y.shape()),unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

type Idx = u32;

///
/// ConvShape(output: (int,int), kernel: (int,int), stride: (int,int), in_channels: int, out_channels: int)
///
#[pyclass]
pub struct ConvShape {
    pub(crate) cs: vf::conv_shape::ConvShape<Idx>,
}

#[pymethods]
impl ConvShape {
    ///[out_height, out_width, out_channels, out_channels]
    #[getter]
    pub fn u_shape(&self) -> (Idx,Idx,Idx,Idx) { tup4(self.cs.u_shape()) }
    ///[out_channels, out_channels]
    #[getter]
    pub fn minicolumn_u_shape(&self) -> (Idx,Idx) { tup2(self.cs.minicolumn_u_shape()) }
    ///[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    #[getter]
    pub fn w_shape(&self) -> (Idx,Idx,Idx,Idx,Idx,Idx) { tup6(self.cs.w_shape()) }
    ///[out_height, out_width, out_channels]
    #[getter]
    pub fn out_shape(&self) -> (Idx,Idx,Idx) { tup3(self.cs.output_shape()) }
    ///[in_height, in_width, in_channels]
    #[getter]
    pub fn in_shape(&self) -> (Idx,Idx,Idx) { tup3(self.cs.input_shape()) }
    ///[kernel_height, kernel_width]
    #[getter]
    pub fn kernel(&self) -> (Idx,Idx) { tup2(self.cs.kernel().clone()) }
    ///[height, width]
    #[getter]
    pub fn stride(&self) -> (Idx,Idx) { tup2(self.cs.stride().clone()) }
    ///[kernel_height, kernel_width, in_channels]
    /// Kernel column is the shape of receptive field of each output neuron. Don't confuse it with
    /// minicolumn which consists of all the output neurons that have the same receptive field.
    #[getter]
    pub fn kernel_column_shape(&self) -> (Idx,Idx,Idx) { tup3(self.cs.kernel_column_shape()) }
    ///[kernel_height, kernel_width, in_channels, out_channels]
    /// This is the shape of weight tensor that constitutes weights of a single minicolumn.
    #[getter]
    pub fn minicolumn_w_shape(&self) -> (Idx,Idx,Idx,Idx) { tup4(self.cs.minicolumn_w_shape()) }
    /// kernel_height * kernel_width
    #[getter]
    pub fn kernel_column_area(&self) -> Idx { self.cs.kernel_column_area() }
    /// kernel_height * kernel_width * in_channels
    #[getter]
    pub fn kernel_column_volume(&self) -> Idx { self.cs.kernel_column_volume() }
    ///[in_height, in_width]
    #[getter]
    pub fn in_grid(&self) -> (Idx,Idx) { tup2(self.cs.in_grid().clone()) }
    #[getter]
    ///[out_height, out_width]
    pub fn out_grid(&self) -> (Idx,Idx) { tup2(self.cs.out_grid().clone()) }
    #[getter]
    pub fn out_width(&self) -> Idx { self.cs.out_width() }
    #[getter]
    pub fn out_height(&self) -> Idx { self.cs.out_height() }
    #[getter]
    pub fn out_channels(&self) -> Idx { self.cs.out_channels() }
    #[getter]
    pub fn in_width(&self) -> Idx { self.cs.in_width() }
    #[getter]
    pub fn in_height(&self) -> Idx { self.cs.in_height() }
    #[getter]
    pub fn in_channels(&self) -> Idx { self.cs.in_channels() }
    #[getter]
    pub fn out_area(&self) -> Idx { self.cs.out_area() }
    #[getter]
    pub fn in_area(&self) -> Idx { self.cs.in_area() }
    #[getter]
    pub fn out_volume(&self) -> Idx { self.cs.out_volume() }
    #[getter]
    pub fn in_volume(&self) -> Idx { self.cs.in_volume() }
    #[text_signature = "(output_pos)"]
    pub fn kernel_offset(&self, output_pos: (Idx,Idx,Idx)) -> (Idx,Idx) { tup2(self.cs.kernel_offset(&arr3(output_pos))) }
    #[text_signature = "(input_pos, output_pos)"]
    pub fn pos_within_kernel(&self, input_pos: (Idx,Idx,Idx), output_pos: (Idx,Idx,Idx)) -> (Idx,Idx,Idx) { tup3(self.cs.pos_within_kernel(&arr3(input_pos), &arr3(output_pos))) }
    #[text_signature = "(input_pos, output_pos)"]
    pub fn idx_within_kernel(&self, input_pos: (Idx,Idx,Idx), output_pos: (Idx,Idx,Idx)) -> Idx { self.cs.idx_within_kernel(&arr3(input_pos), &arr3(output_pos)) }
    #[getter]
    ///[out_height,out_width,out_channels,kernel_height, kernel_width, in_channels]
    pub fn receptive_field_shape(&self)->(Idx,Idx,Idx,Idx,Idx,Idx){
        tup6(self.cs.receptive_field_shape())
    }
    #[getter]
    ///[out_channels,kernel_height, kernel_width, in_channels]
    pub fn minicolumn_receptive_field_shape(&self)->(Idx,Idx,Idx,Idx){
        tup4(self.cs.minicolumn_receptive_field_shape())
    }
    ///((start_y,start_x),(end_y,end_x))
    #[text_signature = "(output_column_pos)"]
    pub fn in_range(&self, output_column_pos: (Idx,Idx)) -> ((Idx,Idx), (Idx,Idx)) {
        let Range { start, end } = self.cs.in_range(&arr2(output_column_pos));
        (tup2(start), tup2(end))
    }
    ///((start_y,start_x),(end_y,end_x))
    #[text_signature = "(input_pos)"]
    pub fn out_range(&self, input_pos: (Idx,Idx)) -> ((Idx,Idx), (Idx,Idx)) {
        let Range { start, end } = self.cs.out_range(&arr2(input_pos));
        (tup2(start), tup2(end))
    }
    #[text_signature = "(conv_tensor)"]
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    pub fn normalize_kernel_columns(&self, conv_tensor: &PyArray6<f32>, norm: usize) {
        assert_eq!(conv_tensor.shape(),self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let rhs = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        self.cs.normalize_kernel_columns(rhs, vf::norm::ln(norm));
    }
    #[text_signature = "(conv_tensor)"]
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    pub fn normalize_minicolumn(&self, conv_tensor: &PyArray4<f32>, norm: usize) {
        assert_eq!(conv_tensor.shape(),self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let rhs = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        self.cs.normalize_minicolumn(rhs, vf::norm::ln(norm));
    }
    #[text_signature = "(input_pos, output_pos)"]
    pub fn idx(&self, input_pos: (Idx,Idx,Idx), output_pos: (Idx,Idx,Idx)) -> Idx { self.cs.idx(&arr3(input_pos), &arr3(output_pos)) }
    #[text_signature = "(lhs_tensor, rhs_conv_tensor, dot_product_output)"]
    /// rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels].
    /// lhs_tensor is a sparse binary vector (list of indices).
    /// dot_product_output is of shape [out_height, out_width, out_channels]
    pub fn sparse_dot<'py>(&self, lhs_tensor: &'py PyArray1<Idx>, rhs_conv_tensor: &'py PyArray6<f32>, dot_product_output: Option<&'py PyArray3<f32>>) -> PyObject {
        assert_eq!(rhs_conv_tensor.shape(),self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let lhs = unsafe{lhs_tensor.as_slice()}.expect("Lhs input tensor is not continuous");
        let rhs = unsafe{rhs_conv_tensor.as_slice()}.expect("Convolutional weights tensor is not continuous");
        let out_shape = self.cs.out_shape().as_scalar::<usize>();
        let out_tensor:&'py PyArray3<f32> = dot_product_output.unwrap_or_else(||PyArray3::zeros(lhs_tensor.py(),out_shape, false));
        assert_eq!(out_tensor.shape(),&out_shape,"Output tensor shape is wrong");
        let out = unsafe{out_tensor.as_slice_mut()}.expect("Output tensor is not continuous");
        self.cs.sparse_dot_slice_(lhs, rhs, out);
        out_tensor.to_object(lhs_tensor.py())
    }
    #[text_signature = "(lhs_tensor, rhs_conv_tensor, dot_product_output)"]
    /// rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels].
    /// lhs_tensor is a sparse binary vector (list of indices).
    /// dot_product_output is of shape [out_height, out_width, out_channels]
    pub fn sparse_dot_repeated<'py>(&self, lhs_tensor: &'py PyArray1<Idx>, rhs_conv_tensor: &'py PyArray4<f32>, dot_product_output: Option<&'py PyArray3<f32>>) -> PyObject {
        assert_eq!(rhs_conv_tensor.shape(),self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let lhs = unsafe{lhs_tensor.as_slice()}.expect("Lhs input tensor is not continuous");
        let rhs = unsafe{rhs_conv_tensor.as_slice()}.expect("Convolutional weights tensor is not continuous");
        let out_shape = self.cs.out_shape().as_scalar::<usize>();
        let out_tensor:&'py PyArray3<f32> = dot_product_output.unwrap_or_else(||PyArray3::zeros(lhs_tensor.py(),out_shape, false));
        assert_eq!(out_tensor.shape(),&out_shape,"Output tensor shape is wrong");
        let out = unsafe{out_tensor.as_slice_mut()}.expect("Output tensor is not continuous");
        self.cs.sparse_dot_repeated_slice_(lhs, rhs, out);
        out_tensor.to_object(lhs_tensor.py())
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    /// input and output are sparse binary vectors (list of indices)
    #[text_signature = "(conv_tensor, epsilon, input, output)"]
    pub fn sparse_mul_assign<'py>(&self, conv_tensor: &'py PyArray6<f32>, epsilon: f32, input:&'py PyArray1<Idx>, output: &'py PyArray1<Idx>) {
        assert_eq!(conv_tensor.shape(),self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe{input.as_slice()}.expect("Input tensor is not continuous");
        let out = unsafe{output.as_slice()}.expect("Output tensor is not continuous");
        let conv = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        self.cs.sparse_mul_assign(conv,epsilon,inp,out)
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    /// input and output are sparse binary vectors (list of indices)
    #[text_signature = "(conv_tensor, epsilon, input, output)"]
    pub fn sparse_mul_assign_repeated<'py>(&self, conv_tensor: &'py PyArray4<f32>, epsilon: f32, input:&'py PyArray1<Idx>, output: &'py PyArray1<Idx>) {
        assert_eq!(conv_tensor.shape(),self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe{input.as_slice()}.expect("Input tensor is not continuous");
        let out = unsafe{output.as_slice()}.expect("Output tensor is not continuous");
        let conv = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        self.cs.sparse_mul_assign_repeated(conv,epsilon,inp,out)
    }
    #[text_signature = "(conv_tensor, epsilon, input, output, biased=False)"]
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    pub fn sparse_increment<'py>(&self, conv_tensor: &'py PyArray6<f32>, epsilon: f32, input:&'py PyArray1<Idx>, output: &'py PyArray1<Idx>, biased:bool) {
        assert_eq!(conv_tensor.shape(),self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe{input.as_slice()}.expect("Input tensor is not continuous");
        let out = unsafe{output.as_slice()}.expect("Output tensor is not continuous");
        let conv = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        if biased{
            self.cs.sparse_biased_increment(conv,epsilon,inp,out)
        }else{
            self.cs.sparse_unbiased_increment(conv,epsilon,inp,out)
        }
    }
    #[text_signature = "(conv_tensor, epsilon, input, output, biased=False)"]
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    pub fn sparse_increment_repeated<'py>(&self, conv_tensor: &'py PyArray4<f32>, epsilon: f32, input:&'py PyArray1<Idx>, output: &'py PyArray1<Idx>, biased:bool) {
        assert_eq!(conv_tensor.shape(),self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe{input.as_slice()}.expect("Input tensor is not continuous");
        let out = unsafe{output.as_slice()}.expect("Output tensor is not continuous");
        let conv = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        if biased{
            self.cs.sparse_biased_increment_repeated(conv,epsilon,inp,out)
        }else{
            self.cs.sparse_unbiased_increment_repeated(conv,epsilon,inp,out)
        }
    }
    #[text_signature = "(conv_tensor, epsilon, s, y, u)"]
    /// u is of shape [out_height, out_width, out_channels, out_channels].
    /// y is a sparse vector of output activations. k and s are of shape [out_height, out_width, out_channels]
    pub fn update_u_as_expected_sk_minus_sj(&self, epsilon:f32, s: &PyArray3<f32>, y: & PyArray1<Idx>, u_weights: & PyArray4<f32>) {
        assert_eq!(u_weights.shape(),self.cs.u_shape().as_scalar::<usize>().as_slice(), "U tensor shape is wrong");
        assert_eq!(s.shape(),self.cs.out_shape().as_scalar::<usize>().as_slice(), "s tensor shape is wrong");
        let s = unsafe{s.as_slice()}.expect("Input tensor is not continuous");
        let y = unsafe{y.as_slice()}.expect("Output tensor is not continuous");
        let u = unsafe{u_weights.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        self.cs.update_u_as_expected_sk_minus_sj(epsilon,s,y,u)
    }
    #[text_signature = "(conv_tensor, epsilon, s, y, u)"]
    /// u is of shape [out_channels, out_channels].
    /// y is a sparse vector of output activations. k and s are of shape [out_height, out_width, out_channels]
    pub fn update_u_as_expected_sk_minus_sj_repeated(&self, epsilon:f32, s: &PyArray3<f32>, y: & PyArray1<Idx>, u_weights: & PyArray2<f32>) {
        assert_eq!(u_weights.shape(),self.cs.minicolumn_u_shape().as_scalar::<usize>().as_slice(), "U tensor shape is wrong");
        assert_eq!(s.shape(),self.cs.out_shape().as_scalar::<usize>().as_slice(), "s tensor shape is wrong");
        let s = unsafe{s.as_slice()}.expect("Input tensor is not continuous");
        let y = unsafe{y.as_slice()}.expect("Output tensor is not continuous");
        let u = unsafe{u_weights.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        self.cs.update_u_as_expected_sk_minus_sj_repeated(epsilon,s,y,u)
    }
    #[text_signature = "(next)"]
    pub fn compose(&self, next: &Self) -> Self {
        Self{cs:self.cs.compose(&next.cs)}
    }
    #[staticmethod]
    #[text_signature = "(shape)"]
    pub fn new_identity(shape: (Idx,Idx,Idx)) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_identity(arr3(shape))}
    }
    #[staticmethod]
    /**This convolution is in fact just a dense linear layer with certain number of inputs and outputs.*/
    #[text_signature = "(input, output)"]
    pub fn new_linear(input: Idx, output: Idx) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_linear(input, output)}
    }
    #[new]
    pub fn new(output: (Idx,Idx), kernel: (Idx,Idx), stride: (Idx,Idx), in_channels: Idx, out_channels: Idx) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new(arr2(output), arr2(kernel), arr2(stride), in_channels, out_channels)}
    }
    // #[staticmethod]
    // pub fn concat<'py>(layers: Vec<&'py ConvShape>) -> Self {
    //     let layers:Vec<vf::conv_shape::ConvShape<Idx>> = layers.iter().map(|cs|cs.cs.clone()).collect();
    //     Self{cs:vf::conv_shape::ConvShape::concat(layers.as_slice())}
    // }
    #[staticmethod]
    #[text_signature = "(input_shape, out_channels, kernel, stride)"]
    pub fn new_in(input_shape: (Idx,Idx,Idx), out_channels: Idx, kernel: (Idx,Idx), stride: (Idx,Idx)) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_in(arr3(input_shape),out_channels, arr2(kernel), arr2(stride))}
    }
    #[staticmethod]
    #[text_signature = "(in_channels, output_shape, kernel, stride)"]
    pub fn new_out(in_channels: Idx, output_shape: (Idx,Idx,Idx), kernel: (Idx,Idx), stride: (Idx,Idx)) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_out(in_channels,arr3(output_shape), arr2(kernel), arr2(stride))}
    }
    #[text_signature = "(new_stride)"]
    pub fn set_stride(&mut self, new_stride: (Idx,Idx)) {
        self.cs.set_stride(arr2(new_stride))
    }
    #[text_signature = "(weights)"]
    ///Input weights are of shape [kernel_height, kernel_width, in_channels, out_channels]. Output is [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    pub fn repeat_minicolumn(&self,weights:&PyArray4<f32>)->PyResult<PyObject>{
        assert_eq!(weights.shape(),self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Weight tensor shape is wrong");
        let inp = unsafe{weights.as_slice()?};
        let out = self.cs.repeat_minicolumn(inp);
        let out = PyArray1::from_vec(weights.py(), out);
        let out = out.reshape(self.cs.w_shape().as_scalar::<usize>())?;
        Ok(out.to_object(weights.py()))
    }
    #[text_signature = "(minicolumn_receptive_field,x,y)"]
    ///minicolumn_receptive_field is of shape [out_channels,kernel_height, kernel_width, in_channels]
    pub fn add_to_receptive_field_repeated(&self, minicolumn_receptive_field:&PyArray4<f32>, x: & PyArray1<Idx>, y:  & PyArray1<Idx>) -> PyResult<()>{
        let minicolumn_receptive_field = unsafe{minicolumn_receptive_field.as_slice_mut()?};
        let x = unsafe{x.as_slice()}.expect("Input tensor is not continuous");
        let y = unsafe{y.as_slice()}.expect("Output tensor is not continuous");
        self.cs.add_to_receptive_field_repeated(minicolumn_receptive_field,x,y);
        Ok(())
    }

}
#[pyproto]
impl PyObjectProtocol for ConvShape {
    fn __repr__(&self) -> String {
        format!("{:?}", &self.cs)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyfunction]
#[text_signature = "()"]
pub fn version<'py>() -> u32 {
    0
}

#[pyfunction]
#[text_signature = "(bools)"]
/// Returns a pair of vectors (indices, offsets). First vector contains indices of all
/// true boolean values within each batch.
/// The second vector contains offsets to the first one. It works just like [[int]] but is flattened.
/// Batches are assumed to be laid out continuously in memory.
pub fn batch_dense_to_sparse(bools: &PyArrayDyn<bool>) ->PyResult<(PyObject, PyObject)>{
    assert!(bools.ndim()>1,"Tensor must have at least 2 dimensions!");
    let b = unsafe{bools.as_slice()?};
    let batch_size = bools.shape()[1..].product();
    let (indices, offsets) = vf::batch_dense_to_sparse::<u32>(batch_size,b);
    let i = PyArray1::<u32>::from_vec(bools.py(),indices);
    let o = PyArray1::<usize>::from_vec(bools.py(),offsets);
    Ok((i.to_object(bools.py()),o.to_object(bools.py())))
}
#[pyfunction]
#[text_signature = "(bools)"]
/// Returns a vector containing indices of all true boolean values
pub fn dense_to_sparse(bools:&PyArrayDyn<bool>)->PyResult<PyObject>{
    let b = unsafe{bools.as_slice()?};
    Ok(PyArray1::<u32>::from_vec(bools.py(),vf::dense_to_sparse(b)).to_object(bools.py()))
}
#[pyfunction]
#[text_signature = "(collector, n, from_inclusive, to_exclusive)"]
/// Returns a vector containing indices of all true boolean values
pub fn rand_set(py:Python, cardinality: usize, from_inclusive: usize, to_exclusive:usize)-> PyObject{
    let mut v = vf::rand_set(cardinality,from_inclusive..to_exclusive);
    PyArray1::from_vec(py,v).to_object(py)
}
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn ecc_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ConvShape>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_range_with_custom_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_range, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_range, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_range_clipped_both_sides, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_range_begin, m)?)?;
    m.add_function(wrap_pyfunction!(conv_stride, m)?)?;
    m.add_function(wrap_pyfunction!(conv_compose_array, m)?)?;
    m.add_function(wrap_pyfunction!(conv_compose, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_u, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_v, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_u_, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_v_, m)?)?;
    m.add_function(wrap_pyfunction!(multiplicative_soft_wta_u_, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_u_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_v_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(multiplicative_soft_wta_u_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_u_repeated_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(soft_wta_v_repeated_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(multiplicative_soft_wta_u_repeated_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(multiplicative_soft_wta_u, m)?)?;
    m.add_function(wrap_pyfunction!(dense_to_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(batch_dense_to_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(rand_set, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat0() -> Result<(), String> {
        Ok(())
    }
}
