#![feature(option_result_contains)]

mod slice_box;
mod util;

use std::ops::Range;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray6};
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyNativeType};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use vf::soft_wta::*;
use vf::{ArrayCast, conv, VectorField};
use vf::tup_arr::{arr2, arr3, tup2, tup3, tup6};
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
#[text_signature = "(u,s)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn soft_wta_u_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(top_u_slice_(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(v,s)"]
/// u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
pub fn soft_wta_v_<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(top_v_slice_(unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
}

#[pyfunction]
#[text_signature = "(u,s)"]
/// u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column).
pub fn multiplicative_soft_wta_u_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(multiplicative_top_u_slice_(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }))
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
    ///[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    #[getter]
    pub fn kernel_columns_shape(&self) -> (Idx,Idx,Idx,Idx,Idx,Idx) { tup6(self.cs.kernel_columns_shape()) }
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
    ///[out_height, out_width, out_channels]
    #[getter]
    pub fn kernel_column_shape(&self) -> (Idx,Idx,Idx) { tup3(self.cs.kernel_column_shape()) }
    #[getter]
    pub fn kernel_column_area(&self) -> Idx { self.cs.kernel_column_area() }
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
    #[text_signature = "(input_pos, output_pos)"]
    pub fn idx(&self, input_pos: (Idx,Idx,Idx), output_pos: (Idx,Idx,Idx)) -> Idx { self.cs.idx(&arr3(input_pos), &arr3(output_pos)) }
    #[text_signature = "(lhs_tensor, rhs_conv_tensor, dot_product_output)"]
    pub fn sparse_dot<'py>(&self, lhs_tensor: &'py PyArray1<Idx>, rhs_conv_tensor: &'py PyArray6<f32>, dot_product_output: Option<&'py PyArray3<f32>>) -> PyObject {
        assert_eq!(rhs_conv_tensor.shape(),self.cs.kernel_columns_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let lhs = unsafe{lhs_tensor.as_slice()}.expect("Lhs input tensor is not continuous");
        let rhs = unsafe{rhs_conv_tensor.as_slice()}.expect("Convolutional weights tensor is not continuous");
        let out_shape = self.cs.out_shape().as_scalar::<usize>();
        let out_tensor:&'py PyArray3<f32> = dot_product_output.unwrap_or_else(||PyArray3::new(lhs_tensor.py(),out_shape, false));
        assert_eq!(out_tensor.shape(),&out_shape,"Output tensor shape is wrong");
        let out = unsafe{out_tensor.as_slice_mut()}.expect("Output tensor is not continuous");
        self.cs.sparse_dot_slice(lhs, rhs, out);
        out_tensor.to_object(lhs_tensor.py())
    }
    #[text_signature = "(conv_tensor, epsilon, input, output, biased=False)"]
    pub fn sparse_increment<'py>(&self, conv_tensor: &'py PyArray6<f32>, epsilon: f32, input:&'py PyArray1<Idx>, output: &'py PyArray1<Idx>, biased:bool) {
        assert_eq!(conv_tensor.shape(),self.cs.kernel_columns_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe{input.as_slice()}.expect("Input tensor is not continuous");
        let out = unsafe{output.as_slice()}.expect("Output tensor is not continuous");
        let conv = unsafe{conv_tensor.as_slice_mut()}.expect("Convolutional weights tensor is not continuous");
        if biased{
            self.cs.sparse_biased_increment(conv,epsilon,inp,out)
        }else{
            self.cs.sparse_unbiased_increment(conv,epsilon,inp,out)
        }
    }
    pub fn compose(&self, next: &Self) -> Self {
        Self{cs:self.cs.compose(&next.cs)}
    }
    #[staticmethod]
    #[text_signature = "(shape:(int,int,int))"]
    pub fn new_identity(shape: (Idx,Idx,Idx)) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_identity(arr3(shape))}
    }
    #[staticmethod]
    /**This convolution is in fact just a dense linear layer with certain number of inputs and outputs.*/
    #[text_signature = "(input:int, output:int)"]
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
    pub fn new_in(input_shape: (Idx,Idx,Idx), out_channels: Idx, kernel: (Idx,Idx), stride: (Idx,Idx)) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_in(arr3(input_shape),out_channels, arr2(kernel), arr2(stride))}
    }
    #[staticmethod]
    pub fn new_out(in_channels: Idx, output_shape: (Idx,Idx,Idx), kernel: (Idx,Idx), stride: (Idx,Idx)) -> Self {
        Self{cs:vf::conv_shape::ConvShape::new_out(in_channels,arr3(output_shape), arr2(kernel), arr2(stride))}
    }
    pub fn set_stride(&mut self, new_stride: (Idx,Idx)) {
        self.cs.set_stride(arr2(new_stride))
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn ecc_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ConvShape>()?;
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
    m.add_function(wrap_pyfunction!(multiplicative_soft_wta_u, m)?)?;
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
