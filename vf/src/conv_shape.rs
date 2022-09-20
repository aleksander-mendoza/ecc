use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::{Step, Sum};
use std::ops::{Add, AddAssign, Div, Mul, Range, Rem, Sub};
use std::process::Output;
use num_traits::{AsPrimitive, MulAdd, Num, One, PrimInt, Zero};
use crate::shape::Shape;
use crate::{conv, vec_range, VectorFieldOne, VectorFieldPartialOrd, VectorFieldSub};
use crate::arr_concat::concat;
use crate::xyzw::{xy3, xy_z3, xyz3, z3};
use crate::from_usize::FromUsize;
use crate::norm::card;

/**[height, width, channels]->[height, width]*/
pub fn grid<X>(arr: &[X; 3]) -> &[X; 2] {
    xy3(arr)
}

/**[height, width, channels]->width*/
pub fn width<X>(arr: &[X; 3]) -> &X {
    &arr[1]
}

/**[height, width, channels]->height*/
pub fn height<X>(arr: &[X; 3]) -> &X {
    &arr[0]
}

/**[height, width, channels]->channels*/
pub fn channels<X>(arr: &[X; 3]) -> &X {
    &arr[2]
}

#[inline]
pub fn idx<Idx: Debug + Mul<Output=Idx> + Add<Output=Idx> + Copy + Ord>(output_idx: Idx, idx_within_kernel_column: Idx, output_volume: Idx) -> Idx {
    debug_assert!(output_idx < output_volume);
    idx_within_kernel_column * output_volume + output_idx
}

#[inline]
pub fn idx_<Idx: Debug + num_traits::PrimInt>(input_pos: &[Idx; 3], kernel_offset: &[Idx; 2], output_idx: Idx, kernel_column: &[Idx; 3], output_volume: Idx) -> Idx {
    let position_within_kernel_column = sub_kernel_offset(input_pos, kernel_offset);
    idx(output_idx, kernel_column.idx(&position_within_kernel_column), output_volume)
}

pub fn sub_kernel_offset<Idx: Debug + Copy + Sub<Output=Idx>>(input_pos: &[Idx; 3], offset: &[Idx; 2]) -> [Idx; 3] {
    xy_z3(xy3(input_pos).sub(offset), *z3(input_pos))
}

#[derive(Clone, Debug)]
pub struct ConvShape<Idx: Debug + PrimInt> {
    /**[in_height, in_width, in_channels]*/
    input_shape: [Idx; 3],
    /**[out_height, out_width, out_channels]*/
    output_shape: [Idx; 3],
    /**[kernel_height, kernel_width]*/
    kernel: [Idx; 2],
    /**[height, width]*/
    stride: [Idx; 2],
}

impl<Idx: Debug + PrimInt> ConvShape<Idx> {
    /**[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn kernel_columns_shape(&self) -> [Idx; 6] {
        let [kernel_height, kernel_width] = self.kernel().clone();
        let [out_height, out_width, out_channels] = self.output_shape();
        [kernel_height, kernel_width, self.in_channels(), out_height, out_width, out_channels]
    }
    /**[out_height, out_width, out_channels]*/
    pub fn out_shape(&self) -> &[Idx; 3] {
        &self.input_shape
    }
    /**[in_height, in_width, in_channels]*/
    pub fn in_shape(&self) -> &[Idx; 3] {
        &self.input_shape
    }
    /**[kernel_height, kernel_width]*/
    pub fn kernel(&self) -> &[Idx; 2] {
        &self.kernel
    }
    /**[height, width]*/
    pub fn stride(&self) -> &[Idx; 2] {
        &self.stride
    }
    /**[out_height, out_width, out_channels]*/
    pub fn output_shape(&self) -> [Idx; 3] {
        self.out_shape().clone()
    }
    /**[in_height, in_width, in_channels]*/
    pub fn input_shape(&self) -> [Idx; 3] {
        self.in_shape().clone()
    }
    /**[kernel_height, kernel_width, in_channels]*/
    pub fn kernel_column_shape(&self) -> [Idx; 3] {
        xy_z3(self.kernel().clone(), self.in_channels())
    }
    pub fn kernel_column_area(&self) -> Idx where Idx : Mul<Output=Idx> + One{
        self.kernel().product()
    }
    pub fn kernel_column_volume(&self) -> Idx  where Idx : Mul<Output=Idx> + One {
        self.kernel_column_area() * self.in_channels()
    }
    /**[in_height, in_width]*/
    pub fn in_grid(&self) -> &[Idx; 2] {
        grid(self.in_shape())
    }
    /**[out_height, out_width]*/
    pub fn out_grid(&self) -> &[Idx; 2] {
        grid(self.out_shape())
    }
    pub fn out_width(&self) -> Idx {
        *width(self.out_shape())
    }
    pub fn out_height(&self) -> Idx {
        *height(self.out_shape())
    }
    pub fn out_channels(&self) -> Idx {
        *channels(self.out_shape())
    }
    pub fn in_width(&self) -> Idx {
        *width(self.in_shape())
    }
    pub fn in_height(&self) -> Idx {
        *height(self.in_shape())
    }
    pub fn in_channels(&self) -> Idx {
        *channels(self.in_shape())
    }
    pub fn out_area(&self) -> Idx where Idx : Mul<Output=Idx> + One {
        self.out_grid().product()
    }
    pub fn in_area(&self) -> Idx where Idx : Mul<Output=Idx> + One {
        self.in_grid().product()
    }
    pub fn out_volume(&self) -> Idx  where Idx : Mul<Output=Idx> + One {
        self.out_shape().product()
    }
    pub fn in_volume(&self) -> Idx where Idx : Mul<Output=Idx> + One {
        self.in_shape().product()
    }
    pub fn kernel_offset(&self, output_pos: &[Idx; 3]) -> [Idx; 2]  where Idx : Mul<Output=Idx>{
        conv::in_range_begin(grid(output_pos), self.stride())
    }
    pub fn pos_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> [Idx; 3]  {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(vec_range::contains(&conv::in_range(grid(output_pos), self.stride(), self.kernel()), grid(input_pos)));
        debug_assert!(vec_range::contains(&conv::out_range_clipped(grid(input_pos), self.stride(), self.kernel()), grid(output_pos)));
        sub_kernel_offset(input_pos, &self.kernel_offset(output_pos))
    }
    pub fn idx_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.kernel_column_shape().idx(&self.pos_within_kernel(input_pos, output_pos))
    }
    pub fn in_range(&self, output_column_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        assert!(output_column_pos.all_lt(self.out_grid()));
        conv::in_range(output_column_pos, self.stride(), self.kernel())
    }
    pub fn out_range(&self, input_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        conv::out_range_clipped_both_sides(input_pos, self.stride(), self.kernel(), self.out_grid())
    }
    pub fn idx(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(vec_range::contains(&conv::in_range(grid(output_pos), self.stride(), self.kernel()), grid(input_pos)));
        debug_assert!(vec_range::contains(&conv::out_range_clipped(grid(input_pos),self.stride(), self.kernel()), grid(output_pos)));
        idx(self.out_shape().idx(output_pos), self.idx_within_kernel(input_pos, output_pos), self.out_volume())
    }
    pub fn sparse_dot_slice<D: AddAssign + Copy>(&self, lhs_tensor: &[Idx], rhs_conv_tensor: &[D], dot_product_output: &mut [D]) where Idx: AsPrimitive<usize> + Step+ Hash{
        self.sparse_dot(lhs_tensor, |output_idx, w_index| dot_product_output[output_idx.as_()] += rhs_conv_tensor[w_index.as_()])
    }
    pub fn sparse_dot(&self, lhs_tensor: &[Idx], mut target: impl FnMut(Idx, Idx)) where Idx: Step + Hash{
        let kernel_column = self.kernel_column_shape();
        let v = self.out_volume();
        let mut used_w = HashSet::new();
        for &input_idx in lhs_tensor {
            let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
            let r = self.out_range(grid(&input_pos));
            vec_range::foreach2d(&r, |output_pos| {
                let kernel_offset = conv::in_range_begin(&output_pos, self.stride());
                for p2 in Idx::zero()..self.out_channels() {
                    let output_pos = xy_z3(output_pos.clone(),p2);
                    let output_idx = self.out_shape().idx(&output_pos);
                    let w_index = idx_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    debug_assert_eq!(w_index, self.idx(&input_pos, &output_pos));
                    debug_assert!(used_w.insert(w_index), "{:?}", w_index);
                    target(output_idx, w_index);
                }
            });
        }
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                    except that this function works with convolutional topology of weights W and the vectors
                    are sparse binary. Here input is X, output is Y, fold generalizes W*/
    pub fn sparse_conjugate<V>(&self, input: &[Idx], output: &[Idx],
                               fold_init: V,
                               mut fold_per_weight_in_kernel_column: impl FnMut(V, Idx) -> V,
                               mut fold_per_kernel_column: impl FnMut(V, Idx) -> V) -> V {
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        let v = self.out_volume();
        let kernel_column = self.kernel_column_shape();
        let mut value = fold_init;
        for &output_idx in output {
            let output_pos = self.out_shape().pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = self.in_range(grid(&output_pos));
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if vec_range::contains(&input_range, grid(input_pos)) {
                    let w_index = idx_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    value = fold_per_weight_in_kernel_column(value, w_index);
                }
            }
            value = fold_per_kernel_column(value, output_idx)
        }
        value
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                        except that this function works with convolutional topology of weights W and the vectors
                        are sparse binary. Here input is X, output is Y, w_slice is W of shape self.kernel_columns_shape()*/
    pub fn sparse_unbiased_increment<D: Copy + Div<Output=D> + AddAssign + FromUsize>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        let w_to_increment: Vec<Idx> = Vec::with_capacity(input.len());
        self.sparse_conjugate(input, output, w_to_increment, |mut w_to_increment, w_index| {
            w_to_increment.push(w_index);
            w_to_increment
        }, |mut w_to_increment, output_idx| {
            let plasticity = epsilon / D::from_usize(w_to_increment.len());
            for w_index in w_to_increment.iter().cloned() {
                w_slice[w_index.as_()] += plasticity;
            }
            w_to_increment.clear();
            w_to_increment
        });
    }
    pub fn sparse_biased_increment<D: Copy + AddAssign>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        self.sparse_conjugate(input, output, (), |(), idx| w_slice[idx.as_()] += epsilon, |_, _| ())
    }

    pub fn compose(&self, next: &Self) -> Self {
        assert_eq!(self.out_shape(), next.in_shape());
        let (kernel, stride) = conv::compose(self.stride(),self.kernel(), next.stride(), next.kernel());
        Self {
            input_shape: self.input_shape(),
            output_shape: next.output_shape(),
            kernel,
            stride,
        }
    }

    pub fn new_identity(shape: [Idx; 3]) -> Self {
        Self {
            input_shape: shape.clone(),
            output_shape: shape,
            kernel: [Idx::one(); 2],
            stride: [Idx::one(); 2],
        }
    }
    /**This convolution is in fact just a dense linear layer with certain number of inputs and outputs.*/
    pub fn new_linear(input: Idx, output: Idx) -> Self {
        Self {
            input_shape: [Idx::one(), Idx::one(), input],
            output_shape: [Idx::one(), Idx::one(), output],
            kernel: [Idx::one(); 2],
            stride: [Idx::one(); 2],
        }
    }
    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx) -> Self {
        Self::new_out(in_channels, xy_z3(output, out_channels), kernel, stride)
    }
    pub fn concat(layers: &[Self]) -> Self where Idx:Sum{
        assert_ne!(layers.len(), 0, "No layers provided!");
        let first_layer = &layers[0];
        let mut out_shape = first_layer.output_shape();
        let in_shape = first_layer.input_shape();
        let kernel = first_layer.kernel.clone();
        let stride = first_layer.stride.clone();
        assert!(layers.iter().all(|a| a.in_shape().all_eq(&in_shape)), "All concatenated layers must have the same input shape!");
        assert!(layers.iter().all(|a| a.out_grid().all_eq(grid(&out_shape))), "All concatenated layers must have the same output width and height!");
        assert!(layers.iter().all(|a| a.stride().all_eq(&stride)), "All concatenated layers must have the same stride!");
        assert!(layers.iter().all(|a| a.kernel().all_eq(&kernel)), "All concatenated layers must have the same kernel!");
        let concatenated_sum: Idx = layers.iter().map(|a| a.out_channels()).sum();
        out_shape[2] = concatenated_sum;
        Self {
            input_shape: in_shape,
            output_shape: out_shape,
            kernel,
            stride,
        }
    }

    pub fn new_in(input_shape: [Idx; 3],
                  out_channels: Idx,
                  kernel: [Idx; 2],
                  stride: [Idx; 2]) -> Self {
        Self {
            input_shape,
            output_shape: xy_z3(conv::out_size(&grid(&input_shape), &stride, &kernel), out_channels),
            kernel,
            stride,
        }
    }
    pub fn new_out(in_channels: Idx,
                   output_shape: [Idx; 3],
                   kernel: [Idx; 2],
                   stride: [Idx; 2]) -> Self {
        Self {
            input_shape: xy_z3(conv::in_size(&grid(&output_shape), &stride, &kernel), in_channels),
            output_shape,
            kernel,
            stride,
        }
    }
    pub fn set_stride(&mut self, new_stride: [Idx; 2]) {
        let input = conv::in_size(self.out_grid(), &new_stride, self.kernel());
        let input = xy_z3(input, self.in_channels());
        self.input_shape = input;
        self.stride = new_stride;
    }


    // fn batch_sum_x<T, O>(&self, input: &[T], output: &[O], f: impl Fn(&T) -> &[Idx] + Send + Sync, of: impl Fn(&O) -> &[Idx] + Send + Sync) -> ConvTensor<f32>{
    //     let mut q: Vec<ConvTensor<f32>> = (0..num_cpus::get()).map(|_| ConvTensor::new(self.clone(),0.)).collect();
    //     parallel_iter_vectors(input,output,&mut q, |i,o,q|{
    //         let i = f(i);
    //         for &o in of(o).iter(){
    //             for &i in i.iter(){
    //                 q.as_slice_mut()[self.idx_within_kernel()]
    //             }
    //         }
    //     });
    //     let mut sum = q.pop().unwrap();
    //     for q in q{
    //         sum.add_assign(&q)
    //     }
    //     sum
    // }
}
