use std::fmt::Debug;
use crate::*;
use std::ops::{Range, Mul, Add, Sub, Div, Rem};
use num_traits::{One, Zero};

pub fn in_range_begin<T: Copy+Mul<Output=T>, const DIM: usize>(out_position: &[T; DIM], stride: &[T; DIM]) -> [T; DIM] {
    (c1(out_position)*c1(stride)).into_arr()
}

/**returns the range of inputs that connect to a specific output neuron*/
pub fn in_range<T: Copy+ Zero+Mul<Output=T> + Add<Output=T>, const DIM: usize>(out_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]>  {
    let from = in_range_begin(out_position, stride);
    let to = add1(c1(&from),c1(kernel_size)).into_arr();
    from..to
}

/**returns the range of inputs that connect to a specific patch of output neuron.
That output patch starts this position specified by this vector*/
pub fn in_range_with_custom_size<T: Copy + Zero +Mul<Output=T> + Add<Output=T> + Div<Output=T>+Sub<Output=T> + One + PartialOrd, const DIM: usize>(out_position: &[T; DIM], output_patch_size: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]>  {
    if all1(gt1(c1(output_patch_size),zeroes1())){
        let from = in_range_begin(out_position, stride);
        let to = ((c1(out_position)+c1(output_patch_size)-ones1::<T,DIM>())*c1(stride)+c1(kernel_size)).into_arr();
        from..to
    } else {
        [T::zero(); DIM]..[T::zero(); DIM]
    }
}

/**returns the range of outputs that connect to a specific input neuron*/
pub fn out_range<T: Copy + Div<Output=T> +Add<Output=T>+Sub<Output=T>+One, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]>  where for<'a> &'a T:Mul<Output=T> + Add<Output=T> + Div<Output=T> + Sub<Output=T>{
    //out_position * stride .. out_position * stride + kernel
    //out_position * stride ..= out_position * stride + kernel - 1
    //
    //in_position_from == out_position * stride
    //in_position_from / stride == out_position
    //round_down(in_position / stride) == out_position_to
    //
    //in_position_to == out_position * stride + kernel - 1
    //(in_position_to +1 - kernel)/stride == out_position
    //round_up((in_position +1 - kernel)/stride) == out_position_from
    //round_down((in_position +1 - kernel + stride - 1)/stride) == out_position_from
    //round_down((in_position - kernel + stride)/stride) == out_position_from
    //
    //(in_position - kernel + stride)/stride ..= in_position / stride
    //(in_position - kernel + stride)/stride .. in_position / stride + 1
    let to = (c1(in_position)/c1(stride)+ones1()).into_arr();
    let from = ((c1(in_position)+c1(stride)-c1(kernel_size))/c1(stride)).into_arr();
    from..to
}

pub fn out_transpose_kernel<T: Copy + Div<Output=T> + Add<Output=T> + One + Sub<Output=T> + Ord, const DIM: usize>(kernel: &[T; DIM], stride: &[T; DIM]) -> [T; DIM] {
    // (in_position - kernel + stride)/stride .. in_position / stride + 1
    //  in_position / stride + 1 - (in_position - kernel + stride)/stride
    //  (in_position- (in_position - kernel + stride))/stride + 1
    //  (kernel - stride)/stride + 1
    debug_assert!(all1(ge1(kernel,stride)));
    ((c1(kernel)-c1(stride))/c1(stride)+ones1()).into_arr()
}

/**returns the range of outputs that connect to a specific input neuron.
output range is clipped to 0, so that you don't get overflow on negative values when dealing with unsigned integers.*/
pub fn out_range_clipped<T: Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + One + Ord, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    let to = (c1(in_position)/c1(stride)+ones1()).into_arr();
    let from = ((max1(c1(in_position)+c1(stride),c1(kernel_size))-c1(kernel_size))/c1(stride)).into_arr();
    from..to
}

pub fn out_range_clipped_both_sides<T: Copy + Div<Output=T> + Add<Output=T> + One + Sub<Output=T> + Ord, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM], max_bounds: &[T; DIM]) -> Range<[T; DIM]> {
    let to = min1(c1(in_position)/c1(stride)+ones1(), c1(max_bounds)).into_arr();
    let from = ((max1(c1(in_position)+c1(stride),c1(kernel_size))-c1(kernel_size))/c1(stride)).into_arr();
    from..to
}

pub fn out_size<T: Debug + Rem<Output=T> + Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Ord + Zero + One, const DIM: usize>(input: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(all1(le1(kernel_size,input)), "Kernel size {:?} is larger than the input shape {:?} ", kernel_size, input);
    let input_sub_kernel = c1(input)-c1(kernel_size);
    assert!(all1(is_zero1( input_sub_kernel%c1(stride))), "Convolution stride {:?} does not evenly divide the input shape {:?}-{:?}={:?} ", stride, input, kernel_size, input_sub_kernel.into_arr());
    (input_sub_kernel/c1(stride)+ones1()).into_arr()
    //(input-kernel)/stride+1 == output
}

pub fn in_size<T: Debug + Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Zero + Ord + One , const DIM: usize>(output: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(all1(gt1(c1(output),zeroes1())), "Output size {:?} contains zero", output);
    ((c1(output)-ones1())*c1(stride)+c1(kernel_size)).into_arr()
    //input == stride*(output-1)+kernel
}

pub fn stride<T: Debug + Rem<Output=T>+ Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Ord + One + Zero, const DIM: usize>(input: &[T; DIM], out_size: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(all1(le1(kernel_size,input)), "Kernel size {:?} is larger than the input shape {:?}", kernel_size, input);
    let input_sub_kernel = c1(input)-c1(kernel_size);
    let out_size_minus_1 = c1(out_size)-ones1();
    assert!(all1(eq1(if1(is_zero1(out_size_minus_1),zeroes1(),input_sub_kernel%out_size_minus_1),zeroes1())), "Output shape {:?}-1 does not evenly divide the input shape {:?}", out_size, input);
    if1(is_zero1(out_size_minus_1),ones1(),input_sub_kernel%out_size_minus_1).into_arr()
    //(input-kernel)/(output-1) == stride
}

pub fn compose<T:Copy +  Div<Output=T> + Add<Output=T> + Sub<Output=T> + Ord + One, const DIM: usize>(self_stride: &[T; DIM], self_kernel: &[T; DIM], next_stride: &[T; DIM], next_kernel: &[T; DIM]) -> ([T; DIM], [T; DIM]) {
    //(A-kernelA)/strideA+1 == B
    //(B-kernelB)/strideB+1 == C
    //((A-kernelA)/strideA+1-kernelB)/strideB+1 == C
    //(A-kernelA+(1-kernelB)*strideA)/(strideA*strideB)+1 == C
    //(A-(kernelA-(1-kernelB)*strideA))/(strideA*strideB)+1 == C
    //(A-(kernelA+(kernelB-1)*strideA))/(strideA*strideB)+1 == C
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^                    composed kernel
    //                                   ^^^^^^^^^^^^^^^ composed stride
    let composed_kernel = (c1(next_kernel)-ones1())*c1(self_stride)+c1(self_kernel);
    let composed_stride = c1(self_stride)*c1(next_stride);
    (composed_stride.into_arr(), composed_kernel.into_arr())
}

// pub fn conv(&[])

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test5() {
        for x in 1..3 {
            for y in 1..4 {
                for sx in 1..2 {
                    for sy in 1..2 {
                        for ix in 1..3 {
                            for iy in 1..4 {
                                let kernel = [x, y];
                                let stride = [x, y];
                                let output_size = [ix, iy];
                                let input_size = in_size(&output_size, &stride, &kernel);
                                assert_eq!(output_size, out_size(&input_size, &stride, &kernel));
                                for ((&expected, &actual), &out) in stride.iter().zip(super::stride(&input_size, &output_size, &kernel).iter()).zip(output_size.iter()) {
                                    if out != 1 {
                                        assert_eq!(expected, actual);
                                    } else {
                                        assert_eq!(1, actual);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test6() {
        for output_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let i = in_range(&[output_idx], &[sx], &[x]);
                    let i_r = i.start[0]..i.end[0];
                    for i in i_r.clone() {
                        let o = out_range(&[i], &[sx], &[x]);
                        let o_r = o.start[0]..o.end[0];
                        assert!(o_r.contains(&output_idx), "o_r={:?}, i_r={:?} output_idx={} sx={} x={}", o_r, i_r, output_idx, sx, x)
                    }
                }
            }
        }
    }

    #[test]
    fn test7() {
        for input_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let o = out_range(&[input_idx], &[sx], &[x]);
                    let o_r = o.start[0]..o.end[0];
                    for o in o_r.clone() {
                        let i = in_range(&[o], &[sx], &[x]);
                        let i_r = i.start[0]..i.end[0];
                        assert!(i_r.contains(&input_idx), "o_r={:?}, i_r={:?} input_idx={} sx={} x={}", o_r, i_r, input_idx, sx, x)
                    }
                }
            }
        }
    }
}