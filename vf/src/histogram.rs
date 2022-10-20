use crate::init::{uninit_boxed_slice, uninit_boxed_static_slice, zeroed_boxed_slice, zeroed_boxed_static_slice};
use crate::{VectorFieldMul, VectorFieldOne, VectorFieldZero};


pub fn histogram(image: &[u8], stride: usize) -> Box<[u32; 256]> {
    let mut slice = zeroed_boxed_static_slice();
    image.iter().step_by(stride).cloned().for_each(|pixel| slice[pixel as usize] += 1);
    slice
}

/**`image` shape is `[height,width,channels]`*/
pub fn histograms(image: &[u8], channels: usize) -> Box<[[u32; 256]]> {
    let mut slice =  zeroed_boxed_slice::<[u32; 256]>(channels);
    for channel in 0..channels{
        let subslice = &mut slice[channel];
        image[channel..].iter().step_by(channels).cloned().for_each(|pixel| subslice[pixel as usize] += 1);
    }
    slice
}

pub fn _normalize(histogram: Box<[u32; 256]>) -> Box<[f32; 256]> {
    let sum = histogram.sum();
    let sum = sum as f32;
    let sum_inv = 1. / sum;
    let u32_ptr = histogram.as_ptr();
    for i in 0..256 {
        unsafe {
            let offset_ptr = u32_ptr.offset(i);
            let f32_ptr = offset_ptr as *mut f32;
            f32_ptr.write(offset_ptr.read() as f32 * sum_inv);
        }
    }
    unsafe { std::mem::transmute(histogram) }
}

pub fn normalize(histogram: Box<[u32; 256]>) -> Box<[f32; 256]> {
    let sum = histogram.sum();
    let sum = sum as f32;
    let sum_inv = 1. / sum;
    let mut box_f32 = uninit_boxed_static_slice();
    for i in 0..256 {
        box_f32[i] = histogram[i] as f32 * sum_inv;
    }
    box_f32
}

pub fn match_histogram(source: &[u8], src_stride: usize, reference: &[u8], ref_stride: usize, output: &mut [u8], out_stride: usize) {
    let hist_ref = _normalize(histogram(reference, ref_stride));
    match_precomputed_histogram(source, src_stride, hist_ref.as_slice(), output, out_stride)
}

pub fn match_precomputed_histogram(source: &[u8], src_stride: usize, hist_ref: &[f32], output: &mut [u8], out_stride: usize) {
    let hist_src = histogram(source, src_stride);
    match_2precomputed_histogram(source, src_stride, hist_src.as_slice(), &hist_ref, output, out_stride)
}

pub fn match_2precomputed_histogram(source: &[u8], src_stride: usize, hist_src: &[u32], hist_ref: &[f32], output: &mut [u8], out_stride: usize) {
    debug_assert_eq!(hist_ref.len(), 256);
    debug_assert_eq!(hist_src.len(), 256);

    let sum_src = hist_src.sum();

    let mut i_ref = 0;
    let mut stack: Vec<(/*reference value to replace source value*/u8, /*how much source value to replace*/usize)> = Vec::new();
    /**for each source pixel value stores a slice (offset,end) into stack that tells us how to replace that value*/
    let mut stack_offsets = zeroed_boxed_static_slice::<(usize, usize), 256>();
    let mut stack_offset = 0;
    let mut popped_src = 0i32;
    for i_src in 0..256 {
        if popped_src <= 0 {
            let to_pop = hist_src[i_src] as i32;
            let to_replace = to_pop.min(-popped_src);
            if to_replace > 0 {
                stack.push(((i_ref - 1) as u8, to_replace as usize));//we should replace `to_replace` pixels of value `i_src` with value `i_ref`
            }
            popped_src += to_pop;
        }
        while popped_src > 0 {
            if i_ref >= 256 {
                stack.push((255u8, popped_src as usize));
                popped_src = 0;
                break;
            }
            let popped_ref = hist_ref[i_ref];
            let corresponding_src = (popped_ref * sum_src as f32) as i32; // it is done in this way for better numerical stability
            let replaced_src = corresponding_src.min(popped_src);
            popped_src -= corresponding_src;
            if replaced_src > 0 {
                stack.push((i_ref as u8, replaced_src as usize));//we should replace `replaced_src` pixels of value `i_src` with value `i_ref`
            }
            i_ref += 1;
        }

        debug_assert_eq!(stack[stack_offset..].iter().map(|&(_, s)| s).sum::<usize>(), hist_src[i_src] as usize);
        stack_offsets[i_src] = (stack_offset, stack.len()); // stack tells us how many pixels of value `i_src` should be replaced into various other values of `i_ref`
        stack_offset = stack.len();
    }
    // debug_assert!(true);
    // If at this point i_ref < 256, that can only be due to floating-point imprecision. (which is unlikely as we use f64)
    // A few pixels might be improperly replaced but that's fine. Nobody will notice.

    'outer: for (src_i, out_i) in source.iter().step_by(src_stride).cloned().zip(output.iter_mut().step_by(out_stride)) {
        let (mut offset, end) = stack_offsets[src_i as usize];
        while offset < end {
            let (ref_i, to_replace) = stack[offset];
            if to_replace > 0 {
                *out_i = ref_i;
                stack[offset].1 = to_replace - 1;
                continue 'outer;
            } else {
                offset += 1;
                stack_offsets[src_i as usize].0 = offset;
            }
        }
        *out_i = src_i; // welp... this should rarely happen. Only possible due to FP imprecision.
    }
}


/**shape == [height, width, channels]*/
pub fn match_images(source: &[u8], src_shape: &[usize; 3], reference: &[u8], ref_shape: &[usize; 3]) -> Box<[u8]> {
    assert_eq!(src_shape[2], ref_shape[2]);
    let channels = src_shape[2];
    let len = src_shape[0] * src_shape[1];
    let mut out = uninit_boxed_slice::<u8>(len * channels);
    for channel in 0..channels {
        match_histogram(&source[channel..], channels, &reference[channel..], channels, &mut out[channel..], channels);
    }
    out
}

/**shape == [height, width, channels], hist_src:[channels,256], hist_ref:[channels, 256]*/
pub fn match_2precomputed_images(source: &[u8], src_shape: &[usize; 3], hist_src: &[u32], hist_ref: &[f32]) -> Box<[u8]> {
    let channels = src_shape[2];
    let len = src_shape[0] * src_shape[1];
    let mut out = uninit_boxed_slice::<u8>(len * channels);
    for channel in 0..channels {
        let hist_offset = channel*256;
        match_2precomputed_histogram(&source[channel..], channels, &hist_src[hist_offset..hist_offset+256],&hist_ref[hist_offset..hist_offset+256], &mut out[channel..], channels);
    }
    out
}
/**shape == [height, width, channels], references:[batch,channels,256]*/
pub fn match_best_images(source: &[u8], src_shape: &[usize; 3], batch:usize, references: &[f32]) -> (Box<[u8]>,usize,f32) {
    assert_eq!(source.len(), src_shape.product());
    let channels = src_shape[2];
    assert_eq!(references.len(), batch*channels*256);
    let hist_src = histograms(source,channels);
    let channels256 = channels*256;
    let mut min_square_diff = f32::INFINITY;
    let mut best_ref_idx = 0;
    for ref_idx in 0..batch {
        let ref_offset = ref_idx*channels256;
        /**`ref_hists` shape is `[channels, 256]`*/
        let ref_hists = &references[ref_offset..ref_offset+channels256];
        let mut square_diff = 0.;
        for channel in 0..channels {
            let offset = channel*256;
            let hist_src = &hist_src[channel];
            let hist_ref = &ref_hists[offset..offset+256];
            let s_inv = 1. / hist_src.sum() as f32;
            fn sq(a:f32)->f32{
                a*a
            }
            fn f(a:&u32)->f32{*a as f32}
            let channel_square_diff = hist_src.iter().map(f).zip(hist_ref.iter()).map(|(s,r)|sq(s*s_inv - r)).sum::<f32>();
            square_diff += channel_square_diff;
        }
        if square_diff < min_square_diff{
            min_square_diff = square_diff;
            best_ref_idx = ref_idx;
        }

    }
    let ref_offset = best_ref_idx*channels256;
    let best_ref_hists = &references[ref_offset..ref_offset+channels256];
    (match_2precomputed_images(source,src_shape,hist_src.flatten(),best_ref_hists), best_ref_idx, min_square_diff)
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;


    #[test]
    fn test6() {
        let s = vec![0, 1, 2];
        let r = vec![3, 4, 5];
        let o = match_images(&s, &[s.len(), 1, 1], &r, &[r.len(), 1, 1]);
        assert_eq!(o.as_ref(), &[3, 4, 5]);
    }

    #[test]
    fn test5() {
        let s = vec![0, 1, 2, 2, 5, 7, 3, 4, 6];
        let r = vec![3, 4, 5];
        let o = match_images(&s, &[s.len(), 1, 1], &r, &[r.len(), 1, 1]);
        assert_eq!(o.as_ref(), &[3, 3, 3, 4, 5, 5, 4, 4, 5]);
    }
}