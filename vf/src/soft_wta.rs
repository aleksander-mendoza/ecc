use std::cmp::Ordering::{Greater, Less};
use itertools::Itertools;

pub const NULL:u8 = 2;

/**u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column). */
pub fn top_v_slice(v:&[bool], s:&[f32]) ->Vec<bool>{
    assert_eq!(v.len(), s.len()*s.len());
    top_v(|k,j|v[k*s.len()+j],s)
}

/**u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column). */
pub fn top_v_slice_(v:&[bool], s:&[f32], y:&mut [u8]){
    assert_eq!(v.len(), s.len()*s.len());
    top_v_(|k,j|v[k*s.len()+j],s, y)
}

pub fn top_v(v:impl Fn(usize,usize)->bool,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![2;s.len()];
    top_v_(v,s,&mut y);
    unsafe{std::mem::transmute(y)}
}
pub fn top_v_(v:impl Fn(usize,usize)->bool,s:&[f32], y:&mut [u8]){
    while let Some(k) = y.iter().cloned().enumerate().filter(|&(_,o)|o==NULL).map(|(k,_)|k).max_by(|&k,&j|if s[k] < s[j]{Less}else{Greater}){
        debug_assert_eq!(y[k],NULL);
        y[k] = 1;
        for j in 0..s.len(){
            if y[j] == NULL{
                if v(k,j) && s[k] > s[j]{
                    y[j] = 0;
                }
            }
        }
    }
    debug_assert!(!y.contains(&NULL));
}


/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn top_u_slice(u:&[f32], s:&[f32]) ->Vec<bool>{
    assert_eq!(u.len(), s.len()*s.len());
    top_u(|k,j|u[k*s.len()+j],s)
}


/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn top_u_slice_(u:&[f32], s:&[f32],y:&mut [u8]){
    assert_eq!(u.len(), s.len()*s.len());
    top_u_(|k,j|u[k*s.len()+j],s, y)
}

pub fn top_u_(u:impl Fn(usize,usize)->f32,s:&[f32],y:&mut [u8]){
    while let Some(k) = y.iter().cloned().enumerate().filter(|&(_,o)|o==NULL).map(|(k,_)|k).max_by(|&k,&j|if s[k] < s[j]{Less}else{Greater}){
        debug_assert_eq!(y[k],NULL);
        y[k] = 1;
        for j in 0..s.len(){
            if y[j] == NULL{
                if s[j] + u(k,j) < s[k]{
                    y[j] = 0;
                }
            }
        }
    }
    debug_assert!(!y.contains(&NULL));
}
pub fn top_u(u:impl Fn(usize,usize)->f32,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![NULL;s.len()];
    top_u_(u,s,&mut y);
    unsafe{std::mem::transmute(y)}
}



/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn multiplicative_top_u_slice(u:&[f32], s:&[f32]) ->Vec<bool>{
    assert_eq!(u.len(), s.len()*s.len());
    multiplicative_top_u(|k,j|u[k*s.len()+j],s)
}

/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn multiplicative_top_u_slice_(u:&[f32], s:&[f32],y:&mut [u8]){
    assert_eq!(u.len(), s.len()*s.len());
    multiplicative_top_u_(|k,j|u[k*s.len()+j],s, y)
}

pub fn multiplicative_top_u_(u:impl Fn(usize,usize)->f32,s:&[f32],y:&mut [u8]){
    while let Some(k) = y.iter().cloned().enumerate().filter(|&(_,o)|o==NULL).map(|(k,_)|k).max_by(|&k,&j|if s[k] < s[j]{Less}else{Greater}){
        debug_assert_eq!(y[k],NULL);
        y[k] = 1;
        for j in 0..s.len(){
            if y[j] == NULL{
                if s[k] * s[j] > u(k,j) {
                    y[j] = 0;
                }
            }
        }
    }
    debug_assert!(!y.contains(&NULL));
}
pub fn multiplicative_top_u(u:impl Fn(usize,usize)->f32,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![NULL;s.len()];
    multiplicative_top_u_(u,s,&mut y);
    unsafe{std::mem::transmute(y)}
}

#[cfg(test)]
mod tests {
    use crate::init_rand::InitRandWithCapacity;
    use crate::VectorFieldPartialOrd;
    use super::*;


    #[test]
    fn test_real(){
        let l = 20;
        for _ in 0..10{
            let s = Vec::<f32>::rand(l);
            let u = Vec::<f32>::rand(l*l);
            let y = top_u_slice_real(&u,&s);
            assert!(y.contains(&true));
            for (j, ye) in y.iter().cloned().enumerate(){
                if !ye{
                    let mut shunned = false;
                    for k in 0..l{
                        if s[k] - s[j] > 1. - u[k*l+j]{
                            shunned = true;
                            break;
                        }
                    }
                    assert!(shunned);
                }
            }
        }

    }


    #[test]
    fn test_bool(){
        let l = 20;
        for _ in 0..10{
            let s = Vec::<f32>::rand(l);
            let u = Vec::<bool>::rand(l*l);
            let y = top_u_slice_bool(&u,&s);
            assert!(y.contains(&true));
            for (j, ye) in y.iter().cloned().enumerate(){
                if !ye{
                    let mut shunned = false;
                    for k in 0..l{
                        if u[k*l+j]  && s[k] > s[j] {
                            shunned = true;
                            break;
                        }
                    }
                    assert!(shunned);
                }
            }
        }

    }
}