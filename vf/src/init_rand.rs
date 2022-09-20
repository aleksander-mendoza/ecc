use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::random;
use crate::init::InitWithCapacity;

pub  trait InitRandWithCapacity{
    fn rand(capacity:usize)->Self;
}

impl <T> InitRandWithCapacity for Vec<T> where Standard: Distribution<T>{
    fn rand(capacity: usize) -> Self {
        Vec::init_with(capacity, |_|random())
    }
}