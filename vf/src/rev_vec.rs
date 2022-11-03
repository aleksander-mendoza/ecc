use std::alloc;
use std::alloc::Layout;
use std::ops::{Deref, DerefMut, Index};
use std::ptr::NonNull;

pub struct RevVec<T> {
    ptr: NonNull<T>,
    offset: usize,
    cap: usize,
}

impl<T> RevVec<T> {
    pub fn new(capacity: usize) -> Self {
        let layout = Layout::array::<T>(capacity).unwrap();
        let ptr = if capacity == 0 {
            NonNull::dangling()
        } else {
            unsafe {NonNull::new_unchecked( alloc::alloc(layout) as * mut T )}
        };
        Self {
            ptr,
            offset: capacity,
            cap: capacity,
        }
    }
    pub fn push_front(&mut self, t:T){
        let o = self.offset;
        assert!(o>0);
        let o = o-1;
        unsafe{std::ptr::write(self.ptr.as_ptr().add(o),t) }
        self.offset=o; // Can't fail, we'll OOM first
    }
    pub fn pop_front(&mut self)->T{
        let o = self.offset;
        assert!(o<self.cap);
        let t = unsafe{std::ptr::read(self.ptr.as_ptr().add(o)) };
        self.offset=o+1;
        t
    }
    pub fn as_mut_ptr(&mut self)->*mut T{
        unsafe{self.ptr.as_ptr().add(self.offset)}
    }
    pub fn as_ptr(&self)->*const T{
        unsafe{self.ptr.as_ptr().add(self.offset) as *const T}
    }
    pub fn len(&self)->usize{
        self.cap-self.offset
    }
    pub fn as_mut_slice(&mut self)->&mut [T]{
        unsafe{std::slice::from_raw_parts_mut(self.as_mut_ptr(),self.len())}
    }
    pub fn as_slice(&self)->&[T]{
        unsafe{std::slice::from_raw_parts(self.as_ptr(),self.len())}
    }
    pub fn into_vec(self)->Vec<T>{
        assert_eq!(self.offset,0,"Reverse vector can be turned into vector only when fully populated! offset={}, capacity={}",self.offset,self.cap);
        let cap = self.cap;
        let ptr = self.ptr;
        let v = unsafe{Vec::from_raw_parts(ptr.as_ptr(),cap,cap)};
        std::mem::forget(self);
        v
    }
    pub fn into_boxed_slice(self)->Box<[T]>{
        assert_eq!(self.offset,0,"Reverse vector can be turned into boxed slice only when fully populated! offset={}, capacity={}",self.offset,self.cap);
        let cap = self.cap;
        let ptr = self.ptr.as_ptr();
        let ptr = unsafe{std::ptr::slice_from_raw_parts_mut(ptr,cap)};
        let v = unsafe{Box::from_raw(ptr)};
        std::mem::forget(self);
        v
    }
}

impl <T> Deref for RevVec<T>{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl <T> DerefMut for RevVec<T>{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for RevVec<T> {
    fn drop(&mut self) {
        unsafe {
            // use drop for [T]
            // use a raw slice to refer to the elements of the vector as weakest necessary type;
            // could avoid questions of validity in certain cases
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len()))
        }
        let layout = Layout::array::<T>(self.cap).unwrap();
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}
pub trait CollectRev:ExactSizeIterator{
    fn collect_rev(self)->RevVec<Self::Item>;
}
impl <T:ExactSizeIterator> CollectRev for T{
    fn collect_rev(self) -> RevVec<Self::Item> {
        let mut v = RevVec::new(self.len());
        for i in self{
            v.push_front(i);
        }
        v
    }
}