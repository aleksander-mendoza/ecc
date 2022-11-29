use crate::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::descriptor_layout::DescriptorLayout;
use crate::sampler::Sampler;

use crate::imageview::{ImageView, Color};
use ash::vk::{DescriptorSetLayoutBinding, DescriptorImageInfo};
use crate::device::Device;
use crate::swap_chain::{SwapChain, SwapchainImageIdx};

use crate::host_buffer::HostBuffer;
use ash::vk;
use std::marker::PhantomData;
use crate::buffer_type::{Uniform, Storage, AsStorage, AsDescriptor};
use crate::buffer::{descriptor_info, Buffer};


enum DescriptorUniform {
    Sampler(DescriptorImageInfo),
    Buffer(/*size in bytes*/usize),
    Storage(vk::DescriptorBufferInfo),
}

pub struct DescriptorsBuilder {
    bindings: Vec<DescriptorSetLayoutBinding>,
    descriptors: Vec<DescriptorUniform>,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct UniformBufferBinding<T, const size: usize>{
    binding:usize,
    _p:PhantomData<[T;size]>
}
impl <T, const size: usize> UniformBufferBinding<T,size>{
    fn new(binding:usize)->Self{
        Self{
            binding,
            _p:PhantomData
        }
    }
}


#[derive(Copy, Clone, Eq, PartialEq)]
pub struct StorageBufferBinding<V, T>{
    binding:usize,
    _p:PhantomData<(V,T)>
}
impl <V, T> StorageBufferBinding<V, T>{
    fn new(binding:usize)->Self{
        Self{
            binding,
            _p:PhantomData
        }
    }
}

impl DescriptorsBuilder {
    pub fn new() -> Self {
        Self {
            bindings: vec![],
            descriptors: vec![],
        }
    }

    pub fn sampler(&mut self, sampler: &Sampler, image_view: &ImageView<Color>) {
        self.bindings.push( vk::DescriptorSetLayoutBinding {
            binding: self.descriptors.len() as u32,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
        });
        self.descriptors.push(DescriptorUniform::Sampler(sampler.descriptor_info(image_view)));
    }


    pub fn singleton_uniform_buffer<T: Copy>(&mut self, buffer: &T, stage:vk::ShaderStageFlags) -> UniformBufferBinding<T, 1> {
        self.array_uniform_buffer(std::array::from_ref(buffer), stage)
    }
    pub fn vert_singleton_uniform_buffer<T: Copy>(&mut self, buffer: &T) -> UniformBufferBinding<T, 1> {
        self.singleton_uniform_buffer(buffer, vk::ShaderStageFlags::VERTEX)
    }
    pub fn frag_singleton_uniform_buffer<T: Copy>(&mut self, buffer: &T) -> UniformBufferBinding<T, 1> {
        self.singleton_uniform_buffer(buffer, vk::ShaderStageFlags::FRAGMENT)
    }
    pub fn vert_array_uniform_buffer<T: Copy, const size: usize>(&mut self, buffer: &[T; size]) -> UniformBufferBinding<T, size> {
        self.array_uniform_buffer(buffer,vk::ShaderStageFlags::VERTEX)
    }
    pub fn frag_array_uniform_buffer<T: Copy, const size: usize>(&mut self, buffer: &[T; size]) -> UniformBufferBinding<T, size> {
        self.array_uniform_buffer(buffer,vk::ShaderStageFlags::FRAGMENT)
    }
    pub fn array_uniform_buffer<T: Copy, const size: usize>(&mut self, buffer: &[T; size], stage:vk::ShaderStageFlags) -> UniformBufferBinding<T, size> {
        let new_idx = self.descriptors.len();
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding: new_idx as u32,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size as u32,
            stage_flags: stage,
            p_immutable_samplers: std::ptr::null(),
        });
        self.descriptors.push(DescriptorUniform::Buffer(std::mem::size_of::<[T; size]>()));
        UniformBufferBinding::new(new_idx)
    }
    pub fn vert_storage_buffer<V: Copy, T:AsStorage + AsDescriptor>(&mut self, buffer: &impl Buffer<V, T>) -> StorageBufferBinding<V, T> {
        self.storage_buffer(buffer,vk::ShaderStageFlags::VERTEX)
    }
    pub fn frag_storage_buffer<V: Copy, T:AsStorage + AsDescriptor>(&mut self, buffer: &impl Buffer<V, T>) -> StorageBufferBinding<V, T> {
        self.storage_buffer(buffer,vk::ShaderStageFlags::FRAGMENT)
    }
    pub fn storage_buffer<V: Copy, T:AsStorage + AsDescriptor>(&mut self, buffer: &impl Buffer<V, T>, stage:vk::ShaderStageFlags) -> StorageBufferBinding<V, T> {
        let new_idx = self.descriptors.len();
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding: new_idx as u32,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: stage,
            p_immutable_samplers: std::ptr::null(),
        });
        self.descriptors.push(DescriptorUniform::Storage(descriptor_info(buffer)));
        StorageBufferBinding::new(new_idx)
    }

    pub fn make_layout(self, device: &Device) -> Result<DescriptorsBuilderLocked, vk::Result> {
        let Self { bindings, descriptors } = self;
        DescriptorLayout::new(device, &bindings).map(move |descriptor_layout| DescriptorsBuilderLocked { descriptors, descriptor_layout })
    }
}

pub struct DescriptorsBuilderLocked {
    descriptors: Vec<DescriptorUniform>,
    descriptor_layout: DescriptorLayout,
}

impl DescriptorsBuilderLocked {
    pub fn layout(&self) -> &DescriptorLayout {
        &self.descriptor_layout
    }
    pub fn build(&self, swapchain: &SwapChain) -> Result<Descriptors, failure::Error> {
        let descriptor_pool = DescriptorPool::new(self.layout(), swapchain)?;
        let descriptor_sets = descriptor_pool.create_sets_with_same_layout(self.layout().clone(), swapchain.len())?;
        let uniform_buffers = self.uniform_buffers_per_binding(swapchain)?;
        for (frame, descriptor_set) in descriptor_sets.iter().enumerate() {
            for (binding, descriptor) in self.descriptors.iter().enumerate() {
                unsafe {
                    match descriptor {
                        DescriptorUniform::Sampler(sampler_info) => {
                            descriptor_set.update_sampler_raw(binding as u32, sampler_info);
                        }
                        &DescriptorUniform::Buffer(_size) => {
                            descriptor_set.update_uniform_buffer_raw(binding as u32, &descriptor_info(uniform_buffers[binding].buffer_per_frame[frame].buffer()));
                        }
                        DescriptorUniform::Storage(info) => {
                            descriptor_set.update_storage_buffer_raw(binding as u32, info);
                        }
                    }
                }
            }
        }

        Ok(Descriptors { descriptor_layout: self.layout().clone(), uniform_buffers, descriptor_pool, descriptor_sets })
    }
    fn uniform_buffers_per_binding(&self, swapchain: &SwapChain) -> Result<Vec<UniformBuffers>,vk::Result> {
        self.descriptors.iter().map(|descriptor| UniformBuffers::new(descriptor, swapchain)).collect()
    }
}

pub struct UniformBuffers {
    buffer_per_frame: Vec<HostBuffer<u8, Uniform>>,
}

impl UniformBuffers {
    fn new(descriptor:&DescriptorUniform, swapchain: &SwapChain) -> Result<UniformBuffers, vk::Result> {
        match descriptor {
            DescriptorUniform::Sampler(_) => {
                Ok(Self::sampler())
            }
            &DescriptorUniform::Buffer(size) => {
                Self::buffer(swapchain, size as u64)
            }
            DescriptorUniform::Storage(_) => {
                Self::storage()
            }
        }
    }
    fn sampler() -> Self {
        Self { buffer_per_frame: Vec::new() }
    }
    fn buffer(swapchain: &SwapChain, size: vk::DeviceSize) -> Result<Self,vk::Result> {
        let buffers: Result<Vec<HostBuffer<u8, Uniform>>, vk::Result> = (0..swapchain.len()).into_iter()
            .map(|_| HostBuffer::with_capacity(swapchain.device(), size)).collect();
        Ok(Self { buffer_per_frame: buffers? })
    }
    fn storage() -> Result<Self,vk::Result> {
        Ok(Self { buffer_per_frame: Vec::new() })
    }
}

pub struct Descriptors {
    descriptor_layout: DescriptorLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<DescriptorSet>,
    uniform_buffers: Vec<UniformBuffers>,
}

impl Descriptors {
    pub fn descriptor_layout(&self) -> &DescriptorLayout {
        &self.descriptor_layout
    }
    pub fn descriptor_pool(&self) -> &DescriptorPool {
        &self.descriptor_pool
    }
    pub fn descriptor_set(&self, image_idx:SwapchainImageIdx) -> &DescriptorSet {
        &self.descriptor_sets[image_idx.get_usize()]
    }
    pub fn uniform_as_slice_mut<T,const size:usize>(&mut self, image_idx:SwapchainImageIdx, buffer:UniformBufferBinding<T,size>) -> &mut [T;size]{
        let bytes:&mut[u8] = self.uniform_buffers[buffer.binding].buffer_per_frame[image_idx.get_usize()].as_slice_mut();
        debug_assert_eq!(bytes.len(),std::mem::size_of::<T>()*size);
        let bytes = bytes.as_mut_ptr() as *mut [T;size];
        unsafe{&mut *bytes}
    }
    pub fn uniform_as_slice<T,const size:usize>(&self, image_idx:SwapchainImageIdx, buffer:UniformBufferBinding<T,size>) -> &[T;size]{
        let bytes:&[u8] = self.uniform_buffers[buffer.binding].buffer_per_frame[image_idx.get_usize()].as_slice();
        debug_assert_eq!(bytes.len(),std::mem::size_of::<T>()*size);
        let bytes = bytes.as_ptr() as *const [T;size];
        unsafe{&*bytes}
    }
}