use ash::vk;
use crate::device::Device;

use std::rc::Rc;

use crate::sampler::Sampler;
use ash::vk::DescriptorPoolSize;
use crate::swap_chain::SwapChain;

struct DescriptorLayoutInner{
    raw:vk::DescriptorSetLayout,
    device: Device,
    bindings_to_layout: Vec<Option<vk::DescriptorSetLayoutBinding>>
}
impl Drop for DescriptorLayoutInner{
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_descriptor_set_layout(self.raw, None) }

    }
}
#[derive(Clone)]
pub struct DescriptorLayout{
    inner:Rc<DescriptorLayoutInner>
}
impl DescriptorLayout{
    pub fn pool_sizes(&self, swapchain:&SwapChain) -> Vec<DescriptorPoolSize> {
        self.manual_pool_sizes(swapchain.len() as u32)
    }
    pub fn manual_pool_sizes(&self, descriptor_count:u32) -> Vec<DescriptorPoolSize> {
        let mut sizes = Vec::new();
        for layout in &self.inner.bindings_to_layout{
            if let Some(layout) = layout{
                sizes.push(vk::DescriptorPoolSize {
                    ty: layout.descriptor_type,
                    descriptor_count,
                });
            }
        }
        sizes
    }
    pub fn layout(&self, binding:u32)->&vk::DescriptorSetLayoutBinding{
        self.inner.bindings_to_layout[binding as usize].as_ref().unwrap()
    }

    pub fn raw(&self)->vk::DescriptorSetLayout{
        self.inner.raw
    }

    pub fn new(device: &Device, layouts:&[vk::DescriptorSetLayoutBinding]) -> Result<Self, ash::vk::Result> {
        let max_binding = layouts.iter().map(|l|l.binding).max().expect("No descriptor set layout bindings provided!") as usize;
        let mut bindings_to_layout = vec![None;max_binding+1];
        for layout in layouts{
            let  prev = &mut bindings_to_layout[layout.binding as usize];
            assert!(prev.is_none());
            prev.insert(layout.clone());
        }
        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layouts);
        unsafe {
            device.inner().create_descriptor_set_layout(&ubo_layout_create_info, None)
        }.map(|raw|Self{inner:Rc::new(DescriptorLayoutInner{raw,bindings_to_layout, device:device.clone()})})
    }
}


