use ash::vk;
use crate::device::Device;

use crate::imageview::{ImageView, Color};

use std::sync::Arc;

struct SamplerInner {
    device: Device,
    raw:vk::Sampler
}
#[derive(Clone)]
pub struct Sampler {
    inner:Arc<SamplerInner>
}

impl Sampler {
    pub fn  device(&self) -> &Device{
        &self.inner.device
    }
    pub fn raw(&self)->vk::Sampler{
        self.inner.raw
    }
    pub fn new(device: &Device, filter:vk::Filter, normalize_coordinates:bool) -> Result<Self, ash::vk::Result> {
        let sampler_create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .min_lod(0.0)
            .max_lod(0.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(!normalize_coordinates);

        unsafe {
            device.inner().create_sampler(&sampler_create_info, None)
        }.map(|raw|Self{inner:Arc::new(SamplerInner{raw,device:device.clone()})})
    }

    pub fn descriptor_info(&self, imageview:&ImageView<Color>) ->vk::DescriptorImageInfo{
        vk::DescriptorImageInfo {
            sampler: self.raw(),
            image_view: imageview.raw(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }
    }
}

impl Drop for SamplerInner{
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_sampler(self.raw, None) }
    }
}