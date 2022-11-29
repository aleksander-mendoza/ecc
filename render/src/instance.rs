
use crate::constants::APP_INFO;
use crate::platforms::{required_extension_names};
use ash::{vk};
use crate::validation_layer::{populate_debug_messenger_create_info, get_validation_layer_support};

use ash::vk::DebugUtilsMessengerCreateInfoEXT;
use crate::device::{pick_physical_device, Device};
use crate::surface::Surface;
use crate::swap_chain::SwapChain;

use std::sync::Arc;
use ash::extensions::khr::Swapchain;

struct InstanceInner{
    raw: ash::Instance,
    debug: Option<(ash::extensions::ext::DebugUtils, ash::vk::DebugUtilsMessengerEXT)>,
}
#[derive(Clone)]
pub struct Instance {
    inner:Arc<InstanceInner>
}


impl Instance {
    pub fn new(entry: &ash::Entry, window:&winit::window::Window, debug: bool) -> Result<Self, failure::Error> {
        let extension_names = required_extension_names(window,debug);
        let mut extension_features_vec = vec![];
        if debug{
            // extension_features_vec.push(vk::ValidationFeatureEnableEXT::BEST_PRACTICES);
            extension_features_vec.push(vk::ValidationFeatureEnableEXT::DEBUG_PRINTF);
        }

        let mut extension_features = vk::ValidationFeaturesEXT::builder().enabled_validation_features(&extension_features_vec).build();
        let mut debug_builder = DebugUtilsMessengerCreateInfoEXT::builder();
        let mut instance_builder = ash::vk::InstanceCreateInfo::builder()
            .application_info(&APP_INFO)
            .enabled_extension_names(&extension_names)
            .push_next(&mut extension_features);
        if debug {
            debug_builder = populate_debug_messenger_create_info(debug_builder);
            instance_builder = instance_builder.push_next(&mut debug_builder)
                .enabled_layer_names(get_validation_layer_support(entry)?);
        }
        let instance = unsafe { entry.create_instance(&instance_builder, None) }?;
        let debug_utils = if debug {
            let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, &instance);
            let utils_messenger = unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_builder, None) }?;
            Some((debug_utils_loader, utils_messenger))
        } else {
            None
        };
        Ok(Self { inner:Arc::new(InstanceInner{raw: instance, debug: debug_utils })})
    }

    pub fn create_swapchain(&self, device: &Device, surface: &Surface, old:Option<&SwapChain>) -> Result<SwapChain, failure::Error> {
        SwapChain::new(self,device,surface, old)
    }
    pub fn raw(&self) -> &ash::Instance{
        &self.inner.raw
    }

    pub fn pick_physical_device(&self,surface:&Surface) -> Result<ash::vk::PhysicalDevice, failure::Error> {
        pick_physical_device(self.raw(),surface,self.inner.debug.is_some())
    }

    pub fn create_device(&self,entry:&ash::Entry, physical_device:ash::vk::PhysicalDevice) -> Result<Device, failure::Error> {
        Device::new(entry, &self,physical_device,self.inner.debug.is_some())
    }

    pub fn create_surface(&self,entry:&ash::Entry, window: winit::window::Window) -> Result<Surface, failure::Error> {
        Surface::new(entry,self,window)
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        unsafe {
            if let Some((debug, messanger)) = self.debug.take() {
                debug.destroy_debug_utils_messenger(messanger, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}