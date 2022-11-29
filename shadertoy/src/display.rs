use render::device::{Device, QUEUE_IDX_GRAPHICS, QUEUE_IDX_COMPUTE, QUEUE_IDX_AMBIENT_COMPUTE};
use render::command_pool::{CommandPool, CommandBuffer};
use render::swap_chain::{SwapChain, SwapchainImageIdx};
use render::vulkan_context::VulkanContext;
use failure::Error;
use render::single_render_pass::SingleRenderPass;
use render::descriptors::{Descriptors, DescriptorsBuilder, UniformBufferBinding, DescriptorsBuilderLocked};
use std::sync::Arc;
use render::ash::vk;
use render::failure;
use render::fence::Fence;
use render::specialization_constants::SpecializationConstants;
use crate::shadertoy::{Shadertoy, ShadertoyBuilder};
use crate::ShadertoyUniform;


pub struct Display {
    graphics_command_buffers: Vec<CommandBuffer>,
    graphics_pipeline: Vec<Shadertoy>,
    render_pass: SingleRenderPass,
    descriptors: Descriptors,
    descriptors_builder: DescriptorsBuilderLocked,
    graphics_cmd_pool: CommandPool,
    uniforms_binding: UniformBufferBinding<ShadertoyUniform, 1>,
    shadertoy:ShadertoyBuilder,
    spec_const:SpecializationConstants,
    vulkan: VulkanContext, // SAFETY: !!! this should be dropped at the very end !!!
}

impl Display {
    const CLEAR_VALUES: [vk::ClearValue; 2] = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [145. / 256., 239. / 256., 2553. / 256., 1.0],
        },
    }, vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.,
            stencil: 0,
        },
    }];

    pub fn graphics_pipeline(&self) -> &[Shadertoy] {
        &self.graphics_pipeline
    }
    pub fn graphics_pipeline_mut(&mut self) -> &mut [Shadertoy] {
        &mut self.graphics_pipeline
    }
    pub fn new(vulkan: VulkanContext, uniform: &ShadertoyUniform, shadertoy_src_code: &str) -> Result<Self, failure::Error> {
        let render_pass = vulkan.create_single_render_pass(None)?;
        let graphics_cmd_pool = CommandPool::new(vulkan.device(), QUEUE_IDX_GRAPHICS, true)?;
        let mut descriptors_builder = DescriptorsBuilder::new();
        let uniforms_binding = descriptors_builder.frag_singleton_uniform_buffer(uniform);
        let spec_const = SpecializationConstants::new();
        let mut shadertoy = ShadertoyBuilder::new(vulkan.device());
        shadertoy.add(&graphics_cmd_pool, shadertoy_src_code);
        let descriptors_builder = descriptors_builder.make_layout(graphics_cmd_pool.device())?;
        let graphics_pipeline = shadertoy.build(&render_pass, descriptors_builder.layout(), &spec_const)?;
        let descriptors = descriptors_builder.build(render_pass.swapchain())?;
        let graphics_command_buffers = graphics_cmd_pool.create_command_buffers((graphics_pipeline.len() * render_pass.framebuffers_len()) as u32)?;
        let display = Self {
            spec_const,
            shadertoy,
            descriptors_builder,
            descriptors,
            graphics_command_buffers,
            graphics_pipeline,
            render_pass,
            graphics_cmd_pool,
            vulkan,
            uniforms_binding,
        };
        Ok(display)
    }
    pub fn graphics_cmd_pool(&self) -> &CommandPool {
        &self.graphics_cmd_pool
    }
    pub fn device(&self) -> &Device {
        self.vulkan.device()
    }
    pub fn window(&self) -> &winit::window::Window {
        self.vulkan.window()
    }
    pub fn destroy(self) -> VulkanContext {
        let Self { vulkan, .. } = self;
        vulkan
    }
    pub fn rerecord_all_graphics_cmd_buffers(&mut self, shadertoy_idx: usize) -> Result<(), Error> {
        Ok(for image_idx in self.swapchain().iter_images() {
            self.record_graphics_cmd_buffer(shadertoy_idx,image_idx)?
        })
    }
    pub fn record_graphics_cmd_buffer(&mut self, shadertoy_idx: usize, image_idx: SwapchainImageIdx) -> Result<(), Error> {
        let Self { graphics_command_buffers: command_buffers, graphics_pipeline, render_pass, descriptors, .. } = self;
        assert!(shadertoy_idx < graphics_pipeline.len());
        let command_buffer = &mut command_buffers[shadertoy_idx * render_pass.framebuffers_len() + image_idx.get_usize()];
        command_buffer.reset()?
            .begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
        let graphics_pipeline = &graphics_pipeline[shadertoy_idx];
        command_buffer
            .render_pass(render_pass, render_pass.framebuffer(image_idx), render_pass.swapchain().render_area(), &Self::CLEAR_VALUES);
        graphics_pipeline.record_cmd_buffer(command_buffer, image_idx, descriptors);
        command_buffer
            .end_render_pass()
            .end()?;
        Ok(())
    }
    pub fn command_buffer(&self, shadertoy_idx: usize, image_idx: SwapchainImageIdx) -> &CommandBuffer {
        &self.graphics_command_buffers[shadertoy_idx * self.render_pass.framebuffers_len() + image_idx.get_usize()]
    }
    pub fn command_buffer_mut(&mut self, shadertoy_idx: usize, image_idx: SwapchainImageIdx) -> &mut CommandBuffer {
        &mut self.graphics_command_buffers[shadertoy_idx * self.render_pass.framebuffers_len() +image_idx.get_usize()]
    }
    pub fn swapchain(&self) -> &SwapChain {
        self.render_pass.swapchain()
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain().extent()
    }
    pub fn recreate_graphics(&mut self) -> Result<(), failure::Error> {
        self.render_pass = self.vulkan.create_single_render_pass(Some(self.render_pass.swapchain()))?;
        self.graphics_pipeline =  self.shadertoy.build(&self.render_pass, self.descriptors_builder.layout(), &self.spec_const)?;
        self.descriptors = self.descriptors_builder.build(self.render_pass.swapchain())?;
        let missing_buffers = self.swapchain().len() - self.graphics_command_buffers.len();
        if 0 < missing_buffers {
            self.graphics_command_buffers.append(&mut self.graphics_cmd_pool().create_command_buffers(missing_buffers as u32)?)
        }
        Ok(())
    }
    pub fn render(&mut self, shadertoy_idx: usize, uniform: &mut ShadertoyUniform) -> Result<bool, failure::Error> {
        let Self { graphics_command_buffers: command_buffers, graphics_pipeline, render_pass, vulkan, descriptors, uniforms_binding, .. } = self;
        let fence = vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = render_pass.swapchain().acquire_next_image(None, Some(image_available), None)?;
        let render_finished = vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        let command_buffer = &mut command_buffers[shadertoy_idx * render_pass.framebuffers_len() + image_idx.get_usize()];
        descriptors.uniform_as_slice_mut(image_idx, *uniforms_binding).copy_from_slice(std::slice::from_ref(uniform));

        command_buffer.submit(&[(image_available, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)],
                              std::slice::from_ref(render_finished),
                              Some(fence))?;
        let result = render_pass.swapchain().present(std::slice::from_ref(render_finished), image_idx);
        let is_resized = match result {
            Ok(is_suboptimal) => is_suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(vk::Result::SUBOPTIMAL_KHR) => true,
            err => err?
        };
        vulkan.frames_in_flight_mut().rotate();
        Ok(is_suboptimal || is_resized)
    }
}