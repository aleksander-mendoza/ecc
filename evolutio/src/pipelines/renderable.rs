use render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use render::command_pool::{CommandPool, CommandBuffer};
use render::single_render_pass::SingleRenderPass;
use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use render::swap_chain::SwapchainImageIdx;
use crate::pipelines::player::Player;
use render::specialization_constants::SpecializationConstants;

pub trait RenderResources:Sized{
    type Render:Renderable;
    fn create_descriptors(&self,descriptors:&mut DescriptorsBuilder, foundations:&FoundationInitializer)->Result<(),failure::Error>;
    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors:&DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, failure::Error>;

}
pub trait Renderable:Sized{
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx:SwapchainImageIdx,descriptors:&Descriptors, render_pass:&SingleRenderPass, foundations:&Foundations)->Result<(),failure::Error>;
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations:&Foundations)->Result<(),failure::Error>;
    fn update_uniforms(&mut self, image_idx:SwapchainImageIdx, player:&mut Player);
    fn recreate(&mut self, render_pass: &SingleRenderPass, constants:&SpecializationConstants) -> Result<(), failure::Error>;
}