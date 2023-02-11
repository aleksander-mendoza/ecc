


use render::command_pool::{CommandPool, CommandBuffer};




use crate::pipelines::renderable::{RenderResources, Renderable};
use render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use failure::Error;
use render::single_render_pass::SingleRenderPass;
use render::swap_chain::SwapchainImageIdx;

use crate::pipelines::joint::{Joint, JointResources};
use crate::pipelines::faces::{FacesResources, BlockWorld};




use crate::pipelines::player::Player;
use render::uniform_types::Vec3;


use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::pipelines::physics::{PhysicsResources, Physics};
use crate::pipelines::computable::{Computable, ComputeResources};
use crate::pipelines::bones::{Bones, BonesBuilder, BoneResources};
use render::specialization_constants::SpecializationConstants;
use crate::pipelines::particles::{ParticleResources, Particles};

pub struct GameResources {
    res: JointResources<BoneResources, JointResources<ParticleResources,FacesResources>>,
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
pub struct ThrowUniform {
    position: Vec3,
    velocity: Vec3,
}

impl GameResources {

    pub fn new(cmd_pool: &CommandPool, foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        let particles = ParticleResources::new(cmd_pool, foundations)?;
        let world = FacesResources::new(cmd_pool, foundations)?;
        let bones = BoneResources::new(cmd_pool, foundations)?;

        let res = JointResources::new(bones, JointResources::new(particles,world ));

        Ok(Self { res })
    }
}

impl RenderResources for GameResources {
    type Render = Game;
    fn create_descriptors(&self, descriptors: &mut DescriptorsBuilder, foundations:&FoundationInitializer) -> Result<(), Error> {
        self.res.create_descriptors(descriptors,foundations)
    }

    fn make_renderable(self, cmd_pool: &CommandPool, render_pass: &SingleRenderPass, descriptors: &DescriptorsBuilderLocked, foundations:&Foundations) -> Result<Self::Render, Error> {
        let Self { res } = self;
        let global = res.make_renderable(cmd_pool, render_pass, descriptors, foundations)?;
        Ok(Game {
            global,
        })
    }
}


pub struct Game {
    global: Joint<Bones,Joint<Particles, BlockWorld>>,
}

impl Game {
    pub fn block_world(&self) -> &BlockWorld {
        self.global.b().b()
    }
    pub fn particles(&self) -> &Particles {
        self.global.b().a()
    }
    pub fn block_world_mut(&mut self) -> &mut BlockWorld {
        self.global.b_mut().b_mut()
    }
    pub fn particles_mut(&mut self) -> &mut Particles {
        self.global.b_mut().a_mut()
    }
}

impl Renderable for Game {
    fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors: &Descriptors, render_pass: &SingleRenderPass, foundations:&Foundations) -> Result<(), Error> {
        self.global.record_cmd_buffer(cmd, image_idx, descriptors, render_pass, foundations)
    }

    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations:&Foundations) -> Result<(), Error> {
        self.global.record_compute_cmd_buffer(cmd,foundations)
    }

    fn update_uniforms(&mut self, image_idx: SwapchainImageIdx, player: &mut Player) {
        self.global.update_uniforms(image_idx, player);
    }

    fn recreate(&mut self, render_pass: &SingleRenderPass, constants:&SpecializationConstants) -> Result<(), Error> {
        self.global.recreate(render_pass, constants)
    }
}