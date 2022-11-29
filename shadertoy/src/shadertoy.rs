use render::descriptors::{DescriptorsBuilder, DescriptorsBuilderLocked, Descriptors};
use render::single_render_pass::SingleRenderPass;
use render::command_pool::{CommandPool, CommandBuffer};
use render::shader_module::{ShaderModule, Fragment, Vertex};

use render::submitter::Submitter;
use render::texture::{StageTexture, Dim2D, TextureView};

use render::swap_chain::SwapchainImageIdx;
use render::imageview::Color;


use render::buffer::Buffer;
use render::descriptor_layout::DescriptorLayout;
use render::device::Device;
use render::failure::Error;
use render::{failure, shaderc};
use render::ash::vk;
use render::pipeline::{Pipeline, PipelineBuilder};
use render::shaderc::Compiler;
use render::specialization_constants::SpecializationConstants;


#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[repr(C, align(16))]
pub struct ShadertoyUniform {
    /**mouse pixel coords. xy: current (if MLB down), zw: click*/
    pub i_mouse: glm::Vec4,
    /**(year, month, day, time in seconds)*/
    pub i_date: glm::Vec4,
    /**shader playback time (in seconds)*/
    pub i_time: f32,
    /**render time (in seconds)*/
    pub i_time_delta: f32,
    /**shader frame rate*/
    pub i_frame_rate: f32,
    /**shader playback frame*/
    pub i_frame: i32,
    /**viewport resolution (in pixels)*/
    pub i_resolution: glm::Vec3,
    padding:u32,
    /**channel resolution (in pixels)*/
    pub i_channel_resolution: [glm::Vec4; 4],
    /**channel playback time (in seconds)*/
    pub i_channel_time: [glm::Vec4; 4],
}

const PREFIX: &'static str = "
#version 450 core
layout(binding = 0) uniform UniformBufferObject {
    vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
    vec4      iDate;                 // (year, month, day, time in seconds)
    float     iTime;                 // shader playback time (in seconds)
    float     iTimeDelta;            // render time (in seconds)
    float     iFrameRate;            // shader frame rate
    int       iFrame;                // shader playback frame
    vec3      iResolution;           // viewport resolution (in pixels)
    vec3      iChannelResolution[4]; // channel resolution (in pixels)
    float     iChannelTime[4];       // channel playback time (in seconds)
};
//uniform samplerXX iChannel0.3;          // input channel. XX = 2D/Cube

layout (location = 0) out vec4 fragColor;

layout (location = 0) in vec2 fragCoord; // the input variable from the vertex shader (same name and same type)

";

const SUFFIX: &'static str = "

void main()
{
    mainImage(fragColor, (fragCoord+vec2(0.5,0.5))*iResolution.xy);
    // if(fragCoord.x > 0){
    //     fragColor = vec4(iResolution.x / 1000.,0,0,1);  // vec4(1,0,0,1);
    // }else{
    //     fragColor = vec4(sin(iTime+fragCoord.y*3.1415926)/2.+0.5,0,0,1);  // vec4(1,0,0,1);
    // }

}
";


pub struct ShadertoyBuilder {
    frag:Vec<ShaderModule<Fragment>>,
    vert:ShaderModule<Vertex>,
    compiler:Compiler,
}
impl ShadertoyBuilder{
    pub fn new(device:&Device)->Self{
        let compiler = shaderc::Compiler::new().unwrap();
        let vert = ShaderModule::new(include_glsl!("assets/shaders/shadertoy.vert") as &[u32], device).unwrap();
        Self{
            compiler,
            vert,
            frag: vec![]
        }
    }
    pub fn add(&mut self, cmd_pool: &CommandPool, shadertoy_source_code:&str) -> Result<(), failure::Error> {

        let mut frag_src = PREFIX.to_string()+shadertoy_source_code+SUFFIX;
        // let mut options = shaderc::CompileOptions::new().unwrap();
        // options.add_macro_definition("EP", Some("main"));
        println!("{}", frag_src);
        let binary_result = self.compiler.compile_into_spirv(
            &frag_src, shaderc::ShaderKind::Fragment,
            "shadertoy.frag", "main", None).unwrap();

        assert_eq!(Some(&0x07230203), binary_result.as_binary().first());
        // let text_result = compiler.compile_into_spirv_assembly(
        //     source, shaderc::ShaderKind::Fragment,
        //     "shader.glsl", "main", Some(&options)).unwrap();
        let frag = ShaderModule::new(binary_result.as_binary(), cmd_pool.device())?;
        self.frag.push(frag);
        Ok(())
    }

    pub fn build(&mut self, render_pass: &SingleRenderPass, descriptors: &DescriptorLayout, spec_constants:&SpecializationConstants) -> Result<Vec<Shadertoy>,Error> {
        let Self {  frag, vert,.. } = self;
        let pipelines:Result<Vec<Shadertoy>,Error> = frag.iter().map(|f|{
            let mut pipeline_builder = PipelineBuilder::new();
            let pipeline = pipeline_builder.descriptor_layout(descriptors.clone())
                .fragment_shader("main", f.clone())
                .vertex_shader("main", vert.clone()) // clone() is reference-counted
                .depth_test(false)
                .cull_face(vk::CullModeFlags::BACK)
                .front_face_clockwise(false)
                .color_blend_attachment_states_disabled()
                .reset_scissors()
                .scissors(render_pass.swapchain().render_area())
                .reset_viewports()
                .viewports(render_pass.swapchain().viewport())
                .build(render_pass, spec_constants);
            pipeline.map(|pipeline|Shadertoy{pipeline})
        }).collect();
        pipelines
    }
}



pub struct Shadertoy {
    pipeline: Pipeline,
}

impl Shadertoy {
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }
    pub fn record_cmd_buffer(&self, cmd: &mut CommandBuffer, image_idx: SwapchainImageIdx, descriptors: &Descriptors) {
        cmd
            .bind_pipeline(self.pipeline())
            .uniform(self.pipeline(), descriptors.descriptor_set(image_idx))
            .draw(6,1,0,0);
    }
}
