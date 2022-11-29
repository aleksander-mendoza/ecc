#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;

use std::time::Instant;
use render::fps::FpsCounter;
use render::input::Input;
use render::vulkan_context::VulkanContext;
use crate::display::Display;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use render::failure;
use render::failure::err_msg;
use crate::shadertoy::ShadertoyUniform;

mod display;
mod shadertoy;

fn main() -> Result<(), failure::Error> {
    run()
}

const SHADERTOY2:&'static str = "
#define t iTime
#define r iResolution.xy

void mainImage( out vec4 fragColor, in vec2 fragCoord ){
	vec3 c;
	float l,z=t;
	for(int i=0;i<3;i++) {
		vec2 uv,p=fragCoord.xy/r;
		uv=p;
		p-=.5;
		p.x*=r.x/r.y;
		z+=.07;
		l=length(p);
		uv+=p/l*(sin(z)+1.)*abs(sin(l*9.-z-z));
		c[i]=.01/length(mod(uv,1.)-.5);
	}
	fragColor=vec4(c/l,t);
}";
const SHADERTOY1:&'static str = "\
precision highp float;


mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c,s,-s,c);
}

const float pi = acos(-1.0);
const float pi2 = pi*2.0;

vec2 pmod(vec2 p, float r) {
    float a = atan(p.x, p.y) + pi/r;
    float n = pi2 / r;
    a = floor(a/n)*n;
    return p*rot(-a);
}

float box( vec3 p, vec3 b ) {
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float ifsBox(vec3 p) {
    for (int i=0; i<5; i++) {
        p = abs(p) - 1.0;
        p.xy *= rot(iTime*0.3);
        p.xz *= rot(iTime*0.1);
    }
    p.xz *= rot(iTime);
    return box(p, vec3(0.4,0.8,0.3));
}

float map(vec3 p, vec3 cPos) {
    vec3 p1 = p;
    p1.x = mod(p1.x-5., 10.) - 5.;
    p1.y = mod(p1.y-5., 10.) - 5.;
    p1.z = mod(p1.z, 16.)-8.;
    p1.xy = pmod(p1.xy, 5.0);
    return ifsBox(p1);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);

    vec3 cPos = vec3(0.0,0.0, -3.0 * iTime);
    // vec3 cPos = vec3(0.3*sin(iTime*0.8), 0.4*cos(iTime*0.3), -6.0 * iTime);
    vec3 cDir = normalize(vec3(0.0, 0.0, -1.0));
    vec3 cUp  = vec3(sin(iTime), 1.0, 0.0);
    vec3 cSide = cross(cDir, cUp);

    vec3 ray = normalize(cSide * p.x + cUp * p.y + cDir);

    // Phantom Mode https://www.shadertoy.com/view/MtScWW by aiekick
    float acc = 0.0;
    float acc2 = 0.0;
    float t = 0.0;
    for (int i = 0; i < 99; i++) {
        vec3 pos = cPos + ray * t;
        float dist = map(pos, cPos);
        dist = max(abs(dist), 0.02);
        float a = exp(-dist*3.0);
        if (mod(length(pos)+24.0*iTime, 30.0) < 3.0) {
            a *= 2.0;
            acc2 += a;
        }
        acc += a;
        t += dist * 0.5;
    }

    vec3 col = vec3(acc * 0.01, acc * 0.011 + acc2*0.002, acc * 0.012+ acc2*0.005);
    fragColor = vec4(col, 1.0 - t * 0.03);
}";
fn run() -> Result<(), failure::Error> {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Shadertoy")
        .with_inner_size(PhysicalSize::new(900, 700))
        .with_resizable(true)
        .build(&event_loop).map_err(err_msg)?;
    // let sdl = sdl2::init().map_err(err_msg)?;
    // let video_subsystem = sdl.video().map_err(err_msg)?;
    // let timer = sdl.timer().map_err(err_msg)?;
    // let window = video_subsystem
    //     .window("∑volut-io", 900, 700)
    //     .vulkan()
    //     .resizable()
    //     .build()?;
    // sdl.mouse().set_relative_mouse_mode(true);
    let mut uniform = ShadertoyUniform::default();

    let vulkan = VulkanContext::new(window)?;
    let mut display = Display::new(vulkan, &uniform, SHADERTOY1)?;
    // let event_pump = sdl.event_pump().map_err(err_msg)?;
    let mut input = Input::new(display.window());
    let mut fps_counter = FpsCounter::new(60);
    input.set_verbose(true);
    display.rerecord_all_graphics_cmd_buffers(0)?;
    let mut frame = 0;
    let time_start = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                // reset input states on new frame
                input.reset()
            }
            Event::MainEventsCleared => {
                // update input state after accumulating event
                fps_counter.update();
                input.finish_poll(display.window()).unwrap();
                if input.quit() {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }

                // render

                uniform.i_frame = frame;
                frame += 1;
                uniform.i_frame_rate = fps_counter.fps();
                uniform.i_time_delta = fps_counter.delta_f32();
                uniform.i_resolution = glm::vec3(input.width() as f32, input.height() as f32, 0.);
                uniform.i_time = fps_counter.ticks().duration_since(time_start).as_secs_f64() as f32;
                // println!("{:?}", uniform);
                if display.render( 0,&mut uniform).unwrap() {
                    display.device().device_wait_idle().unwrap();
                    display.recreate_graphics().unwrap();
                    display.rerecord_all_graphics_cmd_buffers(0).unwrap();
                }
            }
            Event::WindowEvent { event, .. } => input.poll(event),
            Event::LoopDestroyed => {
                display.device().device_wait_idle().unwrap();
            }
            _ => (),
        }
    });
    Ok(())
}
