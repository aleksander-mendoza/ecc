use std::ffi::{CString, NulError};
use std::ptr;
use std::string::FromUtf8Error;
use std::time::Instant;
use glutin::{Api, ContextBuilder, ContextCurrentState, ContextWrapper, GlRequest};
use glutin::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};
use glutin::window::{CursorGrabMode, Window, WindowBuilder};
use thiserror::Error;
use gl::types::*;
use glutin::dpi::{LogicalPosition, PhysicalPosition, PhysicalSize};
use glutin::error::ExternalError;
use vf::{DegreesRadians, VectorFieldMulAssign, ArrayCast};
use vf::Dot;

type Pos = [f32; 3];
type Color = [f32; 3];

#[repr(C, packed)]
struct Vertex(Pos, Color);

#[rustfmt::skip]
const VERTICES: [Vertex; 3] = [
    Vertex([-0.5, -0.5, 0.], [1.0, 0.0, 0.0]),
    Vertex([0.5, -0.5, 0.], [0.0, 1.0, 0.0]),
    Vertex([0.0, 0.5, 0.], [0.0, 0.0, 1.0])
];

pub struct Buffer {
    pub id: GLuint,
    target: GLuint,
}

impl Buffer {
    pub fn new(target: GLuint) -> Self {
        let mut id: GLuint = 0;
        unsafe { gl::GenBuffers(1, &mut id) };
        Self { id, target }
    }
    pub fn bind(&self) {
        unsafe { gl::BindBuffer(self.target, self.id) };
    }
    pub fn set_data<D>(&self, data: &[D], usage: GLuint) {
        self.bind();
        unsafe {
            let (_, data_bytes, _) = data.align_to::<u8>();
            gl::BufferData(
                self.target,
                data_bytes.len() as GLsizeiptr,
                data_bytes.as_ptr() as *const _,
                usage,
            );
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, [self.id].as_ptr());
        }
    }
}

pub struct VertexArray {
    pub id: GLuint,
}

impl VertexArray {
    pub fn new() -> Self {
        let mut id: GLuint = 0;
        unsafe { gl::GenVertexArrays(1, &mut id) };
        Self { id }
    }
    pub fn bind(&self) {
        unsafe { gl::BindVertexArray(self.id) };
    }
    pub fn set_attribute<V: Sized>(
        &self,
        attrib_pos: GLuint,
        components: GLint,
        offset: GLint,
    ) {
        self.bind();
        unsafe {
            gl::VertexAttribPointer(
                attrib_pos,
                components,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<V>() as GLint,
                offset as *const _,
            );
            gl::EnableVertexAttribArray(attrib_pos);
        }
    }
}

impl Drop for VertexArray {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, [self.id].as_ptr());
        }
    }
}

#[derive(Debug, Error)]
pub enum ShaderError {
    #[error("Error while compiling shader: {0}")]
    CompilationError(String),
    #[error("Error while linking shaders: {0}")]
    LinkingError(String),
    #[error{"{0}"}]
    Utf8Error(#[from] FromUtf8Error),
    #[error{"{0}"}]
    NulError(#[from] NulError),
}

const VERTEX_SHADER: &str = "
#version 330
in vec3 position;
in vec3 color;
out vec3 vertexColor;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(position, 1.0);
    vertexColor = color;
}
";
const FRAGMENT_SHADER: &str = "
#version 330
out vec4 FragColor;
in vec3 vertexColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
";

pub struct Shader {
    pub id: GLuint,
}

impl Shader {
    pub fn new(source_code: &str, shader_type: GLenum) -> Result<Self, ShaderError> {
        let source_code = CString::new(source_code)?;
        let shader = Self {
            id: unsafe { gl::CreateShader(shader_type) },
        };
        let mut success: GLint = 0;
        unsafe {
            gl::ShaderSource(shader.id, 1, &source_code.as_ptr(), ptr::null());
            gl::CompileShader(shader.id);
            // check for shader compilation errors
            gl::GetShaderiv(shader.id, gl::COMPILE_STATUS, &mut success);
        }
        if success == 1 {
            Ok(shader)
        } else {
            let mut error_log_size: GLint = 0;
            unsafe {
                gl::GetShaderiv(shader.id, gl::INFO_LOG_LENGTH, &mut error_log_size);
            }
            let mut error_log: Vec<u8> = Vec::with_capacity(error_log_size as usize);
            unsafe {
                gl::GetShaderInfoLog(
                    shader.id,
                    error_log_size,
                    &mut error_log_size,
                    error_log.as_mut_ptr() as *mut _,
                );
                error_log.set_len(error_log_size as usize)
            }
            let log = String::from_utf8(error_log)?;
            Err(ShaderError::CompilationError(log))
        }
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.id);
        }
    }
}

pub struct ShaderProgram {
    pub id: GLuint,
}

impl ShaderProgram {
    pub fn apply(&self) {
        unsafe { gl::UseProgram(self.id) };
    }
    pub fn new(shaders: &[Shader]) -> Result<Self, ShaderError> {
        let program = Self {
            id: unsafe { gl::CreateProgram() },
        };

        for shader in shaders {
            unsafe { gl::AttachShader(program.id, shader.id) }
        }

        unsafe { gl::LinkProgram(program.id) };

        let mut success: GLint = 0;
        unsafe { gl::GetProgramiv(program.id, gl::LINK_STATUS, &mut success) };

        if success == 1 {
            Ok(program)
        } else {
            unsafe {
                let mut error_log_size: GLint = 0;
                gl::GetProgramiv(program.id, gl::INFO_LOG_LENGTH, &mut error_log_size);
                let mut error_log: Vec<u8> = Vec::with_capacity(error_log_size as usize);
                gl::GetProgramInfoLog(
                    program.id,
                    error_log_size,
                    &mut error_log_size,
                    error_log.as_mut_ptr() as *mut _,
                );

                error_log.set_len(error_log_size as usize);
                let log = String::from_utf8(error_log)?;
                Err(ShaderError::LinkingError(log))
            }

        }
    }
    pub fn get_attrib_location(&self, attrib: &str) -> Result<GLuint, NulError> {
        let attrib = CString::new(attrib)?;
        Ok(unsafe{gl::GetAttribLocation(self.id, attrib.as_ptr()) as GLuint})
    }
    pub fn get_uniform_location(&self, attrib:&str) -> Result<GLuint, NulError> {
        let attrib = CString::new(attrib)?;
        Ok(unsafe{gl::GetUniformLocation(self.id, attrib.as_ptr()) as GLuint})
    }
}

impl Drop for ShaderProgram {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.id);
        }
    }
}
#[macro_export]
macro_rules! set_attribute {
    ($vbo:ident, $pos:tt, $t:ident :: $field:tt) => {{
        let dummy = core::mem::MaybeUninit::<$t>::uninit();
        let dummy_ptr = dummy.as_ptr();
        let member_ptr = core::ptr::addr_of!((*dummy_ptr).$field);
        const fn size_of_raw<T>(_: *const T) -> usize {
            core::mem::size_of::<T>()
        }
        let member_offset = member_ptr as i32 - dummy_ptr as i32;
        $vbo.set_attribute::<$t>(
            $pos,
            (size_of_raw(member_ptr) / core::mem::size_of::<f32>()) as i32,
            member_offset,
        )
    }};
}

#[derive(Default)]
struct PressedKeys{
    left:bool,
    right:bool,
    up:bool,
    down:bool,
    forward:bool,
    backward:bool,
}
fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Learn OpenGL with Rust");

    let gl_context = ContextBuilder::new()
        .with_gl(GlRequest::Specific(Api::OpenGl, (3, 3)))
        .build_windowed(window, &event_loop)
        .expect("Cannot create windowed context");

    let gl_context = unsafe {
        gl_context
            .make_current()
            .expect("Failed to make context current")
    };
    gl::load_with(|ptr| gl_context.get_proc_address(ptr) as *const _);

    let vertex_shader = Shader::new(VERTEX_SHADER, gl::VERTEX_SHADER).unwrap();
    let fragment_shader = Shader::new(FRAGMENT_SHADER, gl::FRAGMENT_SHADER).unwrap();
    let program = ShaderProgram::new(&[vertex_shader, fragment_shader]).unwrap();
    let vertex_buffer = Buffer::new(gl::ARRAY_BUFFER);
    vertex_buffer.set_data(&VERTICES, gl::STATIC_DRAW);
    let vertex_array = VertexArray::new();
    let pos_attrib = program.get_attrib_location("position").unwrap();
    unsafe{set_attribute!(vertex_array, pos_attrib, Vertex::0)};
    let color_attrib = program.get_attrib_location("color").unwrap();
    unsafe{set_attribute!(vertex_array, color_attrib, Vertex::1)};
    let mut win_size = gl_context.window().inner_size();
    fn make_proj_mat(win_size:&PhysicalSize<u32>)->[[f32;4];4]{
        vf::perspective_proj(90f32.rad(),win_size.width as f32/win_size.height as f32,0.1,10.)
    }
    let mut projection = make_proj_mat(&win_size);
    let mut view:[[f32;4];4] = vf::id();
    let mut model:[[f32;4];4] = vf::id();
    let mut keys = PressedKeys::default();
    vf::translate4d_(&mut view, &[0.,0.,-1.]);
    let mut window_position = gl_context.window().inner_position().unwrap();

    let mvp_loc = program.get_uniform_location("transform").unwrap() as i32;
    let mut grabbed = false;
    let mut i = 0;
    fn center_curson<T:ContextCurrentState>(gl_context:&ContextWrapper<T,Window>,window_position:&PhysicalPosition<i32>, win_size:&PhysicalSize<u32>) {
        gl_context.window().set_cursor_position(LogicalPosition::new(window_position.x as f32 + win_size.width as f32/2.,window_position.y as f32 + win_size.height as f32/2.)).unwrap()
    }
    let mut prev_frame_time = Instant::now();
    let mut delta = 0f32;
    event_loop.run(move |event, _, control_flow| {
        // *control_flow = ControlFlow::;

        match event {
            Event::LoopDestroyed => (),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {  input:KeyboardInput{virtual_keycode:Some(vk),state,..},.. } =>{
                    let s = state == ElementState::Pressed;
                    match vk{
                        VirtualKeyCode::Left | VirtualKeyCode::A => {keys.left = s},
                        VirtualKeyCode::Right | VirtualKeyCode::D => {keys.right = s},
                        VirtualKeyCode::W | VirtualKeyCode::Up => {keys.forward = s},
                        VirtualKeyCode::S | VirtualKeyCode::Down => {keys.backward = s},
                        VirtualKeyCode::LShift  => {keys.down = s},
                        VirtualKeyCode::Space  => {keys.up = s},
                        VirtualKeyCode::Escape => {
                            if s {
                                grabbed = !grabbed;
                                gl_context.window().set_cursor_grab(if grabbed { CursorGrabMode::Confined } else { CursorGrabMode::None });
                                // gl_context.window().set_cursor_visible(!grabbed);
                                if grabbed{
                                    center_curson(&gl_context,&window_position, &win_size);
                                }
                            }
                        }
                        _ => {}
                    }
                },
                WindowEvent::Moved(pos) => {
                    window_position = pos;
                },
                WindowEvent::MouseInput {state, button,..} =>{
                    let s = state == ElementState::Pressed;

                },
                WindowEvent::CursorMoved {position,..} => {
                    println!("{},{}", position.x,position.y);
                    if grabbed{
                        center_curson(&gl_context,&window_position, &win_size);
                    }

                },
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    gl_context.resize(physical_size);
                    win_size = physical_size;
                    projection = make_proj_mat(&win_size);
                },
                _ => (),
            },
            Event::MainEventsCleared => {
                let now = Instant::now();
                delta = now.duration_since(prev_frame_time).as_millis() as f32 / 1000.;
                prev_frame_time = now;

                i+=1;
                let speed = 8f32 * delta;
                let translation_vector = [keys.left as i8-keys.right as i8,keys.down as i8 - keys.up as i8,keys.forward as i8-keys.backward as i8];
                let mut translation_vector = translation_vector.as_scalar::<f32>();
                translation_vector.mul_scalar_(speed);
                vf::translate4d_(&mut view, &translation_vector);
                println!("{:?}", &view);
                let mvp = projection.dot(&view.dot(&model));
                unsafe {
                    gl::ClearColor(0.3, 0.3, 0.3, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT);
                    program.apply();
                    gl::UniformMatrix4fv(mvp_loc, 1, gl::TRUE, mvp.as_ptr() as *const f32);
                    vertex_array.bind();
                    gl::DrawArrays(gl::TRIANGLES, 0, 3);
                }
                gl_context.swap_buffers().unwrap();
                std::thread::sleep()
            }
            _ => (),
        }
    });
}