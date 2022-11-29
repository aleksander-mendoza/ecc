use crate::semaphore::Semaphore;
use crate::fence::Fence;
use crate::device::Device;
use ash::vk;


pub struct FramesInFlight{
    image_available_semaphores:Vec<Semaphore>,
    render_finished_semaphores:Vec<Semaphore>,
    in_flight_fences:Vec<Fence>,
    current_frame:usize,
    device:Device
}

impl FramesInFlight{
    pub fn device(&self) -> &Device{
        &self.device
    }
    pub fn new(device:&Device,frames_in_flight:usize)->Result<Self,vk::Result>{
        let mut slf = Self{
            image_available_semaphores: Vec::with_capacity(frames_in_flight),
            render_finished_semaphores: Vec::with_capacity(frames_in_flight),
            in_flight_fences: Vec::with_capacity(frames_in_flight),
            current_frame: 0,
            device:device.clone()
        };
        for _ in 0..frames_in_flight{
            slf.image_available_semaphores.push(Semaphore::new(device)?);
            slf.render_finished_semaphores.push(Semaphore::new(device)?);
            slf.in_flight_fences.push(Fence::new(device,true)?);
        }
        Ok(slf)
    }

    pub fn len(&self)->usize{
        self.in_flight_fences.len()
    }

    pub fn rotate(&mut self){
        self.current_frame = (self.current_frame+1)%self.len();
    }

    pub fn current(&self)->usize{
        self.current_frame
    }

    pub fn current_fence(&self)->&Fence{
        &self.in_flight_fences[self.current_frame]
    }

    pub fn current_image_semaphore(&self)->&Semaphore{
        &self.image_available_semaphores[self.current_frame]
    }

    pub fn current_rendering(&self)->&Semaphore{
        &self.render_finished_semaphores[self.current_frame]
    }

}