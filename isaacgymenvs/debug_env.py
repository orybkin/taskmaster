""" 
Debug headless camera rendering
Author: Oleg 
"""


import imageio
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np


gym = gymapi.acquire_gym()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Config    
sim_params = gymapi.SimParams()
sim_params.physx.solver_type = 1
sim_params.physx.num_threads = 0
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81

cam_props = gymapi.CameraProperties()
cam_props.enable_tensors = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)


# create env
env = gym.create_env(sim, gymapi.Vec3(-2.0, 2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

# add camera
cam_handle = gym.create_camera_sensor(env, cam_props)
gym.set_camera_location(cam_handle, env, gymapi.Vec3(8, 0, 1), gymapi.Vec3(0, 0, 1))

# obtain camera tensor
cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)

# wrap camera tensor in a pytorch tensor
torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)

cube_size = 0.5
cube_options = gymapi.AssetOptions()
cube_options.density = 1000
cube_asset = gym.create_box(sim, *([cube_size] * 3), cube_options)
og_color = gymapi.Vec3(0.6, 0.1, 0.1)
new_color = gymapi.Vec3(0.5, 0.5, 0.5)

cube_start_pose = gymapi.Transform()
cube_start_pose.p = gymapi.Vec3(-1.0, 2.0, 1.0)
cube_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

cube_id = gym.create_actor(env, cube_asset, cube_start_pose, "cube", 0, 0, 0)

cube_mass = None

# prepare tensor access
gym.prepare_sim(sim)

actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(1, -1, 13)
rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(1, -1, 13)
images = []

def reset():
    states = []
    for _ in range(1):
        loc = torch.rand(1, 3, device=device)
        root_state[0, :, :2] = rigid_body_state[0, :, :2] = 2 * (loc[:, :2] - 0.5)
        root_state[0, :, 2] = rigid_body_state[0, :, 2] = 1.5 # force z coordinate to be large
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
        for _ in range(10):
            step()
        refresh()
        states.append(root_state[0, :, :3])
        print('candidates:', root_state[0, :, :3])
    
    sampled_state = states[np.random.randint(len(states))]
    root_state[0, :, :3] = sampled_state
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))    
    

def refresh():
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    
def step():
    gym.simulate(sim)
    
def render():
    gym.fetch_results(sim, True)

    # refresh state data in the tensor
    refresh()
    gym.step_graphics(sim)

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    images.append(torch_cam_tensor.cpu().numpy())
    
    gym.end_access_image_tensors(sim)

def print_info():
    print('root:  ', root_state[0][:,:3])
    print('rigid: ', rigid_body_state[0][:,:3])
        
def run_steps(n):
    for _ in range(n):
        step()
        render()
        print_info()

cube_mass = None

def change_mass(factor):
    global cube_mass
    properties = gym.get_actor_rigid_body_properties(env, cube_id)[0]
    if cube_mass is None: cube_mass = properties.mass
    properties.mass = cube_mass * factor
    gym.set_actor_rigid_body_properties(env, cube_id, [properties], True)
    
def update_pos(x, y, z):
    root_state[0, :, :2] = [x, y, z]
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))    

run_steps(5)

reset()
run_steps(5)

# reset()
# run_steps(5)
    
kwargs = {'macro_block_size': None, 'ffmpeg_params': ['-s','1600x912'] }
imageio.mimsave("render.mp4", images, format='MP4', **kwargs)
