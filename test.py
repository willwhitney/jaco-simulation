import numpy as np

from dm_control import suite
import jaco
import pyglet

import inspect
# Load one task:


LOCAL_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}
suite._DOMAINS = {**suite._DOMAINS, **LOCAL_DOMAINS}
env = suite.load(domain_name="jaco", task_name="basic")
# env = jaco.basic()

# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)

width = 640
height = 480

fullwidth = width * 2
window = pyglet.window.Window(width=fullwidth, height=height, display=None)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

action = np.zeros([6])
time_step = env.step(action)

def move_target_to_hand():
  env.physics.named.model.geom_pos['target'] = env.physics.named.data.xpos['jaco_link_hand']

def move_mocap_to_hand():
  env.physics.named.data.mocap_pos['weld'] = env.physics.named.data.xpos['jaco_link_hand']

def zero_mocap_offset():
  env.physics.named.model.eq_data['weld'].fill(0)

zero_mocap_offset()

# env.physics.named.data.mocap_pos[0] = env.physics.named.data.xpos['jaco_link_hand']
while not time_step.last():
  # action = np.random.uniform(-1, 1, size=[6])
  # env.physics.named.data.mocap_pos[0] = env.physics.named.data.geom_xpos['jaco_hand']


  for i in range(50):
    time_step = env.step(action)
    # print(time_step.reward, time_step.discount, time_step.observation, env.physics.finger_to_target_distance())
    print(time_step.reward, env.physics.finger_to_target_distance())
    pixel1 = env.physics.render(height, width, camera_id=1)
    pixel2 = env.physics.render(height, width, camera_id=2)
    pixel = np.concatenate([pixel1, pixel2], 1)
    window.clear()
    window.switch_to()
    window.dispatch_events()
    pyglet.image.ImageData(fullwidth, height, 'RGB', pixel.tobytes(), pitch=fullwidth * -3).blit(0, 0)
    window.flip()
  import ipdb; ipdb.set_trace()
