import numpy as np

from dm_control import suite
import jaco
import pyglet

# Load one task:
# env = suite.load(domain_name="jaco", task_name="basic")
env = jaco.basic()

# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)

width = 320
height = 240
window = pyglet.window.Window(width=width, height=height, display=None)


# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(-1,
                             1,
                             size=[6])
  time_step = env.step(action)
  # import ipdb; ipdb.set_trace()
  print(time_step.reward, time_step.discount, time_step.observation)
  pixel = env.physics.render(height, width, camera_id=1)
  window.clear()
  window.switch_to()
  window.dispatch_events()
  pyglet.image.ImageData(width, height, 'RGB', pixel.tobytes(), pitch=width * -3).blit(0, 0)
  window.flip()
