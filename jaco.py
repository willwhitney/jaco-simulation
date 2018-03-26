# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Jaco arm domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import os.path

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards

from lxml import etree
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import ipdb

_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(), common.ASSETS



@SUITE.add('benchmarking', 'easy')
def basic(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the Cartpole Balance task."""
  # ipdb.set_trace()
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = JacoReacher(random=random)
  return control.Environment(physics, task, time_limit=time_limit)


def _make_model():
    # os.path.dirname(os.path.realpath(__file__))
  # print(os.path.realpath(__file__))
  # print(os.path.join(os.path.dirname( __file__ ), 'jaco_other.xml'))
  model_path = os.path.join(os.path.dirname( __file__ ), 'jaco_other.xml')
  xml_string = common.read_model(model_path)
  # xml_string = common.read_model('/home/will/code/jaco-simulation/jaco_other.xml')
  # return xml_string
  mjcf = etree.fromstring(xml_string)
# xml_string = _make_model()
# import ipdb; ipdb.set_trace()

  # if n_poles == 1:
  #   return xml_string
  # mjcf = etree.fromstring(xml_string)
  # parent = mjcf.find('./worldbody/body/body')  # Find first pole.
  # # Make chain of poles.
  # for pole_index in xrange(2, n_poles+1):
  #   child = etree.Element('body', name='pole_{}'.format(pole_index),
  #                         pos='0 0 1', childclass='pole')
  #   etree.SubElement(child, 'joint', name='hinge_{}'.format(pole_index))
  #   etree.SubElement(child, 'geom', name='pole_{}'.format(pole_index))
  #   parent.append(child)
  #   parent = child
  # Move plane down.
  # floor = mjcf.find('./worldbody/geom')
  # floor.set('pos', '0 0 {}'.format(1 - n_poles - .05))
  # # Move cameras back.
  # cameras = mjcf.findall('./worldbody/camera')
  # cameras[0].set('pos', '0 {} 1'.format(-1 - 2*n_poles))
  # cameras[1].set('pos', '0 {} 2'.format(-2*n_poles))
  return etree.tostring(mjcf, pretty_print=True)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Acrobot domain."""

  # def horizontal(self):
  #   """Returns horizontal (x) component of body frame z-axes."""
  #   return self.named.data.xmat[['upper_arm', 'lower_arm'], 'xz']
  #
  # def vertical(self):
  #   """Returns vertical (z) component of body frame z-axes."""
  #   return self.named.data.xmat[['upper_arm', 'lower_arm'], 'zz']

  def finger_to_target_distance(self):
    """Returns the distance from the tip to the target."""
    # ipdb.set_trace()
    tip_to_target = (self.named.data.geom_xpos['target'] -
                     self.named.data.geom_xpos['jaco_link_fingertip_1'])
    return np.linalg.norm(tip_to_target)

  def finger_to_target(self):
    """Returns the distance from the tip to the target."""
    # ipdb.set_trace()
    tip_to_target = (self.named.data.geom_xpos['target'] -
                     self.named.data.geom_xpos['jaco_link_fingertip_1'])
    return tip_to_target

  def move_hand(self, position):
    self.named.data.mocap_pos[0] = position

  # def orientations(self):
  #   """Returns the sines and cosines of the pole angles."""
  #   return np.concatenate((self.horizontal(), self.vertical()))


class JacoReacher(base.Task):
  """A Cartpole `Task` to balance the pole.
  State is initialized either close to the target configuration or at a random
  configuration.
  """

  def __init__(self, random=None):
    """Initializes an instance of `Balance`.
    Args:
      swing_up: A `bool`, which if `True` sets the cart to the middle of the
        slider and the pole pointing towards the ground. Otherwise, sets the
        cart to a random position on the slider and the pole to a random
        near-vertical position.
      sparse: A `bool`, whether to return a sparse or a smooth reward.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super(JacoReacher, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.
    Initializes the cart and pole according to `swing_up`, and in both cases
    adds a small random initial velocity to break symmetry.
    Args:
      physics: An instance of `Physics`.
    """
    # nv = physics.model.nv
    # if self._swing_up:
    #   physics.named.data.qpos['slider'] = .01*self.random.randn()
    #   physics.named.data.qpos['hinge_1'] = np.pi + .01*self.random.randn()
    #   physics.named.data.qpos[2:] = .1*self.random.randn(nv - 2)
    # else:
    #   physics.named.data.qpos['slider'] = self.random.uniform(-.1, .1)
    #   physics.named.data.qpos[1:] = self.random.uniform(-.034, .034, nv - 1)
    # physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.finger_to_target()
    obs['velocity'] = physics.velocity()
    return obs



  def get_reward(self, physics):
    """Returns a sparse or a smooth reward, as specified in the constructor."""

    # radii = physics.named.model.geom_size[['target', 'jaco_link_fingertip_1'], 0].sum()
    # return rewards.tolerance(physics.finger_to_target_distance(), (0, radii))
    return -physics.finger_to_target_distance()
