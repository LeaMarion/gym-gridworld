import gym
from gym import error, spaces, utils
import numpy as np
from typing import List, Tuple
import cv2
from gym.utils import seeding
import matplotlib as plt

from gym_subgridworld.utils.a_star_path_finding import AStar

class GridWorld2DEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, hardmode = False, random_grid=False, valid_grid=False, valid_path=False,  start = 'ring', **kwargs):
    """
    This environment is a N x M gridworld with walls.
    The agent perceives the world w/o walls and its position in it.
    It does not see the reward, which is fixed to a given position in the
    same plane as the agent's starting position. The agents goal is to find
    the reward within the minimum number of steps. An episode ends if the
    reward has been obtained or the maximum number of steps is exceeded.
    By default there is no restriction to the number of steps.
    At each reset, the agent is moved to a random position in the grid and
    the reward is placed at a specified position of the same plane.

    NOTE:

    You can make the environment harder by setting harmode to True. Then,
    the observation will be an image. By default, the observation is an
    N + M list of mostly zeros with ones at positions x and N+y
    where (x,y) describe the position of the agent.

    If you specify walls, make sure that the list is only two dimensional,
    that is, e.g., [[1,1], [2,3], ...].

    Args:
        hardmode (bool): Whether or not to create an image as observation.
                         Defaults to False.
        random_grid (bool): Whether or not to create a randomized grid for
                            each instantiation of the environment.
                            Defaults to False.
        valid_grid (bool): Whether or not to run A* algorithm to check
                                that a grid has enough paths to the
                                reward at the beginning. This may be
                                computationally expensive. Defaults to False.

        **kwargs:
            grid_size (:obj:`list` of :obj:`int`): The size of the grid.
                                                   Defaults to [10, 10].
            reward_pos (:obj:`list` of :obj:`int`): The position of the
                                                    reward in the grid.
                                                    Defaults to [0, 0].
            max_steps (int): The maximum number of allowed time steps.
                             Defaults to 0, i.e. no restriction.
            walls_coord (:obj:`list`): The coordinates of walls within the
                                       gridworld. Defaults to [].
    """

    if 'grid_size' in kwargs and type(kwargs['grid_size']) is list:
      setattr(self, '_grid_size', kwargs['grid_size'])
    else:
      setattr(self, '_grid_size', [10, 10])
    if 'reward_pos' in kwargs and type(kwargs['reward_pos']) is list:
      setattr(self, '_reward_pos', kwargs['reward_pos'])
    else:
      setattr(self, '_reward_pos', [0, 0])
    if 'max_steps' in kwargs and type(kwargs['max_steps']) is int:
      setattr(self, '_max_steps', kwargs['max_steps'])
    else:
      setattr(self, '_max_steps', 0)
    if 'walls_coord' in kwargs and type(kwargs['walls_coord']) is list:
      setattr(self, '_walls_coord', kwargs['walls_coord'])
    else:
      setattr(self, '_walls_coord', [])

    if any(x <= 1 for x in self._grid_size):
      raise ValueError('The gridworld needs to be 3D. Instead, received ' +
                       f'grid of size {self._grid_size}'
                       )

    for wall in self._walls_coord:
      if not len(wall) == 2:
        raise ValueError('You specified a wall which is not 2D: ' +
                         f'{wall}.')
    if self._reward_pos in self._walls_coord:
      raise ValueError('The reward is located in a wall: ' +
                       f'{self._reward_pos}.')

    self.start = start

    if 'ring_size' in kwargs:
      if self.start is not 'ring':
        raise Warning('You chose a particular ring_size but do not have the corresponding start')
      if type(kwargs['ring_size']) is int:
        setattr(self, 'ring_size', kwargs['ring_size'])
    else:
      setattr(self, 'ring_size', 2)

    #set of posible starting pos


    #if hardmode = True the agent receives the image pixel as observation
    self.hardmode = hardmode

    #checks if a valid random grid was produced, reward can be reached high probability
    self.valid_grid = valid_grid

    #checks if starting from a position to the reward there is a path
    self.valid_path = valid_path

    # list of int: The current position of the agent. Not initial position.
    self._agent_pos = [0, 0]

    #generate random grid walls
    if random_grid:
      self.generate_random_walls()

    # int: image size is [x-size * 7, y-size * 7, z-size * 7] pixels.
    self._img_size = np.array(self._grid_size)* 7

    #:class:`gym.Box`: Image properties to be used as observation.
    self.observation_space = gym.spaces.Box(low=0, high=1,
                                            shape=(self._img_size[0],
                                                   self._img_size[1]),
                                            dtype=np.float32)
    #:class:`gym.Discrete`: The space of actions available to the agent.
    self.action_space = gym.spaces.Discrete(4)

    # function: Sets the static part of the observed image, i.e. walls.
    self._get_static_image()

    # numpy.ndarray of float: The currently observed image.
    self._img = np.zeros(self.observation_space.shape)

    # int: Number of time steps since last reset.
    self._time_step = 0

    self.starting_positions = []

    if self.start == 'fixed':
      self._agent_pos = [5, 5]
      self.starting_positions.append(self._agent_pos)

    elif self.start == 'ring':
      self.starting_positions = self.generating_starting_positions_ring(self.ring_size)


    elif self.start == 'random':
      self.starting_positions = self.generating_starting_positions_random()

  def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
      """
      An action moves the agent in one direction. The agent cannot cross walls
      or borders of the grid. If the maximum number of steps is exceeded, the
      agent receives a negative reward, and the game resets.
      Once the reward is hit, the agent receives the positive reward.

      Args:
          action (int): The index of the action to be taken.

      Returns:
          observation (numpy.ndarray): An array representing the current image
                                       of the environment.
          reward (float): The reward given after this time step.
          done (bool): The information whether or not the episode is finished.

      """
      # Move according to action.
      if action == 0:
        new_pos = [self._agent_pos[0] + 1, *self._agent_pos[1:2]]

      elif action == 1:
        new_pos = [self._agent_pos[0] - 1, *self._agent_pos[1:2]]

      elif action == 2:
        new_pos = [self._agent_pos[0],
                   self._agent_pos[1] + 1,
                   ]
      elif action == 3:
        new_pos = [self._agent_pos[0],
                   self._agent_pos[1] - 1,
                   ]
      else:
        raise TypeError('The action is not valid. The action should be an ' +
                        f'integer 0 <= action <= {self.action_space.n}'
                        )

      # If agent steps on a wall, it is not moved at all.
      if new_pos not in self._walls_coord:
        self._agent_pos = new_pos

      # If agent moves beyond the borders, it is not moved at all instead.
      for index, pos in enumerate(self._agent_pos):
        if pos >= self._grid_size[index]:
          self._agent_pos[index] = self._grid_size[index] - 1
        elif pos < 0:
          self._agent_pos[index] = 0

      # Check whether reward was found. Last step may get rewarded.
      self._time_step += 1
      if list(self._agent_pos) == self._reward_pos:
        reward = 1.
        done = True
      # Check whether maximum number of time steps has been reached.
      elif self._max_steps and self._time_step >= self._max_steps:
        reward = -1.
        done = True
      # Continue otherwise.
      else:
        reward = 0.
        done = False

      # Create a new image and observation.
      self._img = self._get_image()
      observation = self._get_observation()

      if not self.hardmode:
        observation = self.simplify_observation()

      return (observation, reward, done)

  def reset(self) -> np.ndarray:
    """
    Agent is reset to a random position in the cube from where it can reach
    the reward.
    Reward is placed in the respective plane.

    Returns:
        observation (numpy.ndarray): An array representing the current and
                                     the previous image of the environment.
    """
    # Reset internal timer.
    self._time_step = 0

    # Initialize agent position.
    choice = np.random.choice(len(self.starting_positions))
    self._agent_pos = self.starting_positions[choice]

    # Create initial image and observation.
    self._img = self._get_image()
    observation = self._get_observation()

    if not self.hardmode:
      observation = self.simplify_observation()

    return observation

  def render(self, mode: str = 'human') -> None:
    """
    Renders the current state of the environment as an image in a
    popup window.

    Args:
        mode (str): The mode in which the image is rendered.
                    Defaults to 'human' for human-friendly.
                    Currently, only 'human' is supported.
    """
    image = self._img.copy()
    # np.ndarray: the coordinate for the position of the reward in the image
    reward_coord = np.array(self._reward_pos) * 7
    # Draw reward into image.

    image[reward_coord[0]:reward_coord[0]+7,
    reward_coord[1]:reward_coord[1]+7] = self._img_reward

    if mode == 'human':
      cv2.namedWindow('image', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('image', 600, 600)
      cv2.imshow('image', np.uint8(image * 255))
      cv2.waitKey(10)
      # Give this plot a title,
      # so I know it's from matplotlib and not cv2
    else:
      raise NotImplementedError('We only support `human` render mode.')

  def simplify_observation(self) -> np.ndarray:
    """
    Returns position of the agent.

    Returns:
        observation (list): The simplified observation.
    """
    pos = np.array(self._agent_pos)
    return pos

  def get_optimal_path(self):
    """
    Calculates the optimal path for the current position of the agent
    using the A* search algorithm.

    Returns:
        optimal_path (`list` of `tuple`): The optimal path of 2D positions.
    """
    # Reduces to 2D gridworld.
    walls = []
    if len(self._walls_coord) > 0:
      walls = np.array(self._walls_coord)
      walls = list(map(tuple, walls))
    start_pos = self._agent_pos
    end_pos = self._reward_pos
    # Runs the A* algorithm.
    if end_pos != start_pos:
      a_star_alg = AStar()
      a_star_alg.init_grid(self._grid_size[0],
                           self._grid_size[1],
                           walls,
                           start_pos,
                           end_pos
                           )
      optimal_path = a_star_alg.solve()
    else:
      optimal_path = []

    return optimal_path

  def generate_random_walls(self, prob=0.2, max_ratio=0.01) -> None:
    """
    Clears the current set of walls and generates walls at random.
    If check_valid is True the process is repeated until the ratio of
    invalid to valid positions is larger than a given value.
    Valid positions are those that have a path to the reward.

    Args:
        prob (float): Probability of placing a a wall at any given point
                      in the plane. Defaults to 0.3.
        max_ratio (float): The maximum allowed ratio between invalid and
                           valid positions on the grid. Defaults to 0.2.
    """

    self._valid_pos = []
    self._invalid_pos = []
    while len(self._valid_pos) == 0 \
            or len(self._invalid_pos)/len(self._valid_pos) > max_ratio:
      # Reset walls
      self._walls_coord = []

      # Place walls at random.
      for i in range(self._grid_size[0]):
        for j in range(self._grid_size[1]):
          if not (i == self._reward_pos[0]
                  and j == self._reward_pos[1]):
            choice = np.random.choice(2, 1, p=[1-prob, prob])[0]
            if choice:
              self._walls_coord.append([i,j])

      # Get valid positions if required.
      if self.valid_grid:
        self._get_valid_pos()
      else:
        break


    # ----------------- helper methods -----------------------------------------

  def _get_static_image(self) -> None:
    """
    Generate the static part of the gridworld image, i.e. walls, image of
    the agent and reward.
    """
    # Empty world.
    gridworld = np.zeros(self.observation_space.shape)
    # Draw walls.
    wall_draw = np.ones((7, 7))
    for wall in self._walls_coord:
      wall_coord = np.array(wall) * 7
      gridworld[wall_coord[0]:wall_coord[0] + 7,
      wall_coord[1]:wall_coord[1] + 7] = wall_draw

    # array of float: The static part of the gridworld image, i.e. walls.
    self._img_static = gridworld

    # Draw 2D agent image.
    agent_draw_2d = np.zeros((7, 7))
    agent_draw_2d[0, 3] = 0.8
    agent_draw_2d[1, 0:7] = 0.9
    agent_draw_2d[2, 2:5] = 0.9
    agent_draw_2d[3, 2:5] = 0.9
    agent_draw_2d[4, 2] = 0.9
    agent_draw_2d[4, 4] = 0.9
    agent_draw_2d[5, 2] = 0.9
    agent_draw_2d[5, 4] = 0.9
    agent_draw_2d[6, 1:3] = 0.9
    agent_draw_2d[6, 4:6] = 0.9


    # array of float: The static 7 x 7 image of the agent.
    self._img_agent = agent_draw_2d

    # Draw 2D reward image.
    reward_draw_2d = np.zeros((7, 7))
    for i in range(7):
      reward_draw_2d[i, i] = 0.7
      reward_draw_2d[i, 6 - i] = 0.7

    self._img_reward = reward_draw_2d

  def _get_image(self) -> np.ndarray:
    """
    Generate an image from the current state of the environment.

    Returns:
        image (numpy.ndarray): An array representing an environment image.
    """
    image = self._img_static.copy()
    # np.ndarray: the coordinate for the position of the agent in the image
    agent_coord = np.array(self._agent_pos) * 7
    # Draw agent into static image.
    image[agent_coord[0]:agent_coord[0] + 7,
    agent_coord[1]:agent_coord[1] + 7] = self._img_agent

    return image

  def _get_observation(self) -> np.ndarray:
    """
    Generates an observation from an image.

    Returns:
        observation (numpy.ndarray): An 1 x (grid_size * 7) ** 2 array of
                                     the gridworld.
    """
    observation = self._img
    observation = np.reshape(observation,
                             (1, *self.observation_space.shape)
                             )
    return observation

  def _get_valid_pos(self) -> None:
    """
    Gets the valid and invalid positions with and without path to the
    reward.
    """
    #list: list of valid positions with path to reward.
    self._valid_pos = []
    #list: list of invalid positions without path to reward.
    self._invalid_pos = []
    for i in range(self._grid_size[0]):
      for j in range(self._grid_size[1]):
        if [i,j] not in self._walls_coord \
                and [i,j] != self._reward_pos:
          if self.get_optimal_path() is not None:
            self._valid_pos.append([i,j])
          else:
            self._invalid_pos.append([i,j])

  def generating_starting_positions_ring(self,ring_size):
    for i in range(2):
      corner = np.array(np.array(self._reward_pos)+np.array([(-1)**i*ring_size,(-1)**i*ring_size]))
      if corner[0] < self._grid_size[0] and corner[0] >= 0 and corner[1] < self._grid_size[1] and corner[1]>=0:
        self._agent_pos = list(corner)
        if self.get_optimal_path() is not None:
          self.starting_positions.append(list(corner))
      for k in range(1,2*ring_size+1):
        ring = list(corner - (-1)**i*np.array([0,k]))
        ring2 = list(corner - (-1)**i*np.array([k,0]))
        if ring[0] < self._grid_size[0] and ring[0] >= 0 and ring[1] < self._grid_size[1] and ring[1]>=0:
          if ring not in self._walls_coord and ring not in self.starting_positions:
            self._agent_pos = ring
            # Check whether there exists a path to the reward if required.
            if self.get_optimal_path() is not None:
              self.starting_positions.append(ring)

        if ring2[0] < self._grid_size[0] and ring2[0] >= 0 and ring2[1] < self._grid_size[1] and ring2[1] >= 0:
          if ring2 not in self._walls_coord and ring2 not in self.starting_positions:
            self._agent_pos = ring2
            # Check whether there exists a path to the reward if required.
            if self.get_optimal_path() is not None:
              self.starting_positions.append(ring2)
    return self.starting_positions

  def generating_starting_positions_random(self):
    return []
