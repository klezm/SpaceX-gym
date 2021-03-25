# https://github.com/EmbersArc/gym-rocketlander

import numpy as np
import Box2D
from Box2D.b2 import (
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    distanceJointDef,
    contactListener,
)
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering


def curriculum_decay(step):
    CURRICULUM_DECAY = -100
    CURRICULUM_ɛ = .33
    if np.random.uniform() > CURRICULUM_ɛ:
        step += 1  # to avoid division by zero, in case step starts at 0
        decay_height = np.exp(CURRICULUM_DECAY / step) + .1  # start at height 0 and increase each episode
        INIT_Y = np.random.normal(decay_height, .3)
        INIT_Y = np.clip(INIT_Y, .1, .95)
        INIT_X = .31 * (INIT_Y ** 2) * np.random.choice([1, -1])
        return C(INIT_X = INIT_X, INIT_Y = INIT_Y)
    else:
        return C()


def _plot_curriculum_decay():
    """
    Demo of curriculum_decay()
    """
    import matplotlib.pyplot as plt
    import scipy

    running_av = 50
    n = 2000

    plt.plot([curriculum_decay(s).INIT_Y for s in range(n)], lw = .2, label = "height")
    plt.plot(scipy.ndimage.filters.uniform_filter1d([curriculum_decay(s).INIT_Y for s in range(n)], running_av),
             label = f"running av height ({running_av})")
    plt.gcf().set_size_inches(10, 3.8)
    plt.title("Curriculum training: decreased initial height of rocket")  # "\n" f"averaged over {running_av} episodes")
    plt.ylabel("height [%]")
    plt.xlabel("episode")
    plt.legend()
    plt.show()


class C:
    def __init__(self,
                 CONTINUOUS: bool = None,
                 VEL_STATE: bool = None,
                 FPS: int = None,
                 SCALE_S: float = None,
                 INITIAL_RANDOM: float = None,

                 INIT_X: float = None,
                 INIT_Y: float = None,

                 START_HEIGHT: float = None,
                 START_SPEED: float = None,

                 MIN_THROTTLE: float = None,
                 GIMBAL_THRESHOLD: float = None,
                 MAIN_ENGINE_POWER: float = None,
                 SIDE_ENGINE_POWER: float = None,

                 ROCKET_WIDTH: float = None,
                 ROCKET_HEIGHT: float = None,
                 ENGINE_HEIGHT: float = None,
                 ENGINE_WIDTH: float = None,
                 THRUSTER_HEIGHT: float = None,

                 LEG_LENGTH: float = None,
                 BASE_ANGLE: float = None,
                 SPRING_ANGLE: float = None,
                 LEG_AWAY: float = None,

                 SHIP_HEIGHT: float = None,
                 SHIP_WIDTH: float = None,

                 VIEWPORT_H: int = None,
                 VIEWPORT_W: int = None,
                 H: float = None,
                 W: float = None,

                 MAX_SMOKE_LIFETIME: int = None,

                 SHAPING_REWARD: bool = None,
                 FUELCOST_REWARD: bool = None,

                 CURRICULUM: bool = None,
                 CURRICULUM_DECAY: float = None,
                 CURRICULUM_ɛ: float = None,
                 ):
        """

        :param CONTINUOUS:
        :param VEL_STATE: Add velocity info to state
        :param FPS:
        :param SCALE_S: Temporal Scaling, lower is faster - adjust forces appropriately
        :param INITIAL_RANDOM: Random scaling of initial velocity, higher is more difficult
        :param INIT_X: ±limit of N-distr from center
        :param INIT_Y: percentage of height
        :param START_HEIGHT:
        :param START_SPEED:
        :param MIN_THROTTLE:
        :param GIMBAL_THRESHOLD:
        :param MAIN_ENGINE_POWER:
        :param SIDE_ENGINE_POWER:
        :param ROCKET_WIDTH:
        :param ROCKET_HEIGHT:
        :param ENGINE_HEIGHT:
        :param ENGINE_WIDTH:
        :param THRUSTER_HEIGHT:
        :param LEG_LENGTH:
        :param BASE_ANGLE:
        :param SPRING_ANGLE:
        :param LEG_AWAY:
        :param SHIP_HEIGHT:
        :param SHIP_WIDTH:
        :param VIEWPORT_H:
        :param VIEWPORT_W:
        :param H:
        :param W:
        :param MAX_SMOKE_LIFETIME:
        """
        def this_or_that(x, y):
            return y if x is None else x

        self.CONTINUOUS = this_or_that(CONTINUOUS, True)
        self.VEL_STATE = this_or_that(VEL_STATE, True)  # Add velocity info to state
        self.FPS = this_or_that(FPS, 60)
        self.SCALE_S = this_or_that(SCALE_S, 0.35)  # Temporal Scaling, lower is faster - adjust forces appropriately
        self.INITIAL_RANDOM = this_or_that(INITIAL_RANDOM, 0.4)  # Random scaling of initial velocity, higher is more difficult

        self.INIT_X = this_or_that(INIT_X, .3)  # ±limit of N-distr from center
        self.INIT_Y = this_or_that(INIT_Y, .95)  # percentage of height

        self.START_HEIGHT = this_or_that(START_HEIGHT, 800.0)
        self.START_SPEED = this_or_that(START_SPEED, 80.0)

        # ROCKET
        self.MIN_THROTTLE = this_or_that(MIN_THROTTLE, 0.4)
        self.GIMBAL_THRESHOLD = this_or_that(GIMBAL_THRESHOLD, 0.4)
        self.MAIN_ENGINE_POWER = this_or_that(MAIN_ENGINE_POWER, 1600 * self.SCALE_S)
        self.SIDE_ENGINE_POWER = this_or_that(SIDE_ENGINE_POWER, 100 / self.FPS * self.SCALE_S)

        self.ROCKET_WIDTH = this_or_that(ROCKET_WIDTH, 3.66 * self.SCALE_S)
        self.ROCKET_HEIGHT = this_or_that(ROCKET_HEIGHT, self.ROCKET_WIDTH / 3.7 * 47.9)
        self.ENGINE_HEIGHT = this_or_that(ENGINE_HEIGHT, self.ROCKET_WIDTH * 0.5)
        self.ENGINE_WIDTH = this_or_that(ENGINE_WIDTH, self.ENGINE_HEIGHT * 0.7)
        self.THRUSTER_HEIGHT = this_or_that(THRUSTER_HEIGHT, self.ROCKET_HEIGHT * 0.86)

        # LEGS
        self.LEG_LENGTH = this_or_that(LEG_LENGTH, self.ROCKET_WIDTH * 2.2)
        self.BASE_ANGLE = this_or_that(BASE_ANGLE, -0.27)
        self.SPRING_ANGLE = this_or_that(SPRING_ANGLE, 0.27)
        self.LEG_AWAY = this_or_that(LEG_AWAY, self.ROCKET_WIDTH / 2)

        # SHIP
        self.SHIP_HEIGHT = this_or_that(SHIP_HEIGHT, self.ROCKET_WIDTH)
        self.SHIP_WIDTH = this_or_that(SHIP_WIDTH, self.SHIP_HEIGHT * 40)

        # VIEWPORT
        self.VIEWPORT_H = this_or_that(VIEWPORT_H, 720)
        self.VIEWPORT_W = this_or_that(VIEWPORT_W, 500)
        self.H = this_or_that(H, 1.1 * self.START_HEIGHT * self.SCALE_S)
        self.W = this_or_that(W, float(self.VIEWPORT_W) / self.VIEWPORT_H * self.H)

        # SMOKE FOR VISUALS
        self.MAX_SMOKE_LIFETIME = this_or_that(MAX_SMOKE_LIFETIME, 2 * self.FPS)

        # FLAGS FOR REWARDS
        self.SHAPING_REWARD = this_or_that(SHAPING_REWARD, False)  # True
        self.FUELCOST_REWARD = this_or_that(FUELCOST_REWARD, False)  # True

        # FLAG FOR DECAYING CURRICULUM
        self.CURRICULUM = this_or_that(CURRICULUM, True)  # False
        self.CURRICULUM_DECAY = this_or_that(CURRICULUM_DECAY, -100)
        self.CURRICULUM_ɛ = this_or_that(CURRICULUM_ɛ, .33)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
                self.env.water in [contact.fixtureA.body, contact.fixtureB.body]
                or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]
                or self.env.containers[0] in [contact.fixtureA.body, contact.fixtureB.body]
                or self.env.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]
        ):
            self.env.game_over = True
        else:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class GymRocketLander(gym.Env):
    # metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": C.FPS}
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": -1}

    def __init__(self, constants = C()):
        """
        The objective of this environment is to land a rocket on a ship.

        STATE VARIABLES
        The state consists of the following variables:
            - x position
            - y position
            - angle
            - first leg ground contact indicator
            - second leg ground contact indicator
            - throttle
            - engine gimbal
        If self.C.VEL_STATE is set to true, the velocities are included:
            - x velocity
            - y velocity
            - angular velocity
        all state variables are roughly in the range [-1, 1]

        CONTROL INPUTS
        Discrete control inputs are:
            - gimbal left
            - gimbal right
            - throttle up
            - throttle down
            - use first control thruster
            - use second control thruster
            - no action

        Continuous control inputs are:
            - gimbal (left/right)
            - throttle (up/down)
            - control thruster (left/right)
        """
        # if constants is not None:
        #     global C
        self.C = constants
        self.C_INIT_X_Y = (self.C.INIT_X, self.C.INIT_Y)  # save original starting values
        self.metadata["video.frames_per_second"] = self.C.FPS
        # else:
        #     C = self.D
        self._seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.legs = []

        high = np.array([1, 1, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf], dtype = np.float32)
        low = -high
        if not self.C.VEL_STATE:
            high = high[0:7]
            low = low[0:7]

        self.observation_space = spaces.Box(low, high, dtype = np.float32)

        if self.C.CONTINUOUS:
            self.action_space = spaces.Box(-1.0, +1.0, (3,), dtype = np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        self.reset()

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.water:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []

    def curriculum_decay(self):
        if np.random.uniform() > self.C.CURRICULUM_ɛ:
            decay_height = np.exp(self.C.CURRICULUM_DECAY / (self.episode_number + 1)) + .1  # start at height 0 and increase each episode
            INIT_Y = np.random.normal(decay_height, .3)
            self.C.INIT_Y = np.clip(INIT_Y, .1, .95)
            self.C.INIT_X = .31 * (INIT_Y ** 2) * np.random.choice([1, -1])
        else:
            # reset
            self.C.INIT_X, self.C.INIT_Y = self.C_INIT_X_Y

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.gimbal = 0.0
        self.landed_ticks = 0
        self.episode_number += 1
        self.stepnumber = 0
        self.smoke = []

        if self.C.CURRICULUM:
            # slowly increases difficulty of starting position
            self.curriculum_decay()

        # self.terrainheigth = self.np_random.uniform(self.C.H / 20, self.C.H / 10)
        self.terrainheigth = self.C.H / 20
        self.shipheight = self.terrainheigth + self.C.SHIP_HEIGHT
        # ship_pos = self.np_random.uniform(0, self.C.SHIP_WIDTH / SCALE) + self.C.SHIP_WIDTH / SCALE
        ship_pos = self.C.W / 2
        self.helipad_x1 = ship_pos - self.C.SHIP_WIDTH / 2
        self.helipad_x2 = self.helipad_x1 + self.C.SHIP_WIDTH
        self.helipad_y = self.terrainheigth + self.C.SHIP_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures = fixtureDef(
                shape = polygonShape(
                    vertices = (
                        (0, 0),
                        (self.C.W, 0),
                        (self.C.W, self.terrainheigth),
                        (0, self.terrainheigth),
                    )
                ),
                friction = 0.1,
                restitution = 0.0,
            )
        )
        self.water.color1 = rgb(70, 96, 176)

        self.ship = self.world.CreateStaticBody(
            fixtures = fixtureDef(
                shape = polygonShape(
                    vertices = (
                        (self.helipad_x1, self.terrainheigth),
                        (self.helipad_x2, self.terrainheigth),
                        (self.helipad_x2, self.terrainheigth + self.C.SHIP_HEIGHT),
                        (self.helipad_x1, self.terrainheigth + self.C.SHIP_HEIGHT),
                    )
                ),
                friction = 0.5,
                restitution = 0.0,
            )
        )

        self.containers = []
        for side in [-1, 1]:
            self.containers.append(
                self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape = polygonShape(
                            vertices = (
                                (
                                    ship_pos + side * 0.95 * self.C.SHIP_WIDTH / 2,
                                    self.helipad_y,
                                ),
                                (
                                    ship_pos + side * 0.95 * self.C.SHIP_WIDTH / 2,
                                    self.helipad_y + self.C.SHIP_HEIGHT,
                                ),
                                (
                                    ship_pos
                                    + side * 0.95 * self.C.SHIP_WIDTH / 2
                                    - side * self.C.SHIP_HEIGHT,
                                    self.helipad_y + self.C.SHIP_HEIGHT,
                                ),
                                (
                                    ship_pos
                                    + side * 0.95 * self.C.SHIP_WIDTH / 2
                                    - side * self.C.SHIP_HEIGHT,
                                    self.helipad_y,
                                ),
                            )
                        ),
                        friction = 0.2,
                        restitution = 0.0,
                    )
                )
            )
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)

        # initial_x = self.C.W / 2 + self.C.W * np.random.uniform(-0.3, 0.3)
        # initial_y = self.C.H * 0.95
        initial_x = self.C.W / 2 + self.C.W * np.random.uniform(-self.C.INIT_X, self.C.INIT_X)
        initial_y = self.C.H * self.C.INIT_Y
        self.lander = self.world.CreateDynamicBody(
            position = (initial_x, initial_y),
            angle = 0.0,
            fixtures = fixtureDef(
                shape = polygonShape(
                    vertices = (
                        (-self.C.ROCKET_WIDTH / 2, 0),
                        (+self.C.ROCKET_WIDTH / 2, 0),
                        (self.C.ROCKET_WIDTH / 2, +self.C.ROCKET_HEIGHT),
                        (-self.C.ROCKET_WIDTH / 2, +self.C.ROCKET_HEIGHT),
                    )
                ),
                density = 1.0,
                friction = 0.5,
                categoryBits = 0x0010,
                maskBits = 0x001,
                restitution = 0.0,
            ),
        )

        self.lander.color1 = rgb(230, 230, 230)

        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position = (initial_x - i * self.C.LEG_AWAY, initial_y + self.C.ROCKET_WIDTH * 0.2),
                angle = (i * self.C.BASE_ANGLE),
                fixtures = fixtureDef(
                    shape = polygonShape(
                        vertices = (
                            (0, 0),
                            (0, self.C.LEG_LENGTH / 25),
                            (i * self.C.LEG_LENGTH, 0),
                            (i * self.C.LEG_LENGTH, -self.C.LEG_LENGTH / 20),
                            (i * self.C.LEG_LENGTH / 3, -self.C.LEG_LENGTH / 7),
                        )
                    ),
                    density = 1,
                    restitution = 0.0,
                    friction = 0.2,
                    categoryBits = 0x0020,
                    maskBits = 0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA = self.lander,
                bodyB = leg,
                localAnchorA = (i * self.C.LEG_AWAY, self.C.ROCKET_WIDTH * 0.2),
                localAnchorB = (0, 0),
                enableLimit = True,
                maxMotorTorque = 2500.0,
                motorSpeed = -0.05 * i,
                enableMotor = True,
            )
            djd = distanceJointDef(
                bodyA = self.lander,
                bodyB = leg,
                anchorA = (i * self.C.LEG_AWAY, self.C.ROCKET_HEIGHT / 8),
                anchorB = leg.fixtures[0].body.transform * (i * self.C.LEG_LENGTH, 0),
                collideConnected = False,
                frequencyHz = 0.01,
                dampingRatio = 0.9,
            )
            if i == 1:
                rjd.lowerAngle = -self.C.SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = +self.C.SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)

            self.legs.append(leg)

        self.lander.linearVelocity = (
            -self.np_random.uniform(0, self.C.INITIAL_RANDOM)
            * self.C.START_SPEED
            * (initial_x - self.C.W / 2)
            / self.C.W,
            -self.C.START_SPEED,
        )

        self.lander.angularVelocity = (1 + self.C.INITIAL_RANDOM) * np.random.uniform(-1, 1)

        self.drawlist = (
                self.legs + [self.water] + [self.ship] + self.containers + [self.lander]
        )

        if self.C.CONTINUOUS:
            return self.step([0, 0, 0])[0]
        else:
            return self.step(6)[0]

    def step(self, action):

        self.force_dir = 0

        if self.C.CONTINUOUS:
            np.clip(action, -1, 1)
            self.gimbal += action[0] * 0.15 / self.C.FPS
            self.throttle += action[1] * 0.5 / self.C.FPS
            if action[2] > 0.5:
                self.force_dir = 1
            elif action[2] < -0.5:
                self.force_dir = -1
        else:
            if action == 0:
                self.gimbal += 0.01
            elif action == 1:
                self.gimbal -= 0.01
            elif action == 2:
                self.throttle += 0.01
            elif action == 3:
                self.throttle -= 0.01
            elif action == 4:  # left
                self.force_dir = -1
            elif action == 5:  # right
                self.force_dir = 1

        self.gimbal = np.clip(self.gimbal, -self.C.GIMBAL_THRESHOLD, self.C.GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = (
            0
            if self.throttle == 0.0
            else self.C.MIN_THROTTLE + self.throttle * (1 - self.C.MIN_THROTTLE)
        )

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (
            -np.sin(self.lander.angle + self.gimbal) * self.C.MAIN_ENGINE_POWER * self.power,
            np.cos(self.lander.angle + self.gimbal) * self.C.MAIN_ENGINE_POWER * self.power,
        )
        self.lander.ApplyForce(force = force, point = force_pos, wake = False)

        # control thruster force
        force_pos_c = self.lander.position + self.C.THRUSTER_HEIGHT * np.array(
            (np.sin(self.lander.angle), np.cos(self.lander.angle))
        )
        force_c = (
            -self.force_dir * np.cos(self.lander.angle) * self.C.SIDE_ENGINE_POWER,
            self.force_dir * np.sin(self.lander.angle) * self.C.SIDE_ENGINE_POWER,
        )
        self.lander.ApplyLinearImpulse(impulse = force_c, point = force_pos_c, wake = False)

        self.world.Step(1.0 / self.C.FPS, 60, 60)

        pos = self.lander.position
        vel_l = np.array(self.lander.linearVelocity) / self.C.START_SPEED
        vel_a = self.lander.angularVelocity
        x_distance = (pos.x - self.C.W / 2) / self.C.W
        y_distance = (pos.y - self.shipheight) / (self.C.H - self.shipheight)

        angle = (self.lander.angle / np.pi) % 2
        if angle > 1:
            angle -= 2

        state = [
            2 * x_distance,
            2 * (y_distance - 0.5),
            angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            2 * (self.throttle - 0.5),
            (self.gimbal / self.C.GIMBAL_THRESHOLD),
        ]
        if self.C.VEL_STATE:
            state.extend([vel_l[0], vel_l[1], vel_a])

        # REWARD -------------------------------------------------------------------------------------------------------

        # state variables for reward
        distance = np.linalg.norm(
            (3 * x_distance, y_distance)
        )  # weight x position more
        speed = np.linalg.norm(vel_l)
        groundcontact = self.legs[0].ground_contact or self.legs[1].ground_contact
        brokenleg = (
                            self.legs[0].joint.angle < 0 or self.legs[1].joint.angle > -0
                    ) and groundcontact
        outside = abs(pos.x - self.C.W / 2) > self.C.W / 2 or pos.y > self.C.H

        landed = (
                self.legs[0].ground_contact and self.legs[1].ground_contact and speed < 0.1
        )
        done = False

        if self.C.FUELCOST_REWARD:
            fuelcost = 0.1 * (0.5 * self.power + abs(self.force_dir)) / self.C.FPS
            reward = -fuelcost
        else:
            # TODO: should we give a per frame negative reward?
            reward = 0

        if outside or brokenleg:
            self.game_over = True

        if self.game_over:
            done = True
            if not self.C.SHAPING_REWARD:
                reward = -1.0
        else:
            # reward shaping
            if self.C.SHAPING_REWARD:
                shaping = -0.5 * (distance + speed + abs(angle) ** 2 + abs(vel_a) ** 2)
                shaping += 0.1 * (self.legs[0].ground_contact + self.legs[1].ground_contact)
                if self.prev_shaping is not None:
                    reward += shaping - self.prev_shaping
                self.prev_shaping = shaping

            if landed:
                self.landed_ticks += 1
            else:
                self.landed_ticks = 0
            if self.landed_ticks == self.C.FPS:
                done = True
                if not self.C.SHAPING_REWARD:
                    # TODO: should we give a higher "won" reward? (we can set a won flag and outside we can use that)
                    reward = 1.0  # the agent won (= standing for 1 sec on the boat)

        if done and self.C.SHAPING_REWARD:
            reward += max(-1, 0 - 2 * (speed + distance + abs(angle) + abs(vel_a)))

        reward = np.clip(reward, -1, 1)

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1

        return np.array(state), reward, done, {}

    def render(self, mode = "human", close = False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:

            self.viewer = rendering.Viewer(self.C.VIEWPORT_W, self.C.VIEWPORT_H)
            self.viewer.set_bounds(0, self.C.W, 0, self.C.H)

            sky = rendering.FilledPolygon(((0, 0), (0, self.C.H), (self.C.W, self.C.H), (self.C.W, 0)))
            self.sky_color = rgb(126, 150, 233)
            sky.set_color(*self.sky_color)
            self.sky_color_half_transparent = (
                    np.array((np.array(self.sky_color) + rgb(255, 255, 255))) / 2
            )
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()

            engine = rendering.FilledPolygon(
                (
                    (0, 0),
                    (self.C.ENGINE_WIDTH / 2, -self.C.ENGINE_HEIGHT),
                    (-self.C.ENGINE_WIDTH / 2, -self.C.ENGINE_HEIGHT),
                )
            )
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(0.4, 0.4, 0.4)
            self.viewer.add_geom(engine)

            self.fire = rendering.FilledPolygon(
                (
                    (self.C.ENGINE_WIDTH * 0.4, 0),
                    (-self.C.ENGINE_WIDTH * 0.4, 0),
                    (-self.C.ENGINE_WIDTH * 1.2, -self.C.ENGINE_HEIGHT * 5),
                    (0, -self.C.ENGINE_HEIGHT * 8),
                    (self.C.ENGINE_WIDTH * 1.2, -self.C.ENGINE_HEIGHT * 5),
                )
            )
            self.fire.set_color(*rgb(255, 230, 107))
            self.firescale = rendering.Transform(scale = (1, 1))
            self.firetrans = rendering.Transform(translation = (0, -self.C.ENGINE_HEIGHT))
            self.fire.add_attr(self.firescale)
            self.fire.add_attr(self.firetrans)
            self.fire.add_attr(self.enginetrans)
            self.fire.add_attr(self.rockettrans)

            smoke = rendering.FilledPolygon(
                (
                    (self.C.ROCKET_WIDTH / 2, self.C.THRUSTER_HEIGHT * 1),
                    (self.C.ROCKET_WIDTH * 3, self.C.THRUSTER_HEIGHT * 1.03),
                    (self.C.ROCKET_WIDTH * 4, self.C.THRUSTER_HEIGHT * 1),
                    (self.C.ROCKET_WIDTH * 3, self.C.THRUSTER_HEIGHT * 0.97),
                )
            )
            smoke.set_color(*self.sky_color_half_transparent)
            self.smokescale = rendering.Transform(scale = (1, 1))
            smoke.add_attr(self.smokescale)
            smoke.add_attr(self.rockettrans)
            self.viewer.add_geom(smoke)

            self.gridfins = []
            for i in (-1, 1):
                finpoly = (
                    (i * self.C.ROCKET_WIDTH * 1.1, self.C.THRUSTER_HEIGHT * 1.01),
                    (i * self.C.ROCKET_WIDTH * 0.4, self.C.THRUSTER_HEIGHT * 1.01),
                    (i * self.C.ROCKET_WIDTH * 0.4, self.C.THRUSTER_HEIGHT * 0.99),
                    (i * self.C.ROCKET_WIDTH * 1.1, self.C.THRUSTER_HEIGHT * 0.99),
                )
                gridfin = rendering.FilledPolygon(finpoly)
                gridfin.add_attr(self.rockettrans)
                gridfin.set_color(0.25, 0.25, 0.25)
                self.gridfins.append(gridfin)

        if self.stepnumber % round(self.C.FPS / 10) == 0 and self.power > 0:
            s = [
                self.C.MAX_SMOKE_LIFETIME * self.power,  # total lifetime
                0,  # current lifetime
                self.power * (1 + 0.2 * np.random.random()),  # size
                np.array(self.lander.position)
                + self.power
                * self.C.ROCKET_WIDTH
                * 10
                * np.array(
                    (
                        np.sin(self.lander.angle + self.gimbal),
                        -np.cos(self.lander.angle + self.gimbal),
                    )
                )
                + self.power * 5 * (np.random.random(2) - 0.5),
            ]  # position
            self.smoke.append(s)

        for s in self.smoke:
            s[1] += 1
            if s[1] > s[0]:
                self.smoke.remove(s)
                continue
            t = rendering.Transform(translation = (s[3][0], s[3][1] + self.C.H * s[1] / 2000))
            self.viewer.draw_circle(
                radius = 0.05 * s[1] + s[2],
                color = self.sky_color
                        + (1 - (2 * s[1] / s[0] - 1) ** 2)
                        / 3
                        * (self.sky_color_half_transparent - self.sky_color),
            ).add_attr(t)

        self.viewer.add_onetime(self.fire)
        for g in self.gridfins:
            self.viewer.add_onetime(g)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color = obj.color1)

        for l in zip(self.legs, [-1, 1]):
            path = [
                self.lander.fixtures[0].body.transform
                * (l[1] * self.C.ROCKET_WIDTH / 2, self.C.ROCKET_HEIGHT / 8),
                l[0].fixtures[0].body.transform * (l[1] * self.C.LEG_LENGTH * 0.8, 0),
            ]
            self.viewer.draw_polyline(
                path, color = self.ship.color1, linewidth = 1 if self.C.START_HEIGHT > 500 else 2
            )

        self.viewer.draw_polyline(
            (
                (self.helipad_x2, self.terrainheigth + self.C.SHIP_HEIGHT),
                (self.helipad_x1, self.terrainheigth + self.C.SHIP_HEIGHT),
            ),
            color = rgb(206, 206, 2),
            linewidth = 1,
        )

        self.rockettrans.set_translation(*self.lander.position)
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(newx = 1, newy = self.power * np.random.uniform(1, 1.3))
        self.smokescale.set_scale(newx = self.force_dir, newy = 1)

        return self.viewer.render(return_rgb_array = mode == "rgb_array")


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255
