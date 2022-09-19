from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

from utils.extended_scenario import RandomScenario

import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, args=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        # print('Simulation environment is created extending the simglucose library')
        self.args = args
        self.INSULIN_PUMP_HARDWARE = args.pump
        self.SENSOR_HARDWARE = args.sensor

        if patient_name is None:
            patient_name = 'adolescent#001'
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()


        self.meal_announce_time = args.t_meal

    def _step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        future_carb, remaining_time, day_hour, day_min, meal_type = self.announce_meal(meal_announce=None)

        if self.reward_fun is None:
            state, reward, done, info = self.env.step(act)
            return state, reward, done, info
        else:
            state, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
            info['future_carb'] = future_carb
            info['remaining_time'] = remaining_time
            info['day_hour'] = day_hour
            info['day_min'] = day_min
            info['meal_type'] = meal_type

        return state, reward, done, info

    def announce_meal(self, meal_announce=None):
        t = self.env.time.hour * 60 + self.env.time.minute
        #sim_t = str(self.env.time.hour)+":"+str(self.env.time.minute)
        #sim_time = datetime.strptime(sim_t, '%H:%M')

        sampling_rate = self.sampling_time
        meal_type = 0
        for i, m_t in enumerate(self.env.scenario.scenario['meal']['time']):
            # round up to sampling rate floor
            if m_t % sampling_rate != 0:
                m_tr = m_t - (m_t % sampling_rate)
            else:
                m_tr = m_t
            if meal_announce is None:
                ma = self.meal_announce_time
            else:
                ma = meal_announce

            #  meal announcement and type
            if (m_tr - ma) <= t <= m_tr:
                meal_type = 0.3 if self.env.scenario.scenario['meal']['amount'][i] <= 40 else 1
                return self.env.scenario.scenario['meal']['amount'][i], (m_tr - t), \
                       self.env.time.hour, self.env.time.minute, meal_type
            # elif m_tr < t <= (m_tr + ma + ma):
            #     meal_type = 0.3 if self.env.scenario.scenario['meal']['amount'][i] <= 30 else 1
            #     return self.env.scenario.scenario['meal']['amount'][i], (m_tr - t), \
            #            self.env.time.hour, self.env.time.minute, meal_type
            elif t < (m_tr - ma):  # if time is lower than this meal no point comparing future meals.
                break

        return 0, 0, self.env.time.hour, self.env.time.minute, meal_type

    def _reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        #hour = self.np_random.randint(low=0.0, high=24.0)
        hour = 23  #always start at midnight

        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        self.sampling_time = sensor.sample_time
        scenario = RandomScenario(start_time=start_time, seed=seed3, opt=self.args)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))