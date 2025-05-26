import math
import pandas as pd
import pkg_resources
from agents.std_bb.BBController import BasalBolusController

CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class Pump:
    def __init__(self, args, patient_name):
        self.patient_name = patient_name
        self.use_bolus = args.expert_bolus
        self.use_cf = args.expert_cf
        self.action_scale = args.action_scale
        self.t_meal = args.t_meal
        self.pump_min = args.insulin_min
        self.pump_max = args.action_scale
        self.action_type = args.action_type
        self.args = args

        # set patient specific basal
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params[patient_params.Name.str.match(patient_name)]
        u2ss = params.u2ss.values.item()
        BW = params.BW.values.item()
        self.basal = u2ss * BW / 6000

    def _get_cgm(self, step_or_obs):
        """Step 혹은 Observation 객체에서 CGM 값을 추출한다."""
        if hasattr(step_or_obs, "observation"):
            return step_or_obs.observation.CGM
        return step_or_obs.CGM

    def calibrate(self, init_state):
        self.init_state = init_state
        if self.use_bolus:
            self.expert = BasalBolusController(self.args, patient_name=self.patient_name, use_bolus=self.use_bolus, use_cf=self.use_cf)
            self.bolus = self.expert.get_bolus(meal=0, glucose=self._get_cgm(self.init_state))
        else:
            self.bolus = 0

    def get_basal(self):
        return self.basal

    def get_bolus(self, state, info):
        if self.t_meal == 0:  # no meal announcement
            carbs = info['meal'] * info['sample_time']
            bolus_carbs = carbs
        elif self.t_meal == info['remaining_time']:  # with meal announcement
            bolus_carbs = info['future_carb']
        else:
            bolus_carbs = 0
        self.bolus = self.expert.get_bolus(meal=bolus_carbs, glucose=self._get_cgm(state))

    def action(self, agent_action=None, prev_state=None, prev_info=None):
        if self.use_bolus:
            self.get_bolus(prev_state, prev_info)

        if self.action_type == 'normal':
            agent_action = (agent_action + 1) / 2  # convert to [0, 1]
            agent_action = agent_action * self.action_scale

        elif self.action_type == 'sparse':
            if agent_action <= 0:
                agent_action = 0
            else:
                agent_action = agent_action * self.action_scale

        elif self.action_type == 'exponential':
            agent_action = self.action_scale * (math.exp((agent_action - 1) * 4))

        elif self.action_type == 'quadratic':
            if agent_action < 0:
                agent_action = (agent_action**2) * 0.05
                agent_action = min(0.05, agent_action)
            elif agent_action == 0:
                agent_action = 0
            else:
                agent_action = (agent_action**2) * self.action_scale

        elif self.action_type == 'proportional_quadratic':
            if agent_action <= 0.5:
                agent_action = ((agent_action-0.5)**2) * (0.5/(1.5**2))
                agent_action = min(0.5, agent_action)
            else:
                agent_action = ((agent_action-0.5)**2) * (self.action_scale/(0.5**2))

        rl_action = max(self.pump_min, agent_action)  # check if greater than 0
        rl_action = min(rl_action, self.pump_max)

        pump_action = (rl_action + self.bolus) if self.use_bolus else rl_action
        return rl_action, pump_action


def get_basal(patient_name='none'):
    if patient_name == 'none':
        print('Patient name not provided')
    quest = pd.read_csv(CONTROL_QUEST)
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    q = quest[quest.Name.str.match(patient_name)]
    params = patient_params[patient_params.Name.str.match(patient_name)]
    u2ss = params.u2ss.values.item()
    BW = params.BW.values.item()
    basal = u2ss * BW / 6000
    return basal

# patient:adolescent#001, Basal: 0.01393558889998341
# patient:adolescent#002, Basal: 0.01529933523331466
# patient:adolescent#003, Basal: 0.0107966168000268
# patient:adolescent#004, Basal: 0.01456052239999348
# patient:adolescent#005, Basal: 0.012040315333360101
# patient:adolescent#006, Basal: 0.014590183333350241
# patient:adolescent#007, Basal: 0.012943099999997907
# patient:adolescent#008, Basal: 0.009296317679986218
# patient:adolescent#009, Basal: 0.010107192533314517
# patient:adolescent#010, Basal: 0.01311652320003506
# patient:child#001, Basal: 0.006578422760004344
# patient:child#002, Basal: 0.006584850490398568
# patient:child#003, Basal: 0.004813171311526304
# patient:child#004, Basal: 0.008204957581639397
# patient:child#005, Basal: 0.00858548873873053
# patient:child#006, Basal: 0.006734515005432704
# patient:child#007, Basal: 0.007786704078078988
# patient:child#008, Basal: 0.005667427170273473
# patient:child#009, Basal: 0.006523757656342553
# patient:child#010, Basal: 0.006625406512238658
# patient:adult#001, Basal: 0.02112267499992533
# patient:adult#002, Basal: 0.022825539499994
# patient:adult#003, Basal: 0.023755205833326954
# patient:adult#004, Basal: 0.014797182203265
# patient:adult#005, Basal: 0.01966383496660751
# patient:adult#006, Basal: 0.028742228666635828
# patient:adult#007, Basal: 0.022858123833300104
# patient:adult#008, Basal: 0.01902372999996952
# patient:adult#009, Basal: 0.018896863133377337
# patient:adult#010, Basal: 0.01697815740005382
