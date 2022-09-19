import pandas as pd
import pkg_resources
from collections import deque


CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class BasalBolusController:
    def __init__(self, args, patient_name=None, use_bolus=True, use_cf=True):
        quest = pd.read_csv(CONTROL_QUEST)
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params[patient_params.Name.str.match(patient_name)]
        u2ss = params.u2ss.values.item()
        self.BW = params.BW.values.item()

        self.q = quest[quest.Name.str.match(patient_name)]
        self.use_bolus = args.use_bolus
        self.use_cf = args.use_cf
        self.target = args.target_glucose
        self.sample_time = args.sampling_rate
        self.cf_target = args.glucose_cf_target
        self.past_meal_memory = deque(36 * [0], 36)

        self.basal = u2ss * self.BW / 6000
        self.TDI = self.q.TDI.values
        self.CR = self.q.CR.values  # (500/self.TDI)  #
        self.CF = self.q.CF.values  # (1800/self.TDI)  #

        self.adjust_parameters()

    def get_action(self, meal=0, glucose=0):
        # the meal value used here is the info['meal'] * sampling time. so the actual carb amount!.
        cooldown = True if sum(self.past_meal_memory) == 0 else False
        bolus = 0
        if self.use_bolus:
            if meal > 0:
                bolus = (meal / self.CR +
                         (glucose > self.cf_target) * cooldown * (glucose - self.target) / self.CF).item()
                bolus = bolus / self.sample_time
        self.past_meal_memory.append(meal)
        return self.basal + bolus

    def get_bolus(self, meal=0, glucose=0):
        # the meal value used here is the info['meal'] * sampling time. so the actual carb amount!.
        cooldown = True if sum(self.past_meal_memory) == 0 else False
        bolus = 0
        if self.use_bolus:
            if meal > 0:
                bolus = (meal / self.CR +
                         (glucose > self.cf_target) * cooldown * (glucose - self.target) / self.CF ).item()
                bolus = bolus / self.sample_time
        self.past_meal_memory.append(meal)
        return bolus

    def adjust_parameters(self):
        #self.TDI = self.BW * 0.55
        self.CR = 500 / self.TDI
        self.CF = 1800 / self.TDI
        self.basal = (self.TDI * 0.48) / (24 * 60)
        #print('Parameters adjusted!')
