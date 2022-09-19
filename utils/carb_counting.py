import pandas as pd
import pkg_resources
import random

CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')

# paper: https://www.liebertpub.com/doi/full/10.1089/dia.2019.0502
# UVA /Padova Univesity.
# CHO_estimate = CHO_real + CHO_estimate_error


def carb_estimate(cho_real, hour, patient_name, type):
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    params = patient_params[patient_params.Name.str.match(patient_name)]
    BW = params.BW.values.item()

    # identify the meal type:
    cho_estimate_error, snack , lunch, dinner, breakfast = 0, 0, 0, 0, 0
    if 6 <= hour <= 9:
        breakfast = 1
    elif 12 <= hour <= 15:
        lunch = 1
    elif 18 <= hour <= 21:
        dinner = 1
    else:
        snack = 1

    if type == 'linear':  # eqn (5)
        cho_estimate_error = 9.22 - 0.34 * cho_real + (0.09 * BW) + (3.11 * lunch) + (0.68 * dinner) - (7.05 * snack)
    elif type == 'quadratic':  # eqn (7)
        cho_estimate_error = 3.56 - 0.07 * cho_real - 0.0008 * cho_real * cho_real + (6.77 * lunch) + (18.01 * dinner) - (0.49 * snack) - \
                             (0.08 * cho_real * lunch) - (0.25 * cho_real * dinner) - (0.06 * cho_real * snack)
    elif type == 'real':
        cho_estimate_error = 0
    elif type == 'rand':
        # rnd = random.getrandbits(1)
        # sign = 1 if rnd == 1 else -1
        # cho_estimate_error = (cho_real * 0.1) * sign
        r = random.randint(-20, 20)
        cho_estimate_error = (cho_real * r) / 100
    else:
        print('Please specify the type of meal estimation')
        exit()

    carb_est = max(0, cho_real + cho_estimate_error)
    return carb_est