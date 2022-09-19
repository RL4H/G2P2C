import scipy.signal
import numpy as np
import logging
import gym
from gym.envs.registration import register
import warnings
import math
import torch


def get_env(args, patient_name='adult#001', env_id='simglucose-adult1-v0', custom_reward=None, seed=None):
    register(
        id=env_id,
        entry_point='utils.extended_T1DSimEnv:T1DSimEnv',  # simglucose.envs:T1DSimEnv
        kwargs={'patient_name': patient_name,
                'reward_fun': custom_reward,
                'seed':seed,
                'args': args}
    )
    env = gym.make(env_id)
    env_conditions = {'insulin_min': env.action_space.low, 'insulin_max': env.action_space.high,
                      'cgm_low': env.observation_space.low, 'cgm_high': env.observation_space.high}
    logging.info(env_conditions)
    # print("Experiment running for {}, creating env {}.".format(patient_name, env_id))
    # print(env.observation_space.shape[0], env.observation_space.shape[1])
    return env


def get_patient_env():
    patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
    env_ids = (['simglucose-adolescent{}-v0'.format(str(i)) for i in range(1, 11)] +
               ['simglucose-child{}-v0'.format(str(i)) for i in range(1, 11)] +
               ['simglucose-adult{}-v0'.format(str(i)) for i in range(1, 11)])
    return patients, env_ids


def get_patient_index(patient_type=None):
    low_index, high_index = -1, -1
    if patient_type == 'adult':
        low_index, high_index = 20, 29
    elif patient_type == 'child':
        low_index, high_index = 10, 19
    elif patient_type == 'adolescent':
        low_index, high_index = 0, 9
    else:
        print('Error in assigning the patient!')
    return low_index, high_index


def risk_index(BG, horizon):
    # BG is in mg/dL, horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = np.array(BG[-horizon:])
        BG_to_compute[BG_to_compute < 1] = 1
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return LBGI, HBGI, RI


def custom_reward(bg_hist, **kwargs):
    return -risk_index([bg_hist[-1]], 1)[-1]


def custom_reward_traj(bg_hist, k, **kwargs):
    return -risk_index([bg_hist], k)[-1]


def custom_reward2(bg_hist, **kwargs):
    return risk_index([bg_hist[-1]], 1)


def get_exp_avg(arr, scale):
    pow, ema = 0, 0
    for t in reversed(range(len(arr))):
        ema += (scale**pow)*arr[t]
        pow += 1
    return ema


def time_in_range(metric, meal_data, insulin_data, episode, counter, display=False):
    '''
    :param metric: an array with the cgm readings
    :return: time in ranges hypo, hyper, normo
    '''
    hypo = 0
    normo = 0
    hyper = 0
    severe_hypo = 0
    severe_hyper = 0

    if len(metric) == 0:
        metric.append(0)  # to avoid division by zero

    for reading in metric:
        if reading <= 50:
            severe_hypo += 1
        elif reading <= 70:
            hypo += 1
        elif reading <= 180:
            normo += 1
        elif reading <= 300:
            hyper += 1
        else:
            severe_hyper += 1

    LBGI, HBGI, RI = risk_index(metric, len(metric))

    if display:
        print("Episode {} ran for {} steps ({} hours)...".format(episode, counter, counter/20))
        print("Time in Normoglycemia (70 - 180) : {}".format(normo * 100 / len(metric)))
        print("Time in Hypoglycemia (<70): {} ".format(hypo * 100 / len(metric)))
        print("Time in Severe Hypoglycemia (<50): {}".format(severe_hypo * 100 / len(metric)))
        print("Time in Hyperglycemia (>180): {} ".format(hyper * 100 / len(metric)))
        print("LBGI: {}".format(LBGI))
        print("HBGI: {}".format(HBGI))
        print("RI: {}".format(RI))

    # todo add other useful metrics, capability to save metrics to a file.
    return (normo * 100 / len(metric)), (hypo * 100 / len(metric)), (severe_hypo * 100 / len(metric)), \
           (hyper * 100 / len(metric)), LBGI, HBGI, RI, (severe_hyper * 100 / len(metric))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """

    # pw, Gt = 0, 0
    # returns = []
    # for r in x[::-1]:
    #     Gt = r + Gt * (discount ** pw)
    #     pw += 1
    #     returns.append(Gt)
    # returns = returns[::-1]

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def set_logger(LOG_DIR):
    log_filename = LOG_DIR + '/debug.log'
    logging.basicConfig(filename=log_filename, filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)


def linear_scaling(x, x_min, x_max):
    # scale to [-1, 1] range
    y = ((x - x_min) * 2 / (x_max - x_min)) - 1
    return y


def inverse_linear_scaling(y, x_min, x_max):
    # scale back to original
    x = (y+1) * (x_max - x_min) * (1/2) + x_min
    return x


def reverse_kl_approx(p, q):
    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
    # https://github.com/DLR-RM/stable-baselines3/issues/417
    # https://dibyaghosh.com/blog/probability/kldivergence.html
    # KL (q||p) = (r-1) - log(r)
    # x~q, r = p(x)/q(x) = new/old
    log_ratio = p - q
    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
    return approx_kl


def forward_kl_approx(p, q):
    # KL (p||q) = rlog(r)-(r-1)
    # x~q, r = p(x)/q(x)
    log_ratio = p - q
    approx_kl = torch.mean((torch.exp(log_ratio)*log_ratio) - (torch.exp(log_ratio)-1))
    return approx_kl


def f_kl(log_p, log_q):
    # KL[q,p] = (r-1) - log(r) ;forward KL
    log_ratio = log_p - log_q
    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
    return approx_kl

def r_kl(log_p, log_q):
    # KL[p, q] = rlog(r) -(r-1)
    log_ratio = log_p - log_q
    approx_kl = torch.mean(torch.exp(log_ratio)*log_ratio - (torch.exp(log_ratio) - 1))
    return approx_kl