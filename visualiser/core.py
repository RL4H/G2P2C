import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import math


def experiment_error_check(cohort, algorithm, algoAbbreviation,
                           subjects=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                           seeds=['1', '2', '3'], n_trials=500):
    print("Checking Experiment Errors for {} cohort, {} Algorithm".format(cohort, algorithm))
    init_bg = 0
    for sub in subjects:
        print("Checking subject: {}".format(sub))
        incomplete = False
        for s in seeds:
            FOLDER_PATH='/results/'+cohort+'/'+algorithm+'/'+algoAbbreviation+sub+'_'+s+'/testing/data'
            for i in range(0, n_trials):
                test_i = 'logs_worker_'+str(6000+i)+'.csv'
                df = pd.read_csv(MAIN_PATH +FOLDER_PATH+ '/'+test_i) #print(df)
                test_i = 'testing_episode_summary_'+str(6000+i)+'.csv'
                df2 = pd.read_csv(MAIN_PATH +FOLDER_PATH+ '/'+test_i) #print(df2)
                try:
                    init_bg = init_bg + df['cgm'][0]
                except:
                    print('incomplete subject log_workers:')
                    print(sub)
                    incomplete = True
                try:
                    init_bg = init_bg + df2['epi'][0]
                except:
                    print('Incomplete subject testing_episode_summary_:')
                    print(sub)
                    incomplete = True
                if incomplete:
                    break
            if incomplete:
                break
    print('Error checking is complete. No errors detected!')

def display_commands_v2(arr):
    n_disp_experiments = 40
    disp_arr = np.array([True] * n_disp_experiments)
    if len(arr) != 0:
        for cmd in arr:
            k = cmd.split()
            if k[0] == 'off' and len(k) == 1:
                disp_arr = np.array([False] * n_disp_experiments)
            elif k[0] == 'off':
                for i in range(1, len(k)):
                    index = int(k[i])
                    disp_arr[index] = False
            elif k[0] == 'on':
                for i in range(1, len(k)):
                    index = int(k[i])
                    disp_arr[index] = True
            elif k[0] == 'only':
                disp_arr = np.array([False] * n_disp_experiments)
                for i in range(1, len(k)):
                    index = int(k[i])
                    disp_arr[index] = True
    return  disp_arr.tolist()


def display_commands(arr):
    disp_arr = [True, True, True, True, True, True, True, True, True, True]
    colors = {'r':0, 'g':1, 'b':2, 'm':3, 'k': 4, 'c':5, 'lime': 6, 'brown': 7, 'orange': 8,
              'pink': 9, 'darkorange2':10, 'darkslateblue': 11}
    if len(arr) != 0:
        for cmd in arr:
            k = cmd.split()
            if k[0] == 'off' and len(k) == 1:
                disp_arr = [False, False, False, False, False, False, False, False, False, False]
            elif k[0] == 'off':
                for i in range(1, len(k)):
                    if k[i].isalpha():
                        id = colors[k[i]]
                    else:
                        id = int(k[i]) - 1
                    disp_arr[id] = False
            elif k[0] == 'on':
                for i in range(1, len(k)):
                    if k[i].isalpha():
                        id = colors[k[i]]
                    else:
                        id = int(k[i]) - 1
                    disp_arr[id] = True
            elif k[0] == 'only':
                disp_arr = [False, False, False, False, False, False, False, False, False, False]
                for i in range(1, len(k)):
                    if k[i].isalpha():
                        id = colors[k[i]]
                    else:
                        id = int(k[i]) - 1
                    disp_arr[id] = True
    return disp_arr


class ExperimentVisualise:
    def __init__(self, id, version=1.0, plot_version=0, test_seeds=None):
        self.MAIN_PATH = '../../results/' + id + '/'
        self.id = id
        self.version = version
        self.plot_version = plot_version
        with open(self.MAIN_PATH + 'args.json') as json_file:
            self.args = json.load(json_file)
        self.training_workers = self.args['n_training_workers']
        self.testing_workers = self.args['n_testing_workers']
        self.experiment_dir = self.args['experiment_dir']
        self.training_seeds = [x for x in range(0, self.training_workers)]

        t_seeds = 500 if test_seeds is None else test_seeds
        self.testing_seeds = [t_seeds+x for x in range(0, self.testing_workers)]
        self.ins_max = self.args['action_scale']
        
    def get_testing_rewards(self, type=None):
        if type == 'normal':
            data = get_normal(self.MAIN_PATH, self.testing_seeds,
                                    '/testing/data/testing_episode_summary_', 'reward')
        elif type == 'min_max':
            data = get_min_max_mean(self.MAIN_PATH, self.testing_seeds,
                                    '/testing/data/testing_episode_summary_', 'reward')
        data['steps'] = np.arange(len(data))
        data['steps'] = (data['steps'] + 1 ) * self.training_workers * self.args['n_step']
        return data

    def get_testing_metric(self, metric, type=None):
        if type == 'normal':
            data = get_normal(self.MAIN_PATH, self.testing_seeds,
                                    '/testing/data/testing_episode_summary_', metric)
        elif type == 'min_max':
            data = get_min_max_mean(self.MAIN_PATH, self.testing_seeds,
                                    '/testing/data/testing_episode_summary_', metric)
        data['steps'] = np.arange(len(data))
        data['steps'] = (data['steps'] + 1 ) * self.training_workers * self.args['n_step']
        return data

    def get_file_paths(self):
        return self.MAIN_PATH, self.testing_seeds, '/testing/data/testing_episode_summary_'

    def get_training_logs(self):
        model_log = pd.read_csv(self.MAIN_PATH + '/model_log.csv')
        model_log['steps'] = np.arange(len(model_log))
        model_log['steps'] = (model_log['steps'] + 1 ) * self.training_workers * self.args['n_step']
        # policy_grad,value_grad,val_loss,exp_var, pi_loss
        return model_log

    def get_aux_training_logs(self):
        model_log = pd.read_csv(self.MAIN_PATH + '/aux_model_log.csv')
        model_log['steps'] = np.arange(len(model_log))
        model_log['steps'] = (model_log['steps'] + 1 )  # * self.training_workers * self.args['n_step']
        # 'vf_aux_grad', 'vf_aux_loss', 'pi_aux_grad', 'pi_aux_loss'
        return model_log

    def get_planning_training_logs(self):
        model_log = pd.read_csv(self.MAIN_PATH + '/planning_model_log.csv')
        model_log['steps'] = np.arange(len(model_log))
        model_log['steps'] = (model_log['steps'] + 1 ) # * self.training_workers * self.args['n_step']
        # 'plan_grad', 'plan_loss'
        return model_log
    
    def get_value_function(self, horizon):
        glucose = get_concat_recent(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'cgm', horizon)
        state_val = get_concat_recent(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'state_val', horizon)
        return glucose, state_val

    def get_test_episode(self, tester, episode):
        if self.version == 1.0:
            df = pd.read_csv(self.MAIN_PATH + '/testing/data/logs_test_worker_' + str(self.testing_seeds[tester]) +'.csv')
        elif self.version == 1.1:
            df = pd.read_csv(self.MAIN_PATH + '/testing/data/logs_worker_' + str(self.testing_seeds[tester]) + '.csv')
        df = df.loc[df['epi'] == episode]
        if self.plot_version == 1.0:
            df['day_hour'] = df['day_hour'].astype(int)
            df['day_min'] = df['day_min'].astype(int)
            df['time'] = np.arange(len(df))
            prev_hour, day = 0, 1
            for i, row in df.iterrows():
                hour = int(row['day_hour'])
                min = int(row['day_min'])
                if prev_hour > hour:
                    day += 1
                df.at[i, 'time'] = datetime(2018, 1, day, hour, min, 0)
                prev_hour = hour
        else:
            df['time'] = np.arange(len(df))
            df['time'] = (df['time'] + 1) * 5

        return df
    
    def get_summary(self):
        print('\n Experiment summary...')
        latest_epi = math.inf
        for tester in self.testing_seeds:
            df = pd.read_csv(self.MAIN_PATH + '/testing/data/testing_episode_summary_' + str(tester) +'.csv')
            print(df.tail(1).to_dict())
            if df['epi'].iloc[-1] < latest_epi:
                latest_epi = df['epi'].iloc[-1]
        return latest_epi

    def get_testing_summary(self):
        arr = []
        target =  ['normo', 'hypo', 'hyper', 'sev_hypo', 'sev_hyper', 'lgbi', 'hgbi', 'ri', 'reward']
        for tester in self.testing_seeds:
            df = pd.read_csv(self.MAIN_PATH + '/testing/data/testing_episode_summary_' + str(tester) +'.csv')
            arr.append(df.iloc[-1:])
        res = pd.concat(arr)
        failures = res[res['t'] < 312].count()['t']
        res = res[res['t'] == 312]
        res = res[target].describe().loc[['mean']]
        res['fail'] = (failures / self.testing_workers) * 100
        target.append('fail')
        res = res[target].round(2)
        #print(res)
        return res

    def get_testing_summary_reward(self):
        arr = []
        target =  ['reward']
        for tester in self.testing_seeds:
            df = pd.read_csv(self.MAIN_PATH + '/testing/data/testing_episode_summary_' + str(tester) +'.csv')
            arr.append(df.iloc[-1:])
        res = pd.concat(arr)
        failures = res[res['t'] < 312].count()['t']
        # res = res[res['t'] == 312]
        res = res[target].describe().loc[['mean']]
        res['fail'] = (failures / self.testing_workers) * 100
        target.append('fail')
        res = res[target].round(2)
        #print(res)
        return res

    def get_training_action_summary(self, horizon):
        mu = get_concat(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'mu', horizon)
        sigma = get_concat(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'sigma', horizon)
        ins = get_concat(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'ins', horizon)
        cgm = get_concat(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'cgm', horizon)
        rl_ins = get_concat(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'rl_ins', horizon)
        return mu, sigma, ins, cgm, rl_ins

    def get_testing_action_summary(self, horizon):
        mu = get_concat(self.MAIN_PATH, self.testing_seeds, 'testing/data/logs_worker_', 'mu', horizon)
        sigma = get_concat(self.MAIN_PATH, self.testing_seeds, 'testing/data/logs_worker_', 'sigma', horizon)
        ins = get_concat(self.MAIN_PATH, self.testing_seeds, 'testing/data/logs_worker_', 'ins', horizon)
        cgm = get_concat(self.MAIN_PATH, self.testing_seeds, 'testing/data/logs_worker_', 'cgm', horizon)
        rl_ins = get_concat(self.MAIN_PATH, self.testing_seeds, 'testing/data/logs_worker_', 'rl_ins', horizon)
        return mu, sigma, ins, cgm, rl_ins

    def get_training_action_summary_V1(self, horizon):
        IS_ratio = get_concat(self.MAIN_PATH, self.training_seeds, 'training/data/logs_worker_', 'IS', horizon)
        return IS_ratio


def plot_training_action_summary(experiment, horizon):
    mu, sigma, ins, cgm, rl_ins = experiment.get_training_action_summary(horizon)
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(321)
    ax1 = fig.add_subplot(322)
    ax1_1 = fig.add_subplot(323)
    ax2 = fig.add_subplot(325)
    ax3 = fig.add_subplot(326)

    ax.hist(mu, bins=20)
    ax1.hist(sigma, bins=20)
    ax1_1.hist(rl_ins, bins=40)
    ax2.hist(ins, bins=20, color='c')
    ax2.set_yscale('log')
    #ax2.yscale('log', nonposy='clip')
    ax3.hist(cgm, bins=20, color='c')

    ax.set_title(r'Network Output ($\mu$)')
    ax1.set_title('Agent Action Sigma')
    ax1_1.set_title('Agent Action (0,5)')
    ax2.set_title('Agent Training - Final Pump Action (0 - action_scale Units)')
    ax3.set_title('Agent Training - Gluocse Distribution (40 - 600 mg/dL)')
    fig.suptitle('Agent Summary Last 100 actions', fontsize=16)
    plt.show()

def plot_training_action_summary_v1(cohort, sub, horizon):
    experiments = cohort[0]
    linear_ins_arr = []
    for experiment in experiments:
        #temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_training_action_summary(horizon)
        temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_testing_action_summary(horizon)
        linear_ins_arr.append(temp_ins)

    experiments = cohort[1]
    quad_ins_arr = []
    for experiment in experiments:
        #temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_training_action_summary(horizon)
        temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_testing_action_summary(horizon)
        quad_ins_arr.append(temp_ins)

    experiments = cohort[2]
    prop_ins_arr = []
    for experiment in experiments:
        #temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_training_action_summary(horizon)
        temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_testing_action_summary(horizon)
        prop_ins_arr.append(temp_ins)

    experiments = cohort[3]
    exp_ins_arr = []
    for experiment in experiments:
        #temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_training_action_summary(horizon)
        temp_mu, temp_sigma, temp_ins, temp_cgm, temp_rl_ins = experiment.get_testing_action_summary(horizon)
        exp_ins_arr.append(temp_ins)

    linear_ins = pd.concat(linear_ins_arr)
    quad_ins = pd.concat(quad_ins_arr)
    prop_ins = pd.concat(prop_ins_arr)
    exp_ins = pd.concat(exp_ins_arr)

    fig = plt.figure(figsize=(16, 3))
    fig.tight_layout(h_pad=2)

    ax = fig.add_subplot(141)
    ax1 = fig.add_subplot(142)
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(144)

    ax.hist(linear_ins, bins=20,  range=[0, 5], color='r')
    ax1.hist(quad_ins, bins=20,  range=[0, 5], color='g')
    ax2.hist(prop_ins, bins=20,  range=[0, 5], color='k')
    ax3.hist(exp_ins, bins=20,  range=[0, 5], color='b')

    ax.set_yscale('log')
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_xlabel('Insulin Pump Range', fontsize=12)

    ax1.set_yscale('log')
    ax1.set_ylabel('Frequency (log scale)', fontsize=12)
    ax1.set_xlabel('Insulin Pump Range', fontsize=12)

    ax2.set_yscale('log')
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_xlabel('Insulin Pump Range', fontsize=12)

    ax3.set_yscale('log')
    ax3.set_ylabel('Frequency (log scale)', fontsize=12)
    ax3.set_xlabel('Insulin Pump Range', fontsize=12)

    ax.set_title('Linear', fontsize=18)
    ax1.set_title('Quadratic', fontsize=18)
    ax2.set_title('Proportional Quadratic', fontsize=18)
    ax3.set_title('Exponential', fontsize=18)
    fig.suptitle(sub, fontsize=20)
    plt.subplots_adjust(top=2)
    plt.tight_layout()
    plt.show()


def plot_training_action_summary_IS(cohort, sub, horizon):
    experiments = cohort[0]
    norm_arr = []
    for experiment in experiments:
        temp = experiment.get_training_action_summary_V1(horizon)
        norm_arr.append(temp)

    experiments = cohort[1]
    is1 = []
    for experiment in experiments:
        temp = experiment.get_training_action_summary_V1(horizon)
        is1.append(temp)

    experiments = cohort[2]
    is2 = []
    for experiment in experiments:
        temp = experiment.get_training_action_summary_V1(horizon)
        is2.append(temp)

    experiments = cohort[3]
    is3 = []
    for experiment in experiments:
        temp = experiment.get_training_action_summary_V1(horizon)
        is3.append(temp)

    norm_arr = pd.concat(norm_arr)
    is1 = pd.concat(is1)
    is2 = pd.concat(is2)
    is3 = pd.concat(is3)

    fig = plt.figure(figsize=(16, 3))
    fig.tight_layout(h_pad=2)

    ax = fig.add_subplot(141)
    ax1 = fig.add_subplot(142)
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(144)

    ax.hist(norm_arr, bins=20, color='r')
    ax1.hist(is1, bins=20, color='g')
    ax2.hist(is2, bins=20, color='k')
    ax3.hist(is3, bins=20, color='b')

    ax.set_yscale('log')
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_xlabel('Insulin Pump Range', fontsize=12)

    ax1.set_yscale('log')
    ax1.set_ylabel('Frequency (log scale)', fontsize=12)
    ax1.set_xlabel('Insulin Pump Range', fontsize=12)

    ax2.set_yscale('log')
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.set_xlabel('Insulin Pump Range', fontsize=12)

    ax3.set_yscale('log')
    ax3.set_ylabel('Frequency (log scale)', fontsize=12)
    ax3.set_xlabel('Insulin Pump Range', fontsize=12)

    ax.set_title('Linear', fontsize=18)
    ax1.set_title('Quadratic', fontsize=18)
    ax2.set_title('Proportional Quadratic', fontsize=18)
    ax3.set_title('Exponential', fontsize=18)
    fig.suptitle(sub, fontsize=20)
    plt.subplots_adjust(top=2)
    plt.tight_layout()
    plt.show()


def get_concat(path, seeds, filename, column, horizon):
    path = path + filename
    full_arr, refined = [], []
    FILES = [ path + str(seed)+'.csv' for seed in seeds]
    for file in FILES:
        d = pd.read_csv(file)
        full_arr.append(d[column][-horizon:])
    data = pd.concat(full_arr, axis=0)
    return data


def plot_episode(experiment, tester, episode):
    df = experiment.get_test_episode(tester, episode)
    fig = plt.figure(figsize=(16, 6))
    #ax = fig.add_subplot(111)
    #ax2 = ax.twinx()

    ax2 = fig.add_subplot(111)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax2.set_yscale('log')
    ax2.set_ylim((1e-3, 5))
    divider = make_axes_locatable(ax2)
    ax = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)
    #ax.set_ylim((5, 600))

    cgm_color = '#000080'
    ins_color = 'mediumseagreen'
    meal_color = '#800000'

    max_ins = max(df['ins'])
    max_cgm = min(max(df['cgm']) + 100, 620) # ,

    ax.plot(df['time'], df['cgm'], markerfacecolor=cgm_color, linewidth=2.0)

    if experiment.plot_version == 1:
        ax2.bar(df['time'], df['ins'], (1/288), color=ins_color)  # width of bar is given in days
    else:
        ax2.plot(df['time'], df['ins'], (1 / 288), color=ins_color)  # width of bar is given in days

    ax.axhline(y=54, color='r', linestyle='--')
    ax.axhspan(70, 180, alpha=0.2, color='limegreen', lw=0)

    x = True
    for t in range(0, len(df)):
        if df.iloc[t]['meal']:
            off_set = (max_cgm - 125) if x else (max_cgm - 75)
            ax.annotate('Carbohydrates: ' + str(df.iloc[t]['meal'])+'g', (df.iloc[t]['time'], off_set), color=meal_color)  #df.iloc[t]['cgm']
            ax.plot((df.iloc[t]['time'], df.iloc[t]['time']), (df.iloc[t]['cgm'], off_set), color=meal_color)

            m_a_min, sample_rate = 20, 5
            m_a_offset = int(m_a_min / sample_rate)
            # if experiment.plot_version == 1:
            #     ax.plot(df.iloc[t]['time'] - timedelta(hours=0, minutes=m_a_min), df.iloc[t-m_a_offset]['cgm'],
            #             marker='D', color='k')
            # else:
            #     ax.plot(df.iloc[t-m_a_offset]['time'], df.iloc[t-m_a_offset]['cgm'], marker='D', color='k')

            x = not(x)

    if experiment.plot_version == 1:
        start_time = df['time'].iloc[0]  # end_time = df['time'].iloc[-1]
        ax2.set_xlim([start_time, start_time + timedelta(hours=24)]) # start_time + timedelta(hours=3)]
        ax2.xaxis.set_minor_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.set_ylim(5, max_cgm)
    var_ins_max = max_ins * (max_cgm / 50)
    #var_ins_max = 1
    #ax2.set_ylim(0, var_ins_max)
    #ax2.set_yscale('log')
    #ax2.set_ylim(1e-4, 10000)
    #ax2.set_yticks(np.arange(0, var_ins_max + 1, 0.5))

    ax.set_ylabel('CGM [mg/dL]', color=cgm_color)
    ax2.set_ylabel('Insulin [U/min]', color=ins_color)
    ax2.set_xlabel('Time (hrs)')
    #ax.set_title('Simulation: Glucose Regulation')
    ax.grid()

    cgm_line = mlines.Line2D([], [], color=cgm_color, label='CGM (Sensor: GuardianRT)')
    ins_line = mlines.Line2D([], [], color=ins_color, label='Insulin (Pump: Insulet)')
    meal_ann_line = mlines.Line2D([], [], color='k', marker='D', linestyle='None', label='Meal Announcement (20min)')
    ax.legend(handles=[cgm_line, ins_line], loc='upper right')  # meal_ann_line

    ax.axhline(y=250, color='r', linestyle='--')

    # ax.text(df.iloc[1]['time'], 260, 'Severe Hyperglycemia', size=12, color='r')
    # ax.text(df.iloc[1]['time'], 230, 'Hyperglycemia', size=12)
    # ax.text(df.iloc[1]['time'], 100, 'Normoglcemia', size=12, color=cgm_color)
    # ax.text(df.iloc[1]['time'], 54, 'Hypoglycemia', size=12)
    # ax.text(df.iloc[1]['time'], 30, 'Severe Hypoglycemia', size=12, color='r')

    #fig.savefig(experiment.experiment_dir +'/'+ str(tester))

    plt.show()


def plot_episode_new(experiment, tester, episode):
    df = experiment.get_test_episode(tester, episode)
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    #ax2 = ax.twinx()

    #ax2 = fig.add_subplot(111)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #ax2.set_yscale('log')
    #ax2.set_ylim((1e-3, 5))
    #divider = make_axes_locatable(ax2)
    #ax = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)
    #ax.set_ylim((5, 600))

    cgm_color = '#000080'
    ins_color = 'mediumseagreen'
    meal_color = '#800000'

    max_ins = max(df['ins'])
    max_cgm = min(max(df['cgm']) + 100, 620) # ,

    ax.plot(df['time'], df['cgm'], markerfacecolor=cgm_color, linewidth=2.0)

    # if experiment.plot_version == 1:
    #     ax2.bar(df['time'], df['ins'], (1/288), color=ins_color)  # width of bar is given in days
    # else:
    #     ax2.plot(df['time'], df['ins'], (1 / 288), color=ins_color)  # width of bar is given in days

    ax.axhline(y=54, color='r', linestyle='--')
    ax.axhspan(70, 180, alpha=0.2, color='limegreen', lw=0)

    x = True
    for t in range(0, len(df)):
        if df.iloc[t]['meal']:
            off_set = (max_cgm - 125) if x else (max_cgm - 75)
            ax.annotate('Carbohydrates: ' + str(df.iloc[t]['meal'])+'g', (df.iloc[t]['time'], off_set), color=meal_color, size=12)  #df.iloc[t]['cgm']
            ax.plot((df.iloc[t]['time'], df.iloc[t]['time']), (df.iloc[t]['cgm'], off_set), color=meal_color)

            m_a_min, sample_rate = 20, 5
            m_a_offset = int(m_a_min / sample_rate)
            # if experiment.plot_version == 1:
            #     ax.plot(df.iloc[t]['time'] - timedelta(hours=0, minutes=m_a_min), df.iloc[t-m_a_offset]['cgm'],
            #             marker='D', color='k')
            # else:
            #     ax.plot(df.iloc[t-m_a_offset]['time'], df.iloc[t-m_a_offset]['cgm'], marker='D', color='k')

            x = not(x)

    if experiment.plot_version == 1:
        start_time = df['time'].iloc[0]  # end_time = df['time'].iloc[-1]
        ax.set_xlim([start_time, start_time + timedelta(hours=24)]) # start_time + timedelta(hours=3)]
        ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.set_ylim(0, max_cgm)
    var_ins_max = max_ins * (max_cgm / 50)
    #var_ins_max = 1
    #ax2.set_ylim(0, var_ins_max)
    #ax2.set_yscale('log')
    #ax2.set_ylim(1e-4, 10000)
    #ax2.set_yticks(np.arange(0, var_ins_max + 1, 0.5))

    ax.set_ylabel('CGM [mg/dL]', color=cgm_color)
    #ax2.set_ylabel('Insulin [U/min]', color=ins_color)
    ax.set_xlabel('Time (hrs)')
    #ax.set_title('Simulation: Glucose Regulation')
    ax.grid()

    cgm_line = mlines.Line2D([], [], color=cgm_color, label='CGM (Sensor: GuardianRT)')
    ins_line = mlines.Line2D([], [], color=ins_color, label='Insulin (Pump: Insulet)')
    meal_ann_line = mlines.Line2D([], [], color='k', marker='D', linestyle='None', label='Meal Announcement (20min)')
    ax.legend(handles=[cgm_line], loc='upper right')  # meal_ann_line

    ax.axhline(y=250, color='r', linestyle='--')

    ax.text(df.iloc[1]['time'], 260, 'Severe Hyperglycemia (TAR Level 2)', size=12, color='r')
    ax.text(df.iloc[1]['time'], 230, 'Hyperglycemia (TAR Level 1)', size=12)
    ax.text(df.iloc[1]['time'], 100, 'Normoglycemia (TIR)', size=12, color=cgm_color)
    ax.text(df.iloc[1]['time'], 58, 'Hypoglycemia (TBR Level 1)', size=12)
    ax.text(df.iloc[1]['time'], 30, 'Severe Hypoglycemia (TBR Level 2)', size=12, color='r')
    ax.set_yticks([0, 54, 70, 180, 250, 400])

    #fig.savefig(experiment.experiment_dir +'/'+ str(tester))

    plt.show()


def plot_episode_dynamic(experiment, tester, episode, window):
    df = experiment.get_test_episode(tester, episode)
    fig = plt.figure(figsize=(16, 6))
    #ax = fig.add_subplot(111)
    #ax2 = ax.twinx()

    ax2 = fig.add_subplot(111)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax2.set_yscale('log')
    ax2.set_ylim((1e-3, 5))
    divider = make_axes_locatable(ax2)
    ax = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)
    #ax.set_ylim((5, 600))

    cgm_color = '#000080'
    ins_color = 'mediumseagreen'
    meal_color = '#800000'

    max_ins = max(df['ins'])
    max_cgm = min(max(df['cgm']) + 100, 620) # ,

    dfRange = df.iloc[0:window]
    ax.plot(dfRange['time'], dfRange['cgm'], markerfacecolor=cgm_color, linewidth=2.0)

    if experiment.plot_version == 1:
        ax2.bar(dfRange['time'], dfRange['ins'], (1/288), color=ins_color)  # width of bar is given in days
    else:
        ax2.plot(dfRange['time'], dfRange['ins'], (1 / 288), color=ins_color)  # width of bar is given in days

    ax.axhline(y=50, color='r', linestyle='--')
    ax.axhspan(70, 180, alpha=0.2, color='limegreen', lw=0)

    x = True
    for t in range(0, window): #len(df)
        if df.iloc[t]['meal']:
            off_set = (max_cgm - 125) if x else (max_cgm - 75)
            ax.annotate('Carbohydrates: ' + str(df.iloc[t]['meal'])+'g', (df.iloc[t]['time'], off_set), color=meal_color)  #df.iloc[t]['cgm']
            ax.plot((df.iloc[t]['time'], df.iloc[t]['time']), (df.iloc[t]['cgm'], off_set), color=meal_color)

            m_a_min, sample_rate = 20, 5
            m_a_offset = int(m_a_min / sample_rate)
            # if experiment.plot_version == 1:
            #     ax.plot(df.iloc[t]['time'] - timedelta(hours=0, minutes=m_a_min), df.iloc[t-m_a_offset]['cgm'],
            #             marker='D', color='k')
            # else:
            #     ax.plot(df.iloc[t-m_a_offset]['time'], df.iloc[t-m_a_offset]['cgm'], marker='D', color='k')

            x = not(x)

    if experiment.plot_version == 1:
        start_time = df['time'].iloc[0]  # end_time = df['time'].iloc[-1]
        ax2.set_xlim([start_time, start_time + timedelta(hours=24)]) # start_time + timedelta(hours=3)]
        ax2.xaxis.set_minor_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.set_ylim(5, max_cgm)
    var_ins_max = max_ins * (max_cgm / 50)
    #var_ins_max = 1
    #ax2.set_ylim(0, var_ins_max)
    #ax2.set_yscale('log')
    #ax2.set_ylim(1e-4, 10000)
    #ax2.set_yticks(np.arange(0, var_ins_max + 1, 0.5))

    ax.set_ylabel('CGM [mg/dL]', color=cgm_color)
    ax2.set_ylabel('Insulin [U/min]', color=ins_color)
    ax2.set_xlabel('Time (hrs)')
    ax.set_title('Simulation: Glucose Regulation')
    ax.grid()

    cgm_line = mlines.Line2D([], [], color=cgm_color, label='CGM (Sensor: GuardianRT)')
    ins_line = mlines.Line2D([], [], color=ins_color, label='Insulin (Pump: Insulet)')
    meal_ann_line = mlines.Line2D([], [], color='k', marker='D', linestyle='None', label='Meal Announcement (20min)')
    ax.legend(handles=[cgm_line, ins_line], loc='upper right')  # meal_ann_line

    ax.axhline(y=300, color='r', linestyle='--')
    # ax.text(df.iloc[1]['time'], 310, 'Severe Hyperglycemia', size=12, color='r')
    # ax.text(df.iloc[1]['time'], 280, 'Hyperglycemia', size=12)
    # ax.text(df.iloc[1]['time'], 100, 'Normoglcemia', size=12, color=cgm_color)
    # ax.text(df.iloc[1]['time'], 54, 'Hypoglycemia', size=12)
    # ax.text(df.iloc[1]['time'], 30, 'Severe Hypoglycemia', size=12, color='r')

    #fig.savefig(experiment.experiment_dir +'/'+ str(tester))

    plt.show()



def plot_value_function(experiment, horizon):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    data_glucose, data_state_val = experiment.get_value_function(horizon)
    ax.scatter(data_glucose, data_state_val)
    ax.set_title('Debugging Value Function')
    ax.set_xlabel('Glucose (Last glucose value of input state space)')
    ax.set_ylabel('V(s)')
    plt.show()

    
def plot_testing_rewards(dict, type, dis_len):
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    for exp in dict:
        if dict[exp]['show']:
            d1 = dict[exp]['id'].get_testing_rewards(type=type)
            ax.plot(d1['steps'], d1['mean'], color=dict[exp]['color'], label=dict[exp]['id'].id)
            ax.fill_between(d1['steps'], d1['min'], d1['max'], color=dict[exp]['color'], alpha=0.2)
    ax.axhline(y=288, color='k', linestyle='--')
    ax.set_title('Rewards')
    ax.legend(loc="upper right")
    ax.set_ylabel('Total Reward')
    ax.set_xlabel('Interactions')
    ax.grid()
    ax.set_xlim(0, dis_len)
    #ax.set_ylim(0, 1000000)
    plt.show()


def plot_testing_metric(dict, type, dis_len, metric, goal, fill, label=False):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    for exp in dict:
        if dict[exp]['show']:
            d1 = dict[exp]['id'].get_testing_metric(metric=metric, type=type)
            l = dict[exp]['label'] #if label else dict[exp]['id'].id
            ax.plot(d1['steps'], d1['mean'], color=dict[exp]['color'], label=l)
            if fill:
                ax.fill_between(d1['steps'], d1['min'], d1['max'], color=dict[exp]['color'], alpha=0.2)
    ax.axhline(y=goal, color='k', linestyle='--')
    ax.set_title(metric)
    ax.legend(loc="lower right")
    ax.set_ylabel(metric)
    ax.set_xlabel('Interactions')
    ax.grid()
    ax.set_xlim(0, dis_len)
    #ax.set_ylim(0, 1000000)
    plt.show()

def plot_testing_average_metric(dict, groups, type, dis_len, metric, goal, fill, title=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    for groupings in range(0, len(groups)):
        FILES = []
        for i in groups[groupings]:  # exp's inside the group
            # will give the exp list
            path, seeds, filename = dict[i]['id'].get_file_paths()
            FILES = create_file_paths(path, seeds, filename, FILES)
        cur_length, full_arr, refined = [], [], []
        for file in FILES:
            reward_summary = pd.read_csv(file)
            cur_length.append(reward_summary.shape[0])
            full_arr.append(reward_summary[metric])
        for x in full_arr:
            refined.append(x[0:min(cur_length)])
        data = pd.concat(refined, axis=1)
        data['mean'] = data.mean(axis=1)

        if type == 'normal':
            data['std_dev'] = data.std(axis=1)
            data['max'] = data['mean'] + data['std_dev']  # * 2
            data['min'] = data['mean'] - data['std_dev']  # * 2
        else:
            data['max'] = data.max(axis=1)
            data['min'] = data.min(axis=1)

        data['steps'] = np.arange(len(data))
        data['steps'] = (data['steps'] + 1) * dict[i]['id'].training_workers * dict[i]['id'].args['n_step']

        ax.plot(data['steps'], data['mean'], color=dict[i]['color'], label=dict[i]['label'])
        if fill:
            ax.fill_between(data['steps'], data['min'], data['max'], color=dict[i]['color'], alpha=0.1)

    #ax.axhline(y=goal, color='k', linestyle='--')

    graph_title =  title if title is not None else 'Average Rewards (Multiple Seeds)'
    ax.set_title(graph_title, fontsize=32)
    # ax.legend(loc="upper left", fontsize=16)
    ax.set_ylabel('Total Reward', fontsize=24) #ax.set_ylabel(metric)
    ax.set_xlabel('Steps', fontsize=24)
    ax.grid()
    ax.set_xlim(0, dis_len)
    ax.set_ylim(0, 320)
    plt.show()


def timing_analysis(dict, groups, metric):
    data = []
    for groupings in range(0, len(groups)):
        FILES = []
        for i in groups[groupings]:  # exp's inside the group
            # will give the exp list
            path, seeds, filename = dict[i]['id'].get_file_paths()
            FILES = create_file_paths(path, seeds, filename, FILES)
        cur_length, full_arr, refined = [], [], []
        for file in FILES:
            reward_summary = pd.read_csv(file)
            cur_length.append(reward_summary.shape[0])
            full_arr.append(reward_summary[metric])
        for x in full_arr:
            refined.append(x[0:min(cur_length)])
        data = pd.concat(refined, axis=1)
        data['mean'] = data.mean(axis=1)
        data['steps'] = np.arange(len(data))
        data['steps'] = (data['steps'] + 1) * dict[i]['id'].training_workers * dict[i]['id'].args['n_step']

        # timing analysis
        data['reward_percent'] = (data['mean'] / 288) * 100
        data['reward_percent'] = data['reward_percent'].round(2)

    return data

def plot_training_logs(mode, exp_dict, dis_len, params, cols=1):
    fig = plt.figure(figsize=(16, 26))
    subplots = []
    tot_plots = len(params)

    for i in range(0, tot_plots):
        subplots.append(fig.add_subplot(math.ceil(tot_plots/cols), cols, i+1))
        subplots[i].grid(True)
        subplots[i].set_xlim(0, dis_len)
        subplots[i].set_title(params[i])

    for exp in exp_dict:
        if exp_dict[exp]['show']:
            if mode == 'ppo':
                # ['exp_var', 'true_var','val_loss', 'policy_grad', 'value_grad', 'pi_loss', 'avg_rew']
                d1 = exp_dict[exp]['id'].get_training_logs()
            elif mode == 'aux':
                # ['pi_aux_loss', 'vf_aux_loss', 'pi_aux_grad', 'vf_aux_grad']
                d1 = exp_dict[exp]['id'].get_aux_training_logs()
            elif mode == 'planning':
                # ['plan_grad', 'plan_loss']
                d1 = exp_dict[exp]['id'].get_planning_training_logs()
            else:
                d1 = 0
                print('Invalid mode selected')
                exit()
            for j in range(0, tot_plots):
                subplots[j].plot(d1['steps'], d1[params[j]], color=exp_dict[exp]['color'], label=exp_dict[exp]['id'].id)
    for i in range(0, tot_plots):
        subplots[i].legend(loc="upper right")
    plt.show()


def create_file_paths(path, seeds, filename, FILES):
    for seed in seeds:
        FILES.append(path + filename + str(seed)+'.csv')
    return FILES


def get_min_max_mean(path, seeds, filename, column):
    cur_length, full_arr, refined = [], [], []
    FILES = [path + filename + str(seed)+'.csv' for seed in seeds]
    for file in FILES:
        reward_summary = pd.read_csv(file)
        cur_length.append(reward_summary.shape[0])
        full_arr.append(reward_summary[column])
    for x in full_arr:
        refined.append(x[0:min(cur_length)])
    data = pd.concat(refined, axis=1)
    data['mean'] = data.mean(axis=1)
    data['max'] = data.max(axis=1)
    data['min'] = data.min(axis=1)
    return data


def get_normal(path, seeds, filename, column):
    cur_length, full_arr, refined = [], [], []
    FILES = [path + filename + str(seed)+'.csv' for seed in seeds]
    for file in FILES:
        reward_summary = pd.read_csv(file)
        cur_length.append(reward_summary.shape[0])
        full_arr.append(reward_summary[column])
    for x in full_arr:
        refined.append(x[0:min(cur_length)])
    data = pd.concat(refined, axis=1)
    data['mean'] = data.mean(axis=1)
    data['std_dev'] = data.std(axis=1)
    data['max'] = data['mean'] +  data['std_dev'] # * 2
    data['min'] = data['mean'] -  data['std_dev'] # * 2
    return data


def get_concat_recent(path, seeds, filename, column, horizon):
    full_arr, refined = [], []
    FILES = [path + filename + str(seed)+'.csv' for seed in seeds]
    for file in FILES:
        d = pd.read_csv(file)
        full_arr.append(d[column][-horizon:])
    data = pd.concat(full_arr, axis=0)
    return data



