import argparse
from decouple import config
import sys
MAIN_PATH = config('MAIN_PATH')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--agent', type=str, default='ppo', help='agent used for the experiment.')
        self.parser.add_argument('--restart', type=str, default='1', help='')
        self.parser.add_argument('--m', type=str, default='', help='message about the experiment')
        self.parser.add_argument('--device', type=str, default='cuda', help='cpu | cuda | give device name')
        self.parser.add_argument('--verbose', type=bool, default=True, help='')
        self.parser.add_argument('--seed', type=int, default=0, help='')
        self.parser.add_argument('--debug', type=int, default=0, help='if debug ON => 1')
        self.parser.add_argument('--kl', type=int, default=1, help='if debug ON => 1')  # experimenting KL implementation

        # directories
        self.parser.add_argument('--folder_id', type=str, default='testing', help='folder path for results and log')
        self.parser.add_argument('--main_dir', type=str, default=MAIN_PATH, help='main path given from .env file')
        self.parser.add_argument('--experiment_dir', type=str, default='', help='path where results will be saved')

        # simulation
        self.parser.add_argument('--patient_id', type=int, default=0,
                                 help='patient_id = [adolescent child adults] hence 0 - 9 indexes adolescents likewise')
        self.parser.add_argument('--sensor', type=str, default='GuardianRT', help='Dexcom, GuardianRT, Navigator')
        self.parser.add_argument('--pump', type=str, default='Insulet', help='Insulet, Cozmo')
        # simulator backend
        self.parser.add_argument('--sim', type=str, default='simglucose', choices=['simglucose', 'dmms'],
                                 help='Simulation environment: simglucose or dmms')
        self.parser.add_argument('--dmms_exe', type=str, default='', help='Path to DMMS.R executable')
        self.parser.add_argument('--dmms_cfg', type=str, default='', help='Path to DMMS.R config XML')
        self.parser.add_argument('--dmms_server', type=str, default='http://127.0.0.1:5000',
                                 help='DMMS environment FastAPI server URL')

        # for training: # ideal benchmark adult and adolescent doesnt have snacks though => set prob '-1' to remove
        self.parser.add_argument('--meal_prob', type=list, default=[0.95, -1, 0.95, -1, 0.95, -1], help='')
        self.parser.add_argument('--meal_amount', type=list, default=[45, 30, 85, 30, 80, 30], help='')
        self.parser.add_argument('--meal_variance', type=list, default=[5, 3, 5, 3, 10, 3], help='')
        self.parser.add_argument('--time_variance', type=list, default=[60, 30, 60, 30, 60, 30], help='in mins')

        # insulin action limits
        self.parser.add_argument('--action_type', type=str, default='exponential',
                                 help='normal, quadratic, proportional_quadratic, exponential, sparse')
        self.parser.add_argument('--action_scale', type=int, default=1, help='This is the max insulin')
        self.parser.add_argument('--insulin_max', type=int, default=5, help='')
        self.parser.add_argument('--insulin_min', type=int, default=0, help='')
        self.parser.add_argument('--glucose_max', type=int, default=600, help='')  # the sensor range would affect this
        self.parser.add_argument('--glucose_min', type=int, default=39, help='')

        # algorithm training settings.
        self.parser.add_argument('--target_glucose', type=float, default=140, help='target glucose')  # param for pid
        self.parser.add_argument('--use_bolus', type=bool, default=True, help='')  # param for BB
        self.parser.add_argument('--use_cf', type=bool, default=False, help='')  # param for BB
        self.parser.add_argument('--glucose_cf_target', type=float, default=150, help='glucose correction target')  # param for BB
        self.parser.add_argument('--expert_bolus', type=bool, default=False, help='')
        self.parser.add_argument('--expert_cf', type=bool, default=False, help='')
        self.parser.add_argument('--use_meal_announcement', type=bool, default=True, help='')
        self.parser.add_argument('--use_carb_announcement', type=bool, default=True, help='')
        self.parser.add_argument('--carb_estimation_method', type=str, default='real', help='linear, quadratic, real, rand')
        self.parser.add_argument('--use_tod_announcement', type=bool, default=True, help='')
        self.parser.add_argument('--t_meal', type=int, default=20,  # 20 is what i use here
                                 help='if zero, assume no announcmeent; announce meal x min before, '
                                      'Optimal prandial timing of bolus insulin in diabetes management: a review,'
                                      'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836969/')

        # Actor / Critic Network params.
        self.parser.add_argument('--n_features', type=int, default=3, help='# of features in the state space')
        self.parser.add_argument('--n_handcrafted_features', type=int, default=0, help='')
        self.parser.add_argument('--use_handcraft', type=int, default=0, help='')
        self.parser.add_argument('--feature_history', type=int, default=48, help='')
        self.parser.add_argument('--calibration', type=int, default=48, help='should be same as feature_hist')
        self.parser.add_argument('--max_epi_length', type=int, default=2000, help='')  # 30days, 5 min, 8640
        self.parser.add_argument('--n_action', type=int, default=1, help='number of control actions')
        self.parser.add_argument('--n_hidden', type=int, default=12, help='hidden units in lstm')
        self.parser.add_argument('--n_rnn_layers', type=int, default=2, help='layers in the lstm')
        self.parser.add_argument('--rnn_directions', type=int, default=1, help='')
        self.parser.add_argument('--rnn_only', type=bool, default=False, help='')
        self.parser.add_argument('--bidirectional', type=bool, default=False, help='')
        self.parser.add_argument('--n_step', type=int, default=6, help='n step TD learning, consider selected sensor!')
        self.parser.add_argument('--gamma', type=float, default=0.99, help='1 if continous')
        self.parser.add_argument('--lambda_', type=float, default=0.95, help='')
        self.parser.add_argument('--max_test_epi_len', type=int, default=1, help='n time max ep trained.')



        # ppo params
        self.parser.add_argument('--eps_clip', type=float, default=0.2, help=' (Usually small, 0.1 to 0.3.) 0.2')
        self.parser.add_argument('--n_vf_epochs', type=int, default=80, help='')
        self.parser.add_argument('--n_pi_epochs', type=int, default=80, help='')
        self.parser.add_argument('--target_kl', type=float, default=0.05, help='# (Usually small, 0.01 or 0.05.)')
        self.parser.add_argument('--pi_lr', type=float, default=1e-3, help='')
        self.parser.add_argument('--vf_lr', type=float, default=1e-3, help='')
        self.parser.add_argument('--batch_size', type=int, default=64, help='')
        self.parser.add_argument('--n_training_workers', type=int, default=20, help='')
        self.parser.add_argument('--n_testing_workers', type=int, default=5, help='')
        self.parser.add_argument('--entropy_coef', type=float, default=0.01, help='')
        self.parser.add_argument('--grad_clip', type=float, default=20, help='')
        self.parser.add_argument('--normalize_reward', type=bool, default=False, help='')
        self.parser.add_argument('--shuffle_rollout', type=bool, default=False, help='')
        self.parser.add_argument('--return_type', type=str, default='average', help='discount | average')

        # auxiliary model learning
        self.parser.add_argument('--aux_mode', type=str, default='dual', help='off, vf_only, pi_only, dual')
        self.parser.add_argument('--aux_lr', type=float, default=1e-4, help='')
        self.parser.add_argument('--aux_buffer_max', type=int, default=10, help='how many iterations of past data')
        self.parser.add_argument('--aux_frequency', type=int, default=3, help='')
        self.parser.add_argument('--n_aux_epochs', type=int, default=3, help='')
        self.parser.add_argument('--aux_batch_size', type=int, default=100, help='')
        self.parser.add_argument('--aux_vf_coef', type=float, default=1, help='')
        self.parser.add_argument('--aux_pi_coef', type=float, default=1, help='')

        # planning phase
        self.parser.add_argument('--use_planning', type=str, default='no', help='no, yes')
        self.parser.add_argument('--planning_n_step', type=int, default=3, help='')
        self.parser.add_argument('--n_planning_simulations', type=int, default=5, help='')
        self.parser.add_argument('--n_plan_epochs', type=int, default=1, help='')
        self.parser.add_argument('--plan_batch_size', type=int, default=1, help='')
        self.parser.add_argument('--planning_lr', type=float, default=1e-4, help='')

        # deprecated todo: refactor
        self.parser.add_argument('--bgp_pred_mode', type=bool, default=False, help='future bg prediction')
        self.parser.add_argument('--n_bgp_steps', type=int, default=0, help='future eprediction horizon')
        self.parser.add_argument('--pretrain_period', type=int, default=5760, help='')

        # sac - 2023 implementation
        # self.parser.add_argument('--soft_tau', type=float, default=0.005, help='')
        #self.parser.add_argument('--replay_buffer_size', type=int, default=1000, help='')
        self.parser.add_argument('--sample_size', type=int, default=1000, help='')
        self.parser.add_argument('--sac_v2', type=bool, default=False, help='')

        self.parser.add_argument('--discrete_actions', type=bool, default=False, help='')
        # self.parser.add_argument('--n_discrete_actions', type=int, default=50, help='')

        # DDPG - 2024 implementation
        self.parser.add_argument('--noise_model', type=str, default='normal_dist',
                                 help='Noise model for applying exploratory noise to policy')
        self.parser.add_argument('--noise_application', type=int, default=1,
                                 help='Noise application method for policy exploration')
        self.parser.add_argument('--noise_std', type=float, default=0.2,
                                 help='Standard deviation for exploratory noise')
        self.parser.add_argument('--soft_tau', type=float, default=0.005, help='Tau for soft update')

        self.parser.add_argument('--mu_penalty', type=int, default=1, help='Penalty applied to mu during policy optimisation -> 1 for penalty applied, 0 for no penalty')
        self.parser.add_argument('--action_penalty_limit', type=float, default=0, help='Limit of action during policy optimisation')
        self.parser.add_argument('--action_penalty_coef', type=float, default=0.1, help='Policy penalty term coefficient')

        self.parser.add_argument('--replay_buffer_type', type=str, default="random", help='type of replay buffer')
        self.parser.add_argument('--replay_buffer_alpha', type=float, default=0.6, help='Replay buffer alpha')
        self.parser.add_argument('--replay_buffer_beta', type=float, default=0.4, help='Replay buffer beta')
        self.parser.add_argument('--replay_buffer_temporal_decay', type=float, default=1, help='Replay buffer discount factor')

        self.parser.add_argument('--target_action_std', type=float, default=0.2, help='Target action noise level for TD3')
        self.parser.add_argument('--target_action_lim', type=float, default=0.5, help='Target action noise limit for TD3')

        self.parser.add_argument('--fine_tune_from_checkpoint', type=int, default=195, help='fine-tuning checkpoint - from')


        # self.parser.add_argument('--pi_lr', type=float, default=1e-4 * 3, help='Policy learning rate')
        # self.parser.add_argument('--vf_lr', type=float, default=1e-4 * 3, help='Value function learning rate')


        # fixed "HARD" benchmark scenario
        # self.parser.add_argument('--meal_prob', type=list, default=[1, -1, 1, -1, 1, -1], help='')
        # self.parser.add_argument('--meal_amount', type=list, default=[45, 30, 85, 30, 80, 30], help='')
        # self.parser.add_argument('--meal_variance', type=list, default=[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8], help='')
        # self.parser.add_argument('--time_variance', type=list, default=[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8], help='in mins')

        # Fixed "EASY" benchmark scenario: manual control SBB
        # self.parser.add_argument('--meal_prob', type=list, default=[1, -1, 1, -1, 1, -1], help='')
        # self.parser.add_argument('--meal_amount', type=list, default=[40, 20, 80, 10, 60, 30], help='')
        # self.parser.add_argument('--meal_variance', type=list, default=[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8], help='')
        # self.parser.add_argument('--time_variance', type=list, default=[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8], help='in mins')

        # parameters for children
        # self.parser.add_argument('--meal_amount', type=list, default=[30, 15, 45, 15, 45, 15], help='')
        # self.parser.add_argument('--meal_variance', type=list, default=[5, 3, 5, 3, 5, 3], help='')

        # Elena paper meal: Breakfast 30-60g, Lunch 70-100g, Dinner 70-110g, Snack 20-40g
        # June9 exp run: Breakfast 45 (10), Lunch 100 (10, Dinner 90 (1), Snack 10(5)


    # parse 메소드 시그니처 변경: args_list=None 추가
    def parse(self, args_list=None):
        self._initial()

        # args_list가 명시적으로 주어졌는지 확인
        if args_list is not None:
            # 주어졌다면 (빈 리스트 포함), 해당 리스트로 파싱
            args_to_parse = args_list
        else:
            # 주어지지 않았다면 (기본값 None), 기존처럼 sys.argv 사용
            # (주의: 이 경우에도 pytest 등 다른 환경 고려가 필요하다면 추가 조건 가능)
            args_to_parse = sys.argv[1:]

        self.opt = self.parser.parse_args(args_to_parse) # 결정된 인자 리스트로 파싱
        Options.validate_args(self.opt)
        return self.opt

    @staticmethod
    def validate_args(args):
        valid = True
        if args.feature_history != args.calibration:
            valid = False
        if args.t_meal > 0 and args.n_features < 3:
            print("Error: Meal Announcement True, but its not reflected in the state space!")
            valid = False
        # todo: patinet_id vs patient_type mismatch.
        if not valid:
            print("Check the input arguments!")
            exit()
