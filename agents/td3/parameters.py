def set_args(args):
    args.action_type = 'exponential'  # 'normal', quadratic, proportional_quadratic, exponential
    args.feature_history = 12 #24
    args.calibration = 12  #24
    args.action_scale = 5  # 5/factor
    args.insulin_max = 5
    # these params adjust the pump.
    args.expert_bolus = False
    args.expert_cf = False
    args.n_features = 2
    args.t_meal = 20
    args.use_meal_announcement = False  # adds meal announcement as a timeseries feature.
    args.use_carb_announcement = False
    args.use_tod_announcement = False
    args.use_handcraft = 0
    args.n_handcrafted_features = 1
    args.n_hidden = 16 # 128
    args.n_rnn_layers = 1
    args.rnn_directions = 1
    args.bidirectional = False
    args.rnn_only = True  # RNN + 1 Dense layer, deprecated and archietcure fixed
    args.max_epi_length = 288 * 10
    args.n_step = 256
    args.max_test_epi_len = 288
    args.gamma = 0.997
    # Parameters above this line are kept fixed! for consistency between other RL algorithms.

    # parameters important to SAC algo
    args.entropy_coef = 0.001  # 0.001 seems to work
    args.batch_size = 256 if args.debug == 0 else 64  # the mini_batch size
    args.replay_buffer_size = 100000 if args.debug == 0 else 1024   # total <s,a,r,s'> pairs 100000
    args.sample_size = 4096 if args.debug == 0 else 128  #256

    args.sac_v2 = True

    # 200 worked out, 400 runnning

    args.shuffle_rollout = True
    args.n_training_workers = 16 if args.debug == 0 else 2
    args.n_testing_workers = 20 if args.debug == 0 else 2
    args.n_pi_epochs = 5  # can be used to increase number of epochs for all networks updates.
    # args.pi_lr = 1e-4 * 3  # 1e-4 * 3
    # args.vf_lr = 1e-4 * 3  # 1e-4 * 3
    args.grad_clip = 20

    ### todo: refctaor - unused below
    args.eps_clip = 0.1  # 0.05 #0.1  # (Usually small, 0.1 to 0.3.) 0.2
    args.target_kl = 0.01  # 0.005 #0.01  # (Usually small, 0.01 or 0.05.)
    args.normalize_reward = True
    args.reward_lr = 1 * 1e-3
    args.aux_lr = 1e-4 * 3
    args.n_vf_epochs = 1  # FIXED
    args.aux_batch_size = 1024

    # (2) => aux model learning
    args.n_aux_epochs = 5
    args.aux_frequency = 1  # frequency of updates
    args.aux_vf_coef = 0.01 #10 #1 #
    args.aux_pi_coef = 0.01 #10 #1 #
    # (3) = > plannning
    #args.planning_coef = 1
    args.planning_lr = 1e-4 * 3
    args.kl = 1
    args.use_planning = False #if args.planning_coef == -1 else True
    args.planning_n_step = 6
    args.plan_type = 4
    args.n_planning_simulations = 50
    args.plan_batch_size = 1024
    args.n_plan_epochs = 1
    # clean up below: deprecated
    args.bgp_pred_mode = False
    args.n_bgp_steps = 0  # todo: this is fixed, need to be changed manually -> fix

    return args
