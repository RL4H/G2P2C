def set_args(args):
    #args.action_type = 'exponential'  # 'normal', quadratic, proportional_quadratic, exponential
    args.feature_history = 12
    args.calibration = 12
    args.action_scale = 5
    args.insulin_max = 5
    args.n_features = 2
    args.t_meal = 20
    args.use_meal_announcement = False  # adds meal announcement as a timeseries feature.
    args.use_carb_announcement = False
    args.use_tod_announcement = False
    args.use_handcraft = 0
    args.n_handcrafted_features = 1
    args.n_hidden = 16
    args.n_rnn_layers = 1
    args.rnn_directions = 1
    args.bidirectional = False
    args.max_epi_length = 288 * 10
    args.n_step = 256
    args.max_test_epi_len = 288
    #args.return_type = 'average'   # discount | average
    args.gamma = 1 if args.return_type == 'average' else 0.99
    args.lambda_ = 1 if args.return_type == 'average' else 0.95
    args.entropy_coef = 0.001
    args.grad_clip = 20
    args.eps_clip = 0.1
    args.target_kl = 0.01
    args.normalize_reward = True
    args.shuffle_rollout = True
    args.n_training_workers = 32
    args.n_testing_workers = 32
    args.n_pi_epochs = 5
    args.n_vf_epochs = 5
    args.pi_lr = 1e-4 * 3
    args.vf_lr = 1e-4 * 3
    args.batch_size = 1024

    # aux model learning
    args.aux_buffer_max = 35000
    args.aux_frequency = 1  # frequency of updates
    args.aux_vf_coef = 0.01
    args.aux_pi_coef = 0.01
    args.aux_batch_size = 1024
    args.n_aux_epochs = 5
    args.aux_lr = 1e-4 * 3

    # plannning
    args.planning_n_step = 6
    args.n_planning_simulations = 50
    args.plan_batch_size = 1024
    args.n_plan_epochs = 1
    #args.planning_lr = 1e-4 * 3

    return args