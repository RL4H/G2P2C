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


def set_args_dmms_debug(args):
    """
    DMMS.R 단일 시나리오 최적화를 위한 디버깅용 하이퍼파라미터
    """
    # 기본 G2P2C 설정 유지
    args.feature_history = 12
    args.calibration = 12
    args.action_scale = 5
    args.insulin_max = 5
    args.n_features = 2
    args.t_meal = 20
    args.use_meal_announcement = False
    args.use_carb_announcement = False
    args.use_tod_announcement = False
    args.use_handcraft = 0
    args.n_handcrafted_features = 1
    args.n_hidden = 16
    args.n_rnn_layers = 1
    args.rnn_directions = 1
    args.bidirectional = False
    
    # DMMS.R 환경에 최적화된 설정
    args.max_epi_length = 288          # 정확히 1일 (5분 간격, 288스텝)
    args.n_step = 128                  # 더 긴 롤아웃으로 안정성 향상
    args.max_test_epi_len = 288        # 테스트도 1일
    
    # 할인 팩터 설정
    args.gamma = 1 if args.return_type == 'average' else 0.99
    args.lambda_ = 1 if args.return_type == 'average' else 0.95
    
    # 미세조정을 위한 보수적 설정
    args.entropy_coef = 0.01           # 적절한 탐험 수준
    args.grad_clip = 10                # 안정적인 그래디언트 클리핑
    args.eps_clip = 0.2                # 정책 업데이트 범위 완화
    args.target_kl = 0.02              # KL 발산 제한 완화
    
    # 보상 정규화 비활성화 (새로운 환경 적응을 위해)
    args.normalize_reward = False
    args.shuffle_rollout = True
    
    # 확장 훈련을 위한 에피소드 수 설정
    if hasattr(args, 'extended_episodes') and args.extended_episodes:
        args.n_training_episodes = args.extended_episodes
        print(f"[DMMS_DEBUG] Extended training episodes set to: {args.extended_episodes}")
    
    # 단일 시나리오를 위한 워커 설정
    args.n_training_workers = 1        # DMMS.R은 단일 환경
    args.n_testing_workers = 1
    
    # 안정적인 업데이트를 위한 설정
    args.n_pi_epochs = 3               # 정책 업데이트 횟수 감소
    args.n_vf_epochs = 3               # 가치 함수 업데이트 횟수 감소
    
    # 미세조정을 위한 매우 보수적인 학습률 (DMMS.R 환경 적응용)
    args.pi_lr = 1e-5                  # 정책 학습률 매우 보수적 설정
    args.vf_lr = 1e-5                  # 가치 함수 학습률 매우 보수적 설정
    args.batch_size = 512              # 안정적인 그래디언트를 위한 큰 배치 크기
    
    # 보조 모델 학습 설정 (단순화)
    args.aux_buffer_max = 5000         # 작은 버퍼 크기
    args.aux_frequency = 2             # 업데이트 빈도 감소
    args.aux_vf_coef = 0.005           # 계수 감소
    args.aux_pi_coef = 0.005
    args.aux_batch_size = 256
    args.n_aux_epochs = 3
    args.aux_lr = 5e-5
    
    # 계획 단순화
    args.planning_n_step = 6
    args.n_planning_simulations = 25   # 시뮬레이션 수 감소
    args.plan_batch_size = 256
    args.planning_lr = 5e-5
    
    return args