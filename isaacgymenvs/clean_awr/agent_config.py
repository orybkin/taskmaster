import ml_collections
from ml_collections import config_flags

fetch_push_config = ml_collections.ConfigDict({
    'experiment': '',
    'num_envs': 32,
    'clip_observations': 5.0,
    'clip_actions': 1,
    'hidden_dims': (256, 128, 64),
    'temperature': 0.2, # 0 for behavior cloning.
    'horizon_length': 8192,
    'num_minibatches': 32,
    'mini_epochs': 5,
    'gamma': 0.95,
    'tau': 0.95,
    'lr': 0.0005,
    'entropy_coef': 0.05,
    'normalize_value': 1,
    'normalize_input': 1,
    'normalize_advantage': 1,
    'normalize_advantage_joint': 0,
    'value_bootstrap': 1,
    'joint_value_norm': 1,
    'relabel': 1,
    'device': 'cuda:0',
    'graphics_device_id': 0,
    'test_every_episodes': 10,
    'max_epochs': 50000,
    'e_clip': 0.2,
    'clip_value': 1,
    'critic_coef': 4.,
    'bounds_loss_coef': 0.0001,
    'truncate_grads': 0,
    'grad_norm': 5,
})

ig_push_config = ml_collections.ConfigDict({
    'experiment': '',
    'num_envs': 8192,
    'clip_observations': 5.0,
    'clip_actions': 1,
    'hidden_dims': (256, 128, 64),
    'temperature': 0.2, # 0 for behavior cloning.
    'horizon_length': 256,
    'num_minibatches': 32,
    'mini_epochs': 5,
    'gamma': 0.99,
    'tau': 0.95,
    'lr': 0.0005,
    'entropy_coef': 0.01,
    'normalize_value': 1,
    'normalize_input': 1,
    'normalize_advantage': 1,
    'normalize_advantage_joint': 0,
    'value_bootstrap': 1,
    'joint_value_norm': 1,
    'relabel': 1,
    'device': 'cuda:0',
    'graphics_device_id': 0,
    'test_every_episodes': 10,
    'max_epochs': 50000,
    'e_clip': 0.2,
    'clip_value': 1,
    'critic_coef': 4.,
    'bounds_loss_coef': 0.0001,
    'truncate_grads': 0,
    'grad_norm': 5,
    'rnd': {
        'enable': 0,
        'coef': 0.1,
        'warmup_steps': 50,
        'warmup_epochs': 20,
        'calibrate_epochs': 0,
        'units': (256, 128, 64),
        'lr': 30e-5,
        'batch_size': 1024,
        'running_stats_alpha': 0.1
    }
})
