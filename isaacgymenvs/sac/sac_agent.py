from rl_games.algos_torch import torch_ext

from rl_games.common import vecenv
from rl_games.common import schedulers
from isaacgymenvs.ppo.a2c_common import print_statistics
from isaacgymenvs.ppo import model_builder
from isaacgymenvs.ppo.torch_ext import explained_variance
from isaacgymenvs.sac import her_replay_buffer
from isaacgymenvs.sac import experience
from isaacgymenvs.sac import validation_replay_buffer
from isaacgymenvs.utils.rlgames_utils import Every, get_grad_norm, save_cmd

from rl_games.interfaces.base_algorithm import  BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import defaultdict

check_for_none = lambda x: None if x == 'None' else x

class SACAgent(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.gradient_steps = config.get("gradient_steps", 1)
        self.grad_norm = check_for_none(self.config.get('grad_norm', None))
        self.normalize_input = config.get("normalize_input", False)
        self.relabel_ratio = config.get("relabel_ratio", 0.0)
        self.relabel_ratio_random = config.get("relabel_ratio_random", 0.0)
        self.test_every_episodes = config.get('test_every_episodes', 10) 
        self.reset_every_steps = check_for_none(config.get('reset_every_steps', None))
        self.validation_ratio = config.get('validation_ratio', 0.0)
        self.policy_update_fraction = config.get('policy_update_fraction', 1)
        self.mixed_precision = config.get('mixed_precision', False)
        self.rb_precision = config.get('rb_precision', 'float32')
        self.fill_buffer_first = config.get('fill_buffer_first', False)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]
        self.action_scale = (action_space.high[0].item() - action_space.low[0].item())/2

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)
        self.build_network()

        if self.relabel_ratio > 0.0:
            self.replay_buffer = validation_replay_buffer.ValidationHERReplayBuffer(self.env_info['observation_space'].shape,
                                                            self.env_info['action_space'].shape,
                                                            self.replay_buffer_size,
                                                            self.num_actors,
                                                            self._device,
                                                            self.vec_env.env,
                                                            self.rewards_shaper,
                                                            self.relabel_ratio,
                                                            self.relabel_ratio_random,
                                                            self.validation_ratio,
                                                            self.rb_precision)
        else:
            self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
                                                                self.env_info['action_space'].shape,
                                                                self.replay_buffer_size,
                                                                self._device,
                                                                self.rb_precision)
        
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)

    def build_network(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input': self.normalize_input,
        }
    
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        #temporary for Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.update_num = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        save_cmd(self.experiment_dir)

        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.writer = SummaryWriter(self.summaries_dir)
        print("Run Directory:", self.experiment_dir)

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self._device

    def get_weights(self):
        state = {'actor': self.model.sac_network.actor.state_dict(),
         'critic': self.model.sac_network.critic.state_dict(), 
         'critic_target': self.model.sac_network.critic_target.state_dict()}
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def get_full_state_weights(self):
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()        

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("SAC restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):
        pass

    def set_param(self, param_name, param_value):
        pass

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            with torch.no_grad():
                dist = self.model.actor(next_obs)
                next_action = dist.rsample()
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

                next_action = next_action * self.action_scale
                target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
                target_V = torch.min(target_Q1, target_Q2) 
                # target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob

                target_Q = reward + (not_done * self.gamma * target_V)
                target_Q = target_Q.detach()

            # get current Q estimates
            current_Q1, current_Q2 = self.model.critic(obs, action)

            critic1_loss = nn.MSELoss()(current_Q1, target_Q)
            critic2_loss = nn.MSELoss()(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss 

        info = {'losses/c_loss': critic_loss.detach(),
                'losses/c1_loss': critic1_loss.detach(),
                'losses/c2_loss': critic2_loss.detach(),
                'info/train_reward': reward.mean().detach(),
                'info/c_explained_variance': explained_variance(current_Q1, target_Q),}

        if self.relabel_ratio > 0:
            bs = current_Q1.shape[0]
            real = int(bs * (1 - self.relabel_ratio))
            info['losses/c_loss_original'] = nn.MSELoss()(current_Q1[:real], target_Q[:real]).detach()
            info['losses/c_loss_relabeled'] = nn.MSELoss()(current_Q1[real:], target_Q[real:]).detach()

        return critic_loss, info

    def update_actor_and_alpha(self, obs):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            dist = self.model.actor(obs)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = -log_prob.mean() #dist.entropy().sum(-1, keepdim=True).mean()
            action = action * self.action_scale
            actor_Q1, actor_Q2 = self.model.critic(obs, action)
            # actor_Q = (actor_Q1 + actor_Q2) / 2
            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_Q)
        actor_loss = actor_loss.mean()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True
        
        info = {'losses/a_loss': actor_loss.detach(), 
               'losses/entropy': entropy.detach(),
               'info/log_prob': log_prob.mean().detach(),
               'info/alpha': self.alpha.detach(),
               'info/actor_q': actor_Q.mean().detach(),
               'info/target_entropy': torch.ones(1) * self.target_entropy,}
        
        if self.relabel_ratio > 0:
            bs = actor_Q.shape[0]
            real = int(bs * (1 - self.relabel_ratio))
            info['info/actor_q'] = actor_Q[:real].mean().detach()
            info['info/actor_q_relabeled'] = actor_Q[real:].mean().detach()

        return actor_loss, info

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)

        # Critic
        critic_loss, critic_loss_info = self.update_critic(obs, action, reward, next_obs, ~done)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.sac_network.critic.parameters(), self.grad_norm)
        else:
            grad_norm = get_grad_norm(self.model.sac_network.critic.parameters())
        critic_loss_info['info/grad_norm'] = grad_norm.detach()
        self.critic_optimizer.step()

        actor_loss_info = {}
        if step % self.policy_update_fraction == 0:
            # Actor
            actor_loss, actor_loss_info = self.update_actor_and_alpha(obs)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            # Alpha
            if self.learnable_temperature:
                # alpha_loss = (self.log_alpha * (-log_prob - self.target_entropy).detach()).mean()
                alpha_loss = self.alpha * (-actor_loss_info['info/log_prob'] - self.target_entropy)
                self.log_alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                actor_loss_info['losses/alpha_loss'] = alpha_loss # TODO: maybe not self.alpha'
            else:
                alpha_loss = None

        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                     self.critic_tau)
        return actor_loss_info, critic_loss_info

    def validate(self):  
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, validation=True)

        with torch.no_grad():
            critic_loss, critic_loss_info = self.update_critic(obs, action, reward, next_obs, ~done)
            actor_loss, actor_loss_info = self.update_actor_and_alpha(obs)
        
        info = {'val/a_loss': actor_loss_info['losses/a_loss'],
                'val/actor_q': actor_loss_info['info/actor_q'],
                'val/actor_q_relabeled': actor_loss_info['info/actor_q_relabeled'],
                'val/c_loss': critic_loss_info['losses/c_loss'],
                'val/c_loss_original': critic_loss_info['losses/c_loss_original'],
                'val/c_loss_relabeled': critic_loss_info['losses/c_loss_relabeled'],}

        return info


    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        # obs = self.model.norm_obs(obs)

        return obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)

        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, terminated, truncated, infos = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), terminated.to(self._device), truncated.to(self._device), infos
        else:
            return torch.from_numpy(obs).to(self._device).float(), torch.from_numpy(rewards).to(self._device), torch.from_numpy(terminated).to(self._device), torch.from_numpy(truncated).to(self._device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()

        obs = self.obs_to_tensors(obs)

        return obs

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.actor(obs)

        actions = dist.sample() if sample else dist.mean
        actions = actions * self.action_scale
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2

        return actions

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()

    def play_steps(self, random_exploration = False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_metrics = defaultdict(list)
        critic_metrics = defaultdict(list)

        for s in range(self.num_steps_per_episode):
            obs = self.obs
            if isinstance(obs, dict):
                obs = obs['obs']
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()

            with torch.no_grad():
                next_obs, rewards, terminated, truncated, infos = self.env_step(action)

            if isinstance(next_obs, dict):
                next_obs = next_obs['obs']
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            dones = terminated + truncated
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            # if done_indices.numel() > 0:
            #     print(self.current_lengths[done_indices].mean())
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            self.obs = next_obs.clone()
            rewards = self.rewards_shaper(rewards)

            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(terminated, 1), torch.unsqueeze(dones, 1))

            if self.training_now() and not random_exploration:
                self.set_train()
                update_time_start = time.time()
                for _ in range(self.gradient_steps):
                    actor_loss_info, critic_loss_info = self.update(self.update_num)
                    for key, value in actor_loss_info.items(): actor_metrics[key].append(value)
                    for key, value in critic_loss_info.items(): critic_metrics[key].append(value)
                    self.update_num += 1
                update_time_end = time.time()
                update_time = update_time_end - update_time_start
            else:
                update_time = 0

            total_update_time += update_time

            if dones.any():
                obs = self.env_reset()
                if isinstance(obs, dict):
                    obs = obs['obs']
                self.obs[dones.bool()] = obs[dones.bool()]

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_metrics, critic_metrics

    def training_now(self):
        if self.fill_buffer_first:
            return self.replay_buffer.full
        else:
            return True

    def train_epoch(self):
        random_exploration = self.epoch_num < self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        test_check = Every(self.test_every_episodes * (self.vec_env.env.max_episode_length-1))
        render_check = Every(self.vec_env.env.render_every_episodes * (self.vec_env.env.max_episode_length-1))
        reset_check = Every(self.reset_every_steps)
        total_time = 0
        # rep_count = 0

        self.obs = self.env_reset()

        while True:
            if reset_check.check(self.frame):
                print('Reset network!')
                self.build_network()
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_metrics, critic_metrics = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time
            
            if self.epoch_num % 1000 == 0:
                self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
                self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
                self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
                self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
                self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
                self.writer.add_scalar('performance/step_time', step_time, self.frame)

                print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                    self.epoch_num, self.max_epochs, self.frame, self.max_frames, self.game_rewards.get_mean())
            
                if self.epoch_num >= self.num_warmup_steps:
                    for key, value in critic_metrics.items():
                        if value[0] is not None:
                            self.writer.add_scalar(key, torch_ext.mean_list(value).item(), self.frame)
                    for key, value in actor_metrics.items():
                        if value[0] is not None:
                            self.writer.add_scalar(key, torch_ext.mean_list(value).item(), self.frame)

                self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
                self.writer.add_scalar('info/updates', self.update_num, self.frame)
                self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

                if self.validation_ratio > 0.0:
                    val_info = self.validate()
                    for key, value in val_info.items():
                        self.writer.add_scalar(key, value.item(), self.frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    checkpoint_name = os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                            + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_'))

                    should_exit = False

                    if self.save_freq > 0:
                        if self.epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + self.config['name']))
                            self.save(checkpoint_name)

                    if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                            print('Maximum reward achieved. Network won!')
                            self.save(checkpoint_name)
                            should_exit = True

                    if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                        if self.game_rewards.current_size == 0:
                            print('WARNING: Max epochs reached before any env terminated at least once')
                            mean_rewards = -np.inf

                        self.save(checkpoint_name)
                        print('MAX EPOCHS NUM!')
                        should_exit = True

                    if self.frame >= self.max_frames and self.max_frames != -1:
                        if self.game_rewards.current_size == 0:
                            print('WARNING: Max frames reached before any env terminated at least once')
                            mean_rewards = -np.inf

                        self.save(checkpoint_name)
                        print('MAX FRAMES NUM!')
                        should_exit = True

                    update_time = 0

                    if should_exit:
                        return self.last_mean_rewards, self.epoch_num
                
            # Test
            iteration = self.frame / self.num_actors
            if test_check.check(iteration):
                print("Testing...")
                self.test(render=render_check.check(iteration))
                self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time, '_test')
                print("Done Testing.")

    def test(self, render):
        self.set_eval()
        self.vec_env.env.test = True
        if render:
            self.vec_env.env.override_render = True

        obs = self.env_reset()
        if isinstance(obs, dict):
            obs = obs['obs']
        self.obs = obs

        for n in range(self.vec_env.env.max_episode_length - 1):
            with torch.no_grad():
                action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            obs, rewards, terminated, truncated, infos = self.env_step(action)
            if isinstance(obs, dict):
                obs = obs['obs']

            # Save images
            dones = terminated + truncated
            all_done_indices = dones.nonzero(as_tuple=False)[::self.num_agents]
            self.algo_observer.process_infos(infos, all_done_indices)

        self.vec_env.env.test = False
        if render:
            self.vec_env.env.override_render = False

        obs = self.env_reset()
        if isinstance(obs, dict):
            obs = obs['obs']
        self.obs = obs