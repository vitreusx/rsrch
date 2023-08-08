import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions.v3 as D
from rsrch.exp import prof
from rsrch.exp.board import TensorBoard
from rsrch.exp.dir import ExpDir
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils.detach import detach

from . import core, deter, oracle, rssm, test


class Dreamer(core.Dreamer):
    def setup_vars(self):
        self.val_every = int(2.5e3)
        self.val_episodes = 8
        self.env_steps = int(1e6)
        self.env_step_ratio = 4
        self.buffer_cap = int(1e5)
        self.batch_size = 128
        self.batch_seq_len = 32
        self.device = torch.device("cuda")
        self.log_every = int(1e2)
        self.kl_reg_coeff = 0.5
        self.horizon = 7
        self.copy_critic_every = 100
        self.wm_loss_scale = dict(kl=1.0, obs=1.0, term=1.0, reward=1.0)
        self.actor_loss_scale = dict(vpg=1.0, value=1.0, ent=1.0)
        self.prefill_steps = int(1e4)
        self.gamma = 0.99
        self.gae_lambda = 0.0
        self.clip_grad_norm = 100.0
        # self.policy_opt_delay = int(50e3)
        self.policy_opt_delay = 0
        self._use_oracle = False

    def setup_envs(self):
        self.env_name, self.env_type = "CartPole-v1", "other"
        # self.env_name, self.env_type = "ALE/Pong-v5", "atari"
        if self._use_oracle:
            self._mdp = oracle.Cartpole(device=self.device)

        if self.env_type == "atari":
            self.wm_loss_scale.update(kl=0.1)
            self.actor_loss_scale.update(ent=1e-3)
            # self.horizon = 10
        else:
            self.actor_loss_scale.update(ent=1e-4)
            # self.horizon = 15
        self.horizon = 7

        self.train_env = self._make_env()
        self.val_env = self._make_env()

    def _make_env(self):
        if self._use_oracle:
            return oracle.TrueEnv(self._mdp)
        if self.env_type == "atari":
            env = gym.wrappers.AtariPreprocessing(
                env=gym.make(self.env_name, frameskip=4),
                screen_size=64,
                frame_skip=1,
                grayscale_newaxis=True,
                scale_obs=True,
            )
        else:
            env = gym.make(self.env_name)
        return env

    def _setup_models_rssm(self):
        if self.env_type == "atari":
            deter_dim = hidden_dim = 600
            stoch_dim, num_heads = 1024, 32
        else:
            deter_dim = stoch_dim = 128
            hidden_dim = 128
            num_heads = 4

        self.wm = rssm.nets.RSSM(
            gym.wrappers.ToTensor(self.train_env, device=self.device),
            deter_dim,
            stoch_dim,
            hidden_dim,
            num_heads,
        )
        self.wm = self.wm.to(self.device)

        if isinstance(self.wm.obs_enc, rssm.nets.VisEncoder):
            vis_enc = self.wm.obs_enc
            self.obs_pred = rssm.nets.VisDecoder(
                vis_enc,
                deter_dim,
                stoch_dim,
            )
        elif isinstance(self.wm.obs_enc, rssm.nets.ProprioEncoder):
            pro_enc = self.wm.obs_enc
            self.obs_pred = rssm.nets.ProprioDecoder(
                pro_enc.input_shape,
                deter_dim,
                stoch_dim,
                fc_layers=[hidden_dim, hidden_dim],
            )

        self.obs_pred = self.obs_pred.to(self.device)

        wm_opt_params = [*self.wm.parameters(), *self.obs_pred.parameters()]
        self.wm_opt = torch.optim.Adam(wm_opt_params, lr=1e-4)

        self.actor = rssm.nets.Actor(
            self.wm.act_space,
            deter_dim,
            stoch_dim,
            # fc_layers=[hidden_dim] * 3,
            fc_layers=[hidden_dim] * 2,
        )
        self.actor = self.actor.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-5)

        make_critic = lambda: rssm.nets.Critic(
            deter_dim,
            stoch_dim,
            # fc_layers=[hidden_dim] * 3,
            fc_layers=[hidden_dim] * 2,
        )

        self.critic = make_critic()
        self.critic = self.critic.to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-5)

        self.target_critic = make_critic()
        self.target_critic = self.target_critic.to(self.device)
        self.target_critic.requires_grad_(False)
        polyak.update(self.critic, self.target_critic, tau=0.0)

    def _setup_models_deter(self):
        state_dim = 32
        hidden_dim = 64

        self.wm = deter.nets.DeterWM(self.train_env, state_dim, hidden_dim)
        self.wm = self.wm.to(self.device)

        if isinstance(self.wm.obs_enc, deter.nets.VisEncoder):
            vis_enc = self.wm.obs_enc
            self.obs_pred = deter.nets.VisDecoder(vis_enc, state_dim)
        elif isinstance(self.wm.obs_enc, deter.nets.ProprioEncoder):
            pro_enc = self.wm.obs_enc
            self.obs_pred = deter.nets.ProprioDecoder(pro_enc.input_shape, state_dim)
        self.obs_pred = self.obs_pred.to(self.device)

        wm_params = [*self.wm.parameters(), *self.obs_pred.parameters()]
        self.wm_opt = torch.optim.Adam(wm_params, lr=1e-3)

        self.actor = deter.nets.Actor(self.wm.act_space, state_dim, hidden_dim)
        self.actor = self.actor.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = deter.nets.Critic(state_dim, hidden_dim)
        self.critic = self.critic.to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.target_critic = deter.nets.Critic(state_dim, hidden_dim)
        self.target_critic = self.target_critic.to(self.device)
        self.target_critic.requires_grad_(False)
        polyak.update(self.critic, self.target_critic, tau=0.0)

        self.state_div = self._kl_state_div

    def _setup_models_test(self):
        hidden_dim = 64

        self.wm = test.TestWM(
            gym.wrappers.ToTensor(self.train_env, device=self.device),
            hidden_dim=hidden_dim,
        )
        self.wm = self.wm.to(self.device)
        state_dim = self.wm.state_dim

        class ObsPred(nn.Module):
            def forward(self, s):
                return D.Normal(s, D.Normal.MSE_SIGMA, 1)

        self.obs_pred = ObsPred()

        wm_params = [*self.wm.parameters(), *self.obs_pred.parameters()]
        self.wm_opt = torch.optim.Adam(wm_params, lr=1e-3)

        self.actor = deter.nets.Actor(self.wm.act_space, state_dim, hidden_dim)
        self.actor = self.actor.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = deter.nets.Critic(state_dim, hidden_dim)
        self.critic = self.critic.to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.target_critic = deter.nets.Critic(state_dim, hidden_dim)
        self.target_critic = self.target_critic.to(self.device)
        self.target_critic.requires_grad_(False)
        polyak.update(self.critic, self.target_critic, tau=0.0)

        self.state_div = self._l2_state_div

    def _setup_models_oracle(self):
        self.wm = oracle.TrueWM(self._mdp)
        self.obs_pred = lambda s: D.Normal(s, D.Normal.MSE_SIGMA, len(s.shape) - 1)

        self.wm_opt = None

        state_dim = int(np.prod(self._mdp.obs_space.shape))
        hidden_dim = 64

        self.actor = nn.Sequential(
            nn.Flatten(),
            deter.nets.Actor(self.wm.act_space, state_dim, hidden_dim),
        )
        self.actor = self.actor.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = nn.Sequential(
            nn.Flatten(),
            deter.nets.Critic(state_dim, hidden_dim),
        )
        self.critic = self.critic.to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.target_critic = deter.nets.Critic(state_dim, hidden_dim)
        self.target_critic = self.target_critic.to(self.device)
        self.target_critic.requires_grad_(False)
        polyak.update(self.critic, self.target_critic, tau=0.0)

    def setup_models(self):
        if self._use_oracle:
            return self._setup_models_oracle()
        else:
            return self._setup_models_test()
            # return self._setup_models_rssm()
            # return self._setup_models_deter()

    def _kl_state_div(self, post: D.Distribution, prior: D.Distribution) -> Tensor:
        stop_post, stop_prior = 0.0, 0.0
        if self.kl_reg_coeff != 0.0:
            stop_post = D.kl_divergence(detach(post), prior)
        if self.kl_reg_coeff != 1.0:
            stop_prior = D.kl_divergence(post, detach(prior))
        return self.kl_reg_coeff * stop_post + (1.0 - self.kl_reg_coeff) * stop_prior

    def _l2_state_div(self, post: D.Distribution, prior: D.Distribution):
        post, prior = post.mean, prior.mean
        stop_post, stop_prior = 0.0, 0.0
        if self.kl_reg_coeff != 0.0:
            stop_post = (post.detach() - prior).square().mean(-1)
        if self.kl_reg_coeff != 1.0:
            stop_prior = (post - prior.detach()).square().mean(-1)
        return self.kl_reg_coeff * stop_post + (1.0 - self.kl_reg_coeff) * stop_prior

    def state_div(self, post, prior) -> Tensor:
        ...

    def setup_extras(self):
        self.exp_dir = ExpDir(root="runs/v2_ext")
        self.board = TensorBoard(
            root_dir=self.exp_dir.path / "board",
            step_fn=lambda: self.env_step,
        )

        profile = False
        anomaly = False

        if profile:
            self.prof = prof.TorchProfiler(
                schedule=prof.TorchProfiler.schedule(5, 1, 3, 4),
                on_trace=prof.TorchProfiler.on_trace_fn(self.exp_dir.path),
                use_cuda=(self.device.type == "cuda"),
            )
        else:
            self.prof = prof.NullProfiler()

        torch.autograd.set_detect_anomaly(anomaly)
