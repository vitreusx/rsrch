import torch

from rsrch.exp.board import TensorBoard
from rsrch.exp.dir import ExpDir
from rsrch.rl import gym

from . import core, nets, rssm, wm


class Dreamer(core.Dreamer):
    def setup_vars(self):
        self.val_every = int(1e4)
        self.val_episodes = 32
        self.env_steps = int(1e6)
        self.env_step_ratio = 16
        self.buffer_cap = int(1e5)
        self.batch_size = 50
        self.batch_seq_len = 50
        self.device = torch.device("cuda")
        self.log_every = int(1e3)
        self.kl_reg_coeff = 1.0
        self.horizon = 15
        self.copy_critic_every = 100
        self.wm_loss_scale = dict(kl=1.0, obs=1.0, term=1.0, reward=1.0)
        self.actor_loss_scale = dict(vpg=1.0, value=1.0, ent=1.0)
        self.prefill_size = 64

    def setup_envs(self):
        self.env_name = "CartPole-v1"
        self.env_type = "other"

        if self.env_type == "atari":
            self.wm_loss_scale.update(kl=0.1)
            self.actor_loss_scale.update(ent=1e-3)
            self.horizon = 10
        else:
            self.actor_loss_scale.update(ent=1e-4)
            self.horizon = 15

        self.train_env = self._make_env()
        self.val_env = self._make_env()

    def _make_env(self):
        if self.env_type == "atari":
            env = gym.wrappers.AtariPreprocessing(
                env=gym.make(self.env_name, frameskip=4),
                screen_size=64,
                frame_skip=1,
                grayscale_newaxis=True,
            )
            env = gym.wrappers.ToTensor(env, self.device, visual_obs=True)
            env = gym.wrappers.OneHotActions(env)
        else:
            env = gym.make(self.env_name)
            env = gym.wrappers.ToTensor(env, self.device, visual_obs=False)

        if isinstance(env.action_space, gym.spaces.TensorDiscrete):
            env = gym.wrappers.OneHotActions(env)

        return env

    def setup_models(self):
        if self.env_type == "atari":
            deter_dim = hidden_dim = 600
            stoch_dim, num_heads = 1024, 32
        else:
            deter_dim = stoch_dim = hidden_dim = 128
            num_heads = 4

        self.rssm = nets.RSSM(
            self.train_env,
            deter_dim,
            stoch_dim,
            hidden_dim,
            num_heads,
        )
        self.rssm = self.rssm.to(self.device)

        self.wm = rssm.WorldModel(self.rssm)

        if isinstance(self.wm.obs_enc, nets.VisEncoder):
            vis_enc = self.wm.obs_enc
            self.obs_pred = nets.VisDecoder(
                vis_enc,
                deter_dim,
                stoch_dim,
            )
        elif isinstance(self.wm.obs_enc, nets.ProprioEncoder):
            pro_enc = self.wm.obs_enc
            self.obs_pred = nets.ProprioDecoder(
                pro_enc.input_shape,
                deter_dim,
                stoch_dim,
            )
        else:
            raise ValueError(f"Invalid wm.obs_enc type {type(self.wm.obs_enc)}")
        self.obs_pred = self.obs_pred.to(self.device)

        wm_opt_params = [*self.rssm.parameters(), *self.obs_pred.parameters()]
        self.wm_opt = torch.optim.Adam(wm_opt_params, lr=1e-3)

        self.actor = nets.Actor(
            self.wm.act_space,
            deter_dim,
            stoch_dim,
            fc_layers=[hidden_dim] * 3,
        )
        self.actor = self.actor.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = nets.Critic(
            deter_dim,
            stoch_dim,
            fc_layers=[hidden_dim] * 3,
        )
        self.critic = self.critic.to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.target_critic = nets.Critic(deter_dim, stoch_dim)
        self.target_critic = self.target_critic.to(self.device)
        self.target_critic.requires_grad_(False)

    def setup_extras(self):
        self.exp_dir = ExpDir(root="runs/v2_ext")
        self.board = TensorBoard(
            root_dir=self.exp_dir.path / "board",
            step_fn=lambda: self.env_step,
        )
