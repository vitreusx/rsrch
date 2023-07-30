import torch
from torch import nn

from rsrch.exp.board import TensorBoard
from rsrch.exp.dir import ExpDir
from rsrch.rl import agents, gym
from rsrch.rl.data import PaddedSeqBatch, SeqBuffer, interact
from rsrch.rl.data import transforms as T
from rsrch.utils import data

from . import api, core, nets


class Dreamer(core.Dreamer):
    def setup_conf(self):
        self.max_steps = int(1e6)
        self.val_every_step = int(10e3)
        self.val_episodes = 4
        self.batch_size = 50
        self.batch_seq_len = 50
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.alpha = 0.8
        self.copy_critic_every = 100
        self.train_every = 16
        self.log_every = int(5e2)
        self.per_env_conf()

        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

        self.device = torch.device("cuda")
        self.buf_capacity = int(1e5)
        self.buf_prefill_steps = int(1e3)

    def per_env_conf(self):
        # self.env_name = "ALE/Pong-v5"
        # self.env_type = "atari"
        self.env_name = "CartPole-v1"
        self.env_type = "other"

        if self.env_type == "atari":
            self.beta = 0.1
            self.rho = 1.0
            self.eta = 1e-3
            self.horizon = 10
            self.action_repeat = 4
        elif self.env_type in ["dmc", "other"]:
            self.beta = 1.0
            self.rho = 1.0
            self.eta = 1e-4
            self.horizon = 15
            self.action_repeat = 1

    def setup_data(self):
        self.train_env = self.make_env()
        self.val_env = self.make_env()

        self.buffer = SeqBuffer(
            capacity=self.buf_capacity,
            min_seq_len=self.batch_seq_len,
        )
        self.prefill()

        ds = data.Pipeline(
            self.buffer,
            T.Subsample(
                min_seq_len=self.batch_seq_len,
                max_seq_len=self.batch_seq_len,
                prioritize_ends=True,
            ),
            T.ToTensorSeq(),
        )
        loader = data.DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            sampler=data.InfiniteSampler(ds, shuffle=True),
            collate_fn=PaddedSeqBatch.collate_fn,
        )
        dev_loader = data.Pipeline(
            loader,
            T.ToDevice(self.device),
        )
        self.train_batches = iter(dev_loader)

    def make_env(self):
        if self.env_type == "atari":
            env = gym.make(self.env_name, frameskip=self.action_repeat)
            env = gym.wrappers.AtariPreprocessing(
                env=env,
                screen_size=64,
                frame_skip=1,
                grayscale_newaxis=True,
            )
            env = gym.wrappers.ToTensor(
                env,
                device=self.device,
                visual_obs=True,
            )
        else:
            env = gym.make(self.env_name)
            env = env = gym.wrappers.ToTensor(
                env,
                device=self.device,
                visual_obs=False,
            )

        return env

    def prefill(self):
        prefill_env = self.make_env()
        prefill_agent = agents.RandomAgent(self.train_env)
        steps_ex = interact.steps_ex(
            prefill_env,
            prefill_agent,
            max_steps=self.buf_prefill_steps,
        )
        for step, done in steps_ex:
            self.buffer.add(step, done)
        ...

    def setup_models_and_optimizers(self):
        num_classes = 32
        if self.env_type == "atari":
            h_dim = hidden_dim = 600
            z_dim = 1024
        elif self.env_type == "other":
            h_dim = z_dim = 128
            hidden_dim = 128

        self.rssm = nets.RSSM(self.train_env, h_dim, z_dim, hidden_dim, num_classes)
        self.rssm = self.rssm.to(self.device)

        if isinstance(self.rssm.obs_enc, nets.VisEncoder2):
            vis_enc = self.rssm.obs_enc
            self.obs_pred = nets.VisDecoder3(vis_enc, h_dim, z_dim)
        elif isinstance(self.rssm.obs_enc, nets.ProprioEncoder):
            pro_enc = self.rssm.obs_enc
            self.obs_pred = nets.ProprioDecoder(pro_enc.input_shape, h_dim, z_dim)

        self.obs_pred = self.obs_pred.to(self.device)

        wm_params = [*self.rssm.parameters(), *self.obs_pred.parameters()]
        self.wm_optim = torch.optim.Adam(wm_params, lr=1e-3)

        self.critic = nets.Critic(h_dim, z_dim, fc_layers=[hidden_dim] * 3).to(
            self.device
        )

        self.target_critic = self.critic.clone()
        self.target_critic.requires_grad_ = False

        act_space = self.train_env.action_space
        self.actor = nets.Actor(
            act_space,
            h_dim,
            z_dim,
            fc_layers=[hidden_dim] * 3,
        ).to(self.device)

        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def setup_extras(self):
        self.exp_dir = ExpDir()
        self.board = TensorBoard(
            root_dir=self.exp_dir.path,
            step_fn=lambda: self.step,
        )
        self.detect_anomaly = False
        self.prof_enabled = False
        self.prof_cycles = 4
        self.prof_wait, self.prof_warmup, self.prof_active = 5, 1, 3
