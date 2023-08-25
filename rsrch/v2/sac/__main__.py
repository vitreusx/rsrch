import torch
from rsrch.rl import gym
from .config import Config
from rsrch.rl.data.v2 import *
from torch import nn


def main():
    cfg = Config()

    def make_env(self):
        env = gym.make(cfg.env_name)
        return env

    sampler = UniformSampler()
    buffer = StepBuffer(cfg.buffer_cap, sampler=sampler)

    alpha = nn.Parameter(torch.tensor([1.0]))

    qs = []
    for q_idx in range(cfg.num_q_values):
        qs.append((q, target_q, q_opt, q_polyak))

    while True:
        for _ in range(env_iters):
            ...

        for _ in range(opt_iters):
            batch: StepBatch = buffer[sampler.sample(cfg.batch_size)]

            for q, target_q, q_opt, q_polyak in qs:
                gamma_t = cfg.gamma * (1.0 - batch.term)
                next_act_rv = pi(batch.next_obs)
                next_act = next_act_rv.sample()
                soft_v = target_q(
                    batch.next_obs, next_act
                ) - alpha * next_act_rv.log_prob(next_act)
                q_targ = batch.reward + gamma_t * soft_v
                q_pred = q(batch.obs, batch.act)
                q_loss = F.mse_loss(q_pred, q_targ)
                q_opt.optimize(q_loss)
                q_polyak.step()

            act_rv = pi(batch.obs)
            act = act_rv.rsample()
            q_preds = [q(batch.obs, act) for q in qs]
            min_q = torch.min(torch.stack(q_preds), dim=0)
            pi_loss = alpha * act_rv.log_prob(act) - min_q
            pi_opt.optimize(pi_loss)

            alpha_loss = alpha * (act_rv.entropy() - min_ent)
            alpha_opt.optimize(alpha_loss)


if __name__ == "__main__":
    ...
