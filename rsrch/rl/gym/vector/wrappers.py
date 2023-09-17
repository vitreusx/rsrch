from gymnasium.vector import VectorEnvWrapper


class TransformReward(VectorEnvWrapper):
    def __init__(self, env, f):
        super().__init__(env)
        self.f = f

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        reward = self.f(reward)
        return next_obs, reward, term, trunc, info
