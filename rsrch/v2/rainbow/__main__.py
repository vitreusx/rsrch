from rsrch.rl import gym


def main():
    def make_env():
        env = gym.wrappers.AtariPreprocessing(
            env=gym.make("ALE/Alien-v5", frameskip=4),
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
            noop_max=30,
        )
        env = gym.wrappers.TransformReward(env, lambda r: min(max(r, -1.0), 1.0))
        env = gym.wrappers.TimeLimit(env, int(108e3))
        return env


if __name__ == "__main__":
    main()
