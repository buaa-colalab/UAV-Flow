# SPDX-License-Identifier: Apache-2.0
# Modified for UAV-Flow-Eval; upstream attribution and notices: LICENSE.
import time

from gym import Wrapper


class EarlyDoneWrapper(Wrapper):
    def __init__(self, env, max_lost_steps=50):
        super().__init__(env)
        self.max_lost_steps = max_lost_steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # take a step in the wrapped environment

        if not info["metrics"]["target_viewed"]:
            self.count_lost += 1
        else:
            self.count_lost = 0
        _env = self.env.unwrapped
        if self.count_lost > self.max_lost_steps:
            info["Done"] = True
            done = True
        return obs, reward, done, info  # return the same results as the wrapped environment

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        self.start_time = time.time()
        self.count_lost = 0
        return states  # return the same results as the wrapped environment
