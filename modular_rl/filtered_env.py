from gym import Env, spaces
import numpy as np


class FilteredEnv(Env):  #pylint: disable=W0223

    def __init__(self, env, ob_filter=None, act_filter=None, rew_filter=None, skiprate=None):
        self.env = env
        self.ob_filter = ob_filter
        self.rew_filter = rew_filter
        self.act_filter = act_filter
        self.skiprate = skiprate
        self.metadata = self.env.metadata


        if ob_filter:
            ob_space = self.env.observation_space
            shape = self.ob_filter.output_shape(ob_space)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape)
        else:
            self.observation_space = self.env.observation_space

        if act_filter:
            self.action_space = spaces.Discrete(act_filter.output_shape())
        else:
            self.action_space = self.env.action_space

    def _step(self, ac):
        nac = self.act_filter(ac) if self.act_filter else ac
        if self.skiprate:
            total_nrew = 0.0
            total_rew = 0.0
            num_steps = np.random.randint(self.skiprate[0], self.skiprate[1])
            nob = None
            done = False
            for _ in range(num_steps):
                ob, rew, done, info = self.env.step(nac)
                nob = self.ob_filter(ob) if self.ob_filter else ob
                nrew = self.rew_filter(rew) if self.rew_filter else rew
                total_nrew += nrew
                total_rew += rew
                if done:
                    info["reward_raw"] = total_rew
                    return (nob, total_nrew, done, info)
            info["reward_raw"] = total_rew
            return (nob, total_nrew, done, info)
        else:
            ob, rew, done, info = self.env.step(nac)
            nob = self.ob_filter(ob) if self.ob_filter else ob
            nrew = self.rew_filter(rew) if self.rew_filter else rew
            info["reward_raw"] = rew
            return (nob, nrew, done, info)

    def _reset(self):
        ob = self.env.reset()
        return self.ob_filter(ob) if self.ob_filter else ob

    def _render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def _configure(self, *args, **kwargs):
        self.env.configure(*args, **kwargs)
