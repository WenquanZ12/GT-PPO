import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs

import numpy as np
import gym
from gym import spaces


class FuzzySJSSP(gym.Env):
    def __init__(self, n_j, n_m):
        super(FuzzySJSSP, self).__init__()

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines

        # Define fuzzy processing times (a, b, c) for each task
        self.dur = np.random.randint(1, 10,
                                     (self.number_of_jobs, self.number_of_machines, 3))  # 3 for fuzzy values (a, b, c)

        # Initialize states
        self.scheduling_tag = np.zeros(self.number_of_tasks, dtype=int)  # 1 if scheduled, 0 if unscheduled
        self.remaining_operations = np.full(self.number_of_jobs, self.number_of_machines,
                                            dtype=int)  # Remaining unscheduled operations per job
        self.remaining_workload = np.zeros(
            self.number_of_jobs)  # Cumulative average processing time for unscheduled operations
        self.working_tag = np.zeros(self.number_of_machines, dtype=int)  # 1 if machine is working, 0 if free
        self.completion_time = np.zeros(self.number_of_machines)  # Completion times for machines
        self.waiting_time = np.zeros(self.number_of_tasks)  # Waiting times for each operation
        self.remaining_processing_time = np.zeros(self.number_of_tasks)  # Remaining processing times

        # Define action and observation space
        self.action_space = spaces.Discrete(self.number_of_tasks)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.number_of_tasks, 8), dtype=np.float32)

    def fuzzy_mean(self, fuzzy_time):
        """
        Calculate the fuzzy mean for a triangular fuzzy number (a, b, c).
        """
        a, b, c = fuzzy_time
        return (a + b + c) / 3  # The mean value of the triangular fuzzy number

    def fuzzy_add(self, fuzzy_a, fuzzy_b):
        """
        Add two triangular fuzzy numbers.
        """
        a1, b1, c1 = fuzzy_a
        a2, b2, c2 = fuzzy_b
        return np.array([a1 + a2, b1 + b2, c1 + c2])

    def fuzzy_compare(self, fuzzy_s, fuzzy_t):
        """
        Compare two triangular fuzzy numbers.
        """
        # Calculate fuzzy means
        f1_s = (fuzzy_s[0] + 2 * fuzzy_s[1] + fuzzy_s[2]) / 4
        f1_t = (fuzzy_t[0] + 2 * fuzzy_t[1] + fuzzy_t[2]) / 4

        if f1_s > f1_t:
            return 1  # s > t
        elif f1_s < f1_t:
            return -1  # s < t
        else:
            # If fuzzy means are equal, compare using the second criterion
            f2_s = fuzzy_s[1]
            f2_t = fuzzy_t[1]
            if f2_s > f2_t:
                return 1
            elif f2_s < f2_t:
                return -1
            else:
                # If still equal, compare using the third criterion
                f3_s = fuzzy_s[2] - fuzzy_s[0]
                f3_t = fuzzy_t[2] - fuzzy_t[0]
                if f3_s > f3_t:
                    return 1
                else:
                    return -1

    def fuzzy_max(self, fuzzy_s, fuzzy_t):
        """
        Return the maximum of two triangular fuzzy numbers.
        """
        if self.fuzzy_compare(fuzzy_s, fuzzy_t) > 0:
            return fuzzy_s
        else:
            return fuzzy_t

    def step(self, action):
        # action is an integer index, e.g., 0-224 for a 15x15 instance
        if action not in self.partial_sol_sequeence:  # Avoid redundant actions

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines  # Determine job index
            col = action % self.number_of_machines  # Determine machine index
            self.step_count += 1  # Increment step count
            self.finished_mark[row, col] = 1  # Mark task as finished
            fuzzy_dur = self.dur[row, col]  # Get fuzzy processing time (a, b, c)
            dur_a = self.fuzzy_mean(fuzzy_dur)  # Convert fuzzy duration to mean
            self.partial_sol_sequeence.append(action)  # Store the scheduled action

            # UPDATE STATE:
            # Compute permissible left shift using fuzzy duration
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m,
                                                     mchsStartTimes=self.mchsStartTimes,
                                                     opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)

            # Update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            # Update task completion time using fuzzy mean duration
            self.temp1[row, col] = startTime_a + dur_a

            # Compute lower bound for schedule completion time
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # UPDATE ADJACENCY MATRIX:
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)  # Get neighboring nodes
            self.adj[action] = 0  # Reset adjacency for this node
            self.adj[action, action] = 1  # Self-loop
            if action not in self.first_col:
                self.adj[action, action - 1] = 1  # Maintain order constraints
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:
                self.adj[succd, precd] = 0  # Remove old arc if a new operation inserts between

        # PREPARE OUTPUT:
        fea = np.concatenate((self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)

        reward = - (self.LBs.max() - self.max_endTime)  # Reward based on improvement
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()  # Update max completion time

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    import numpy as np

    def fuzzy_mean(fuzzy_time):
        """
        计算三角模糊数 (a, b, c) 的模糊均值。
        """
        a, b, c = fuzzy_time
        return (a + b + c) / 3  # 采用三角模糊数的均值计算

    def reset(self, data):
        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)  # 处理时间
        self.dur_cp = np.copy(self.dur)
        self.partial_sol_sequeence = []  # 记录调度过程
        self.flags = []  # 记录标志信息
        self.posRewards = 0  # 记录奖励

        # 初始化邻接矩阵
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        conj_nei_up_stream[self.first_col] = 0  # 第一列没有上游邻接
        conj_nei_low_stream[self.last_col] = 0  # 最后一列没有下游邻接
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # 初始化模糊处理时间
        self.fuzzy_dur = np.apply_along_axis(fuzzy_mean, 2, self.dur)  # 计算模糊均值
        self.LBs = np.cumsum(self.fuzzy_dur, axis=1, dtype=np.single)  # 计算累积下界
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)

        # 初始化可调度集合 omega
        self.omega = self.first_col.astype(np.int64)

        # 初始化掩码 mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # 机器上的开始时间初始化
        self.mchsStartTimes = -configs.high * np.ones_like(self.fuzzy_dur.transpose(), dtype=np.int32)
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.fuzzy_dur.transpose(), dtype=np.int32)
        self.temp1 = np.zeros_like(self.fuzzy_dur, dtype=np.single)

        return self.adj, fea, self.omega, self.mask

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False
