import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from cal_fuzzy import TFN
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
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def fuzzy_mean(self, fuzzy_triplet):
        a, b, c = fuzzy_triplet
        return (a + 2 * b + c) / 4.0  # 可以根据需要调整权重

    def _get_unscheduled_tasks(self, job_id):

        unscheduled = []
        for op_id in range(self.num_machines):
            task_id = job_id * self.num_machines + op_id
            if self.schedule_flags[task_id] == 0:
                unscheduled.append(task_id)
        return unscheduled

    def _get_previous_operation_end_time(self, job_id):

        for op_id in range(self.num_machines):
            task_id = job_id * self.num_machines + op_id
            if self.schedule_flags[task_id] == 0:
                if op_id == 0:
                    return 0  # 第一个工序没有前置工序
                prev_task_id = job_id * self.num_machines + (op_id - 1)
                return self.end_time.get(prev_task_id, 0)  # 如果未定义就返回 0
        # 如果这个作业所有工序都已调度，返回最后一个工序的结束时间
        last_task_id = job_id * self.num_machines + (self.num_machines - 1)
        return self.end_time.get(last_task_id, 0)

    def step(self, action):
        # Avoid repeated actions
        if action not in self.partial_sol_sequeence:

            # === Basic Information Update ===
            row = action // self.number_of_machines  # Job index
            col = action % self.number_of_machines  # Operation index
            self.step_count += 1
            self.finished_mark[row, col] = 1
            self.partial_sol_sequeence.append(action)

            fuzzy_dur = self.dur[row, col]  # Triangular fuzzy number
            dur_a = self.fuzzy_mean(fuzzy_dur)  # Mean duration for simulation

            # Compute permissible left shift and start time
            start_time_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m,
                                                      mchsStartTimes=self.mchsStartTimes,
                                                      opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = start_time_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)
            # === Task-Specific Information ===
            task = self.tasks[action]
            job_id = task['job_id']
            machine_id = task['machine_id']
            processing_time = task['processing_time']

            # Start and end time based on current machine state
            start_time = self.completion_time[machine_id]
            end_time = start_time + processing_time

            # === State Updates ===
            self.scheduling_tag[action] = 1
            self.working_tag[machine_id] = 1
            self.completion_time[machine_id] = end_time

            # Update remaining operations
            self.remaining_operations[job_id] -= 1

            # Update workload estimate for unscheduled operations
            if self.remaining_operations[job_id] > 0:
                unscheduled = self._get_unscheduled_tasks_of_job(job_id)
                avg_proc_time = np.mean([self.tasks[idx]['processing_time'] for idx in unscheduled])
                self.remaining_workload[job_id] = avg_proc_time * self.remaining_operations[job_id]
            else:
                self.remaining_workload[job_id] = 0

            # Update waiting time
            self.waiting_time[action] = start_time - self._get_previous_operation_end_time(job_id)
            self.remaining_processing_time[action] = 0

            # Update remaining processing time for unscheduled tasks
            for i in range(len(self.remaining_processing_time)):
                if self.scheduling_tag[i] == 0:
                    self.remaining_processing_time[i] = self.tasks[i]['processing_time']

            # === Omega and Mask Updates ===
            if action not in self.last_col:
                self.omega[job_id] += 1
            else:
                self.mask[job_id] = 1

            # === Update Schedule Completion Matrix ===
            self.temp1[row, col] = start_time_a + dur_a
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # === Update Adjacency Matrix ===
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1  # self-loop
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:
                self.adj[succd, precd] = 0
        # 构造特征 fea
        job_ids_per_task = np.repeat(np.arange(self.number_of_jobs), self.number_of_machines)

        fea = np.concatenate((
            self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
            self.finished_mark.reshape(-1, 1),
            self.scheduling_tag.reshape(-1, 1),
            self.waiting_time.reshape(-1, 1),
            self.remaining_processing_time.reshape(-1, 1) / configs.et_normalize_coef,
            self.remaining_operations[job_ids_per_task].reshape(-1, 1) / self.number_of_machines,
            self.remaining_workload[job_ids_per_task].reshape(-1, 1) / configs.et_normalize_coef
        ), axis=1)

        # === Compute Reward ===
        reward = -(self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return self.adj, fea, reward, self.done(), self.omega, self.mask


    def reset(self, data):
        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)  # 处理时间（三角模糊数）
        self.dur_cp = np.copy(self.dur)
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        self.end_time = {}
        self.schedule_flags = [0] * (self.num_jobs * self.num_machines)

        # 初始化邻接矩阵
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        conj_nei_up_stream[self.first_col] = 0
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # 初始化模糊处理时间
        self.fuzzy_dur = np.apply_along_axis(self.dur, 2, self.dur)
        self.LBs = np.cumsum(self.fuzzy_dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)



        # 初始化可调度集合 omega 和 mask
        self.omega = self.first_col.astype(np.int64)
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # 初始化机器状态
        self.completion_time = np.zeros(self.number_of_machines, dtype=np.single)
        self.working_tag = np.zeros(self.number_of_machines, dtype=np.int32)

        # 初始化任务调度状态
        self.scheduling_tag = np.zeros(self.number_of_tasks, dtype=np.int32)

        # 初始化每个 job 的剩余操作数量
        self.remaining_operations = np.full(self.number_of_jobs, fill_value=self.number_of_machines, dtype=np.int32)

        # 初始化剩余工作负载（初始为总处理时间）
        self.remaining_workload = np.zeros(self.number_of_jobs, dtype=np.single)
        for job_id in range(self.number_of_jobs):
            task_ids = self._get_unscheduled_tasks_of_job(job_id)
            self.remaining_workload[job_id] = sum(self.tasks[tid]['processing_time'] for tid in task_ids)

        # 初始化等待时间和剩余处理时间
        self.waiting_time = np.zeros(self.number_of_tasks, dtype=np.single)
        self.remaining_processing_time = np.array(
            [self.tasks[i]['processing_time'] for i in range(self.number_of_tasks)],
            dtype=np.single)

        # 初始化特征 fea
        job_ids_per_task = np.repeat(np.arange(self.number_of_jobs), self.number_of_machines)

        fea = np.concatenate((
            self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
            self.finished_mark.reshape(-1, 1),
            self.scheduling_tag.reshape(-1, 1),
            self.waiting_time.reshape(-1, 1),
            self.remaining_processing_time.reshape(-1, 1) / configs.et_normalize_coef,
            self.remaining_operations[job_ids_per_task].reshape(-1, 1) / self.number_of_machines,
            self.remaining_workload[job_ids_per_task].reshape(-1, 1) / configs.et_normalize_coef
        ), axis=1)

        # 初始化机器调度结构（用于左移计算）
        self.mchsStartTimes = -configs.high * np.ones_like(self.fuzzy_dur.transpose(), dtype=np.int32)
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.fuzzy_dur.transpose(), dtype=np.int32)

        # 初始化任务完成时间矩阵（每个任务的调度完成时刻）
        self.temp1 = np.zeros_like(self.fuzzy_dur, dtype=np.single)

        return self.adj, fea, self.omega, self.mask

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False
