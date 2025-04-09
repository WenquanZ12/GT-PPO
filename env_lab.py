from mb_agg import *
import torch
from agent_utils import select_action
from agent_utils import greedy_select_action
from models.graph_trans import GT

torch.manual_seed(configs.torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(configs.torch_seed)
device = torch.device(configs.device)

# define network
gt = GT(n_j=n_j,
                           n_m=n_m,
                           num_layers=configs.num_layers,
                           learn_eps=False,
                           neighbor_pooling_type=configs.neighbor_pooling_type,
                           input_dim=configs.input_dim,
                           hidden_dim=configs.hidden_dim,
                           num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                           num_mlp_layers_actor=configs.num_mlp_layers_actor,
                           hidden_dim_actor=configs.hidden_dim_actor,
                           num_mlp_layers_critic=configs.num_mlp_layers_critic,
                           hidden_dim_critic=configs.hidden_dim_critic,
                           device=device)

# calculate g_pool for each step
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
                         n_nodes=n_j * n_m,
                         device=device)

data = uni_instance_gen1(n_j=n_j, n_m=n_m, low=low, high=high)
adj, fea, omega, mask = env.reset(data)
rewards = [- env.initQuality]
while True:
    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
    adj_tensor = torch.from_numpy(np.copy(adj)).to(device)
    candidate_tensor = torch.from_numpy(np.copy(omega)).to(device)
    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
    with torch.no_grad():
        pi, _ = GT(x=fea_tensor,
                             graph_pool=g_pool_step,
                             padded_nei=None,
                             adj=adj_tensor,
                             candidate=candidate_tensor.unsqueeze(0),
                             mask=mask_tensor.unsqueeze(0))
        # action, _ = select_action(pi, omega, None)
        _, indices = pi.squeeze().cpu().max(0)
        action = omega[indices.numpy().item()]
        adj, fea, reward, done, omega, mask = env.step(action.item())
        rewards.append(reward)
        if env.done():
            break
makespan = sum(rewards) - env.posRewards
print(makespan)
print(env.posRewards)
print(env.opIDsOnMchs)'''
# np.save('sol', env.opIDsOnMchs // n_m)
# np.save('jobSequence', env.opIDsOnMchs)
# np.save('testData', data)
# print(env.opIDsOnMchs // n_m + 1)
# print(env.step_count)
# print(t)
# print(np.concatenate((fea, data[1].reshape(-1, 1)), axis=1))
# print()
# print(env.adj)

