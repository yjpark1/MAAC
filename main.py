import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from utils.make_env import make_env
from utils.scenarios import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import time
import pickle

TEST_ONLY = True

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, local_observation=False, discrete_action=True)
            env.seed(seed)
            np.random.seed(seed)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config, run_num):
    model_dir = Path('Models') / config.env_id / config.model_name
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num + 12345678)
    np.random.seed(run_num + 12345678)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num + 12345678)

    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       attend_tau=config.attend_tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)

    action_space = []
    for acsp in env.action_space:
        if acsp.__class__.__name__ == 'MultiDiscrete':
            action_space.append(sum(acsp.high + 1))
        else:
            action_space.append(acsp.n)

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 action_space)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.envs[0].n)]  # individual agent reward
    
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve

    t = 0
    t_start = time.time()
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            # make shared reward
            rew_shared = np.array([[np.sum(rewards)] * env.envs[0].n])
            replay_buffer.push(obs, agent_actions, rew_shared, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads

            for i, rew in enumerate(rewards[0]):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if (len(replay_buffer) >= max(config.pi_batch_size, config.q_batch_size) and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_critic_updates):
                    sample = replay_buffer.sample(config.q_batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                for u_i in range(config.num_pol_updates):
                    sample = replay_buffer.sample(config.pi_batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_policies(sample, logger=logger)
                model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        episode_rewards.append(0)
        for a in agent_rewards:
            a.append(0)

        # save model, display training output
        if len(episode_rewards) % config.save_interval == 0:
            # print statement depends on whether or not there are adversaries
            print("episodes: {}, mean episode reward: {}, time: {}".format(
                len(episode_rewards), np.mean(episode_rewards[-config.save_interval:]),
                round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-config.save_interval:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-config.save_interval:]))

        # if ep_i % config.save_interval < config.n_rollout_threads:
        #     model.prep_rollouts(device='cpu')
        #     os.makedirs(run_dir / 'incremental', exist_ok=True)
        #     model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
        #     model.save(run_dir / 'model.pt')

    # saves final episode reward for plotting training curve later
    if len(episode_rewards) >= config.n_episodes:
        hist = {'reward_episodes': episode_rewards, 'reward_episodes_by_agents': agent_rewards}
        file_name = 'Models/history_' + config.env_id + '_' + str(run_num) + '.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(hist, fp)
        print('...Finished total of {} episodes.'.format(len(episode_rewards)))

    model.save('Models/model_' + config.env_id + '_' + str(run_num) + '.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


def run_test(config, run_num):
    torch.manual_seed(run_num + 12345678)
    np.random.seed(run_num + 12345678)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num + 12345678)

    fname = 'D:\MEGA/GitHub/multiagent_rl/Models/env_origin/MAAC/model_' + config.env_id + '_' + str(run_num) + '.pt'
    model = AttentionSAC.init_from_save(filename=fname)

    action_space = []
    for acsp in env.action_space:
        if acsp.__class__.__name__ == 'MultiDiscrete':
            action_space.append(sum(acsp.high + 1))
        else:
            action_space.append(acsp.n)

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 action_space)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.envs[0].n)]  # individual agent reward

    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve

    t = 0
    t_start = time.time()
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            # make shared reward
            rew_shared = np.array([[np.sum(rewards)] * env.envs[0].n])
            replay_buffer.push(obs, agent_actions, rew_shared, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads

            for i, rew in enumerate(rewards[0]):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

        episode_rewards.append(0)
        for a in agent_rewards:
            a.append(0)

        # save model, display training output
        if len(episode_rewards) % 10 == 0:
            # print statement depends on whether or not there are adversaries
            print("episodes: {}, mean episode reward: {}, time: {}".format(
                len(episode_rewards), np.mean(episode_rewards[-10:]),
                round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-10:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-10:]))

    # saves final episode reward for plotting training curve later
    if len(episode_rewards) >= config.n_episodes:
        hist = {'reward_episodes': episode_rewards,
                'reward_episodes_by_agents': agent_rewards,}
        file_name = 'Models/test_history_' + config.env_id + '_' + str(run_num) + '.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(hist, fp)
        print('...Finished total of {} episodes.'.format(len(episode_rewards)))
        import sys
        sys.getsizeof(hist)
    env.close()



if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_critic_updates", default=1, type=int,
                        help="Number of critic updates per update cycle")
    parser.add_argument("--num_pol_updates", default=1, type=int,
                        help="Number of policy updates per update cycle")
    parser.add_argument("--pi_batch_size",
                        default=1024, type=int,
                        help="Batch size for policy training")
    parser.add_argument("--q_batch_size",
                        default=1024, type=int,
                        help="Batch size for critic training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=64, type=int)
    parser.add_argument("--critic_hidden_dim", default=64, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.04, type=float)
    parser.add_argument("--attend_tau", default=0.002, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--reward_scale", default=1., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()
    config.model_name = 'MAAC'
    config.use_gpu = True

    scenarios = ['simple_spread', 'simple_reference', 'simple_speaker_listener',
                 'fullobs_collect_treasure', 'multi_speaker_listener']

    for sce in scenarios:
        config.env_id = sce
        for rn in range(10):
            torch.cuda.empty_cache()
            if TEST_ONLY:
                run_test(config, run_num=rn)
            else:
                run(config, run_num=rn)
