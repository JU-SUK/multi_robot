import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic with Hindsight Experience Replay')
    parser.add_argument('--env_name', default="FetchPickAndPlace_test",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=10000000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_false",
                        help='run on CUDA (default: False)')
    parser.add_argument('--gradient_steps_per_epoch', type=int, default=50, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--episodes_per_epoch', type=int, default=8, metavar='N',
                        help='model updates per simulator step (default: 1)')
    
    parser.add_argument('--replay-strategy', type=str, default='future',
                        help='the HER strategy to be used')
    parser.add_argument('--replay-k', type=int, default=4,
                        help='ratio to be replace')
    parser.add_argument('--clip_range', type=float, default=5.0, help='clip range')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='number of rollouts per mpi')
    parser.add_argument('--clip-obs', type=float, default=200, help='clip obs')
    
    

    args = parser.parse_args()
    
    return args