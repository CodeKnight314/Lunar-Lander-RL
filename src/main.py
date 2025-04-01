from discrete_env import D_environment
from continuous_env import C_environment
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lunar Lander Reinforcement Learning')
    parser.add_argument('--c', type=str, default='src/config.yaml', help='Path to the config file')
    parser.add_argument('--o', type=str, default='src/weights', help='Path to the output directory')
    parser.add_argument('--w', type=str, default=None, help='Path to the weights file')
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--env", type=str, choices=["discrete", "continuous"], default="discrete", 
                        help="Choose environment type: discrete or continuous")
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    if args.env == "discrete":
        env = D_environment(args.c, args.w)
        if args.train:
            print("Training DQN in discrete Lunar Lander environment...")
            env.train_dqn(args.o)
            env.test_dqn(args.o)
        elif args.test:
            print("Testing DQN in discrete Lunar Lander environment...")
            env.test_dqn(args.o)
    else:
        env = C_environment(args.c, args.w)
        if args.train:
            print("Training off-policy actor-critic in continuous Lunar Lander environment...")
            env.train_continuous(args.o)
            env.test_continuous(args.o)
        elif args.test:
            print("Testing off-policy actor-critic in continuous Lunar Lander environment...")
            env.test_continuous(args.o)
    
    env.close()