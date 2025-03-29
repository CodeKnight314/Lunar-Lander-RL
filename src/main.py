from discrete_env import D_environment
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Q Network')
    parser.add_argument('--c', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--o', type=str, default='output', help='Path to the output directory')
    parser.add_argument('--w', type=str, default=None, help='Path to the weights file')
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    if args.train:
        env = D_environment(args.c, args.w)
        env.train_dqn(args.o)
        env.test_dqn(args.o)
    else: 
        env = D_environment(args.c, args.w)
        env.test_dqn(args.o)
    env.close()