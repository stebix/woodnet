import argparse

from woodnet.train import run_training_experiment
from woodnet.predict import run_prediction_experiment

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Main entrypoint to perform training '
                                                 'and prediction runs with the woodnet '
                                                 'package.')
    parser.add_argument('task', type=str, choices=['train', 'predict'],
                        help='Choose training and prediction action.')

    parser.add_argument('configuration', type=str,
                        help='Location of the configuration YAML file that fully defines '
                             'the experiment run. Must match the beforehand selected action '
                             'and can thus be a training or prediction configuration.')
    return parser


def cli() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args()
    return args


def main() -> None:
    args = cli()

    if args.task == 'train':
        run_training_experiment(args.configuration)
    elif args.task == 'predict':
        run_prediction_experiment(args.configuration)
        
    else:
        raise RuntimeError(f'invalid task action \'{args.task}\'')
    



if __name__ == '__main__':
    main()