import argparse

from woodnet.train import run_training_experiment, run_training_experiment_batch
from woodnet.predict import run_prediction_experiment

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Main entrypoint to perform training '
                                                 'and prediction runs with the woodnet '
                                                 'package.')
    parser.add_argument('task', type=str, choices=['train', 'batchtrain', 'predict'],
                        help='Choose training and prediction action.')

    parser.add_argument('configuration', type=str, nargs='+',
                        help='Location of the configuration YAML file that fully defines '
                             'the experiment run. Must match the beforehand selected action '
                             'and can thus be a training or prediction configuration.')
    
    parser.add_argument('--torch_num_threads', '-t', type=int, default=None,
                        help='Set number of torch CPU thread count parallelism. If not set, '
                             'value will be read from environment variable or basal '
                             'fallback value.')

    parser.add_argument('--torch_num_interop_threads', type=int, default=None,
                        help='Set number of torch CPU thread count interop parallelism. '
                             'If not set, value will be read from environment variable '
                             'or basal fallback value.')

    return parser


def cli() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args()
    return args


def main() -> None:
    args = cli()

    # if only the first option `torch_num_threads` is set, use this count globally
    if args.torch_num_threads is not None and args.torch_num_interop_threads is None:
        torch_num_threads = args.torch_num_threads
        torch_num_interop_threads = torch_num_threads
    else:
        torch_num_threads = args.torch_num_threads
        torch_num_interop_threads = args.torch_num_interop_threads

    global_torchconf = {'torch_num_threads' : torch_num_threads,
                        'torch_num_interop_threads' : torch_num_interop_threads}

    if args.task == 'train':
        if len(args.configuration) != 1:
            raise RuntimeError('train verb requires singular configuration specification')
        run_training_experiment(args.configuration[0], global_torchconf=global_torchconf)
    elif args.task == 'batchtrain':
        # TODO: maybe more clean if we include subparser for multiple number of arguments
        run_training_experiment_batch(args.configuration,
                                      global_torchconf=global_torchconf)
    elif args.task == 'predict':
        run_prediction_experiment(args.configuration,
                                  global_torchconf=global_torchconf)
        
    else:
        raise RuntimeError(f'invalid task action \'{args.task}\'')
    



if __name__ == '__main__':
    main()