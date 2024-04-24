"""
Functionality for the top-level command line interface
of the woodnet package.

@jsteb 2024
"""
import argparse


def configure_globopts(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the incoming `parser` with global options/parameters that are
    required for every sub-task specific parser.
    """
    parser.add_argument('--torch-num-threads', '-t', type=int, default=None,
                        help='Set number of torch CPU thread count parallelism. If not set, '
                             'value will be read from environment variable or basal '
                             'fallback value.')
    parser.add_argument('--torch-num-interop_threads', type=int, default=None,
                        help='Set number of torch CPU thread count interop parallelism. '
                             'If not set, value will be read from environment variable '
                             'or basal fallback value.')
    return parser


def configure_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """In-place modify the `parser` to work as a single training run parser."""
    parser.add_argument('configuration', type=str,
                        help='Location of the configuration YAML file that fully defines '
                             'the single training experiment run.')
    configure_globopts(parser)
    return parser


def configure_batchtrain_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """In-place modify the `parser` to work as a batch training run parser."""
    parser.add_argument('configurations', type=str, nargs='+',
                        help='Location of the configuration YAML files that fully define '
                             'the experiment run. Can be a variable number of paths pointing '
                             'to configuration files or paths to directories. '
                             'Directories will be crawled for YAML configuration files.')
    configure_globopts(parser)
    return parser


def configure_evaluate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """In-place modify the `parser` to work as a evaluation experiment run parser."""
    parser.add_argument('basedir', type=str,
                        help='Path to the basal experiment dir that holds the '
                             'cross validation fold-wise experiment runs.')
    parser.add_argument('transforms', type=str,
                        help='Specify the transformations for the robustness evaluation '
                             'as an ID name for the presets.')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Sets the inference loader batch size')
    parser.add_argument('--num-workers', '-n', type=int, default=0,
                        help='Set the worker process count for the PyTorch DataLoader.')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Set the inference device. Canb be \'cpu\' or \'cuda:$N\'')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision (amp).')
    parser.add_argument('--use-no-grad', action='store_true',
                        help='Explicitly use no_grad mode during inference run. '
                             'If not set (i.e. per default), the inference_mode is used.')
    parser.add_argument('--blocking-transfer', action='store_true',
                        help='Sets blocking behaviour of the host-device data transfer.')
    parser.add_argument('--dtype', type=str, choices=('float32', 'float64'), default='float32',
                        help='Select the basal float data type of the evaluation run.')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Set the memory pinning behaviour of the data loader.')
    parser.add_argument('--compilation', type=str, choices=['never', 'useconf'],
                        help='Set PyTorch model compilation behaviour. The option \'never\' '
                             'fully disables compilation. For the setting \'useconf\' '
                             'compilation options and kwargs are deduced from the training '
                             'configuration.')
    parser.add_argument('--save-protocols', type=str, default='both',
                        choices=['json', 'pickle', 'all'],
                        help='Set the saving protocol for the results dictionary. '
                             'Defaults to \'all\', i.e. saving with all available protocols. '
                             'Currently this is as JSON file and as pickled Python object.')
    parser.add_argument('--inject-early-to-device', '-j', action='store_true',
                        help='Enables to-device transfer of the tensors as first action '
                             'in the transformation pipeline. Use with care: only supported '
                             'with single thread data loading (num_workers=0) and subsequent '
                             'transforms that run on device-located tensors. May also lead to '
                             'OOM problems.')
    configure_globopts(parser)
    return parser



def create_parser() -> argparse.ArgumentParser:
    mainparser = argparse.ArgumentParser(description='Main entrypoint to perform training '
                                                     'and prediction runs with the woodnet '
                                                     'package.')
    subparser = mainparser.add_subparsers(dest='task_verb',
                                          help='Task selection: choose between training '
                                               'and evaluation tasks.')
    
    train_parser = subparser.add_parser('train', help='Perform single configuration training.')
    batchtrain_parser = subparser.add_parser('batchtrain', help='Perform batched multi-'
                                                                'configuration training.')
    evaluate_parser = subparser.add_parser('evaluate', help='Perform evaluation experiment.')
    # add the task-spcific options ot the subparsers
    configure_train_parser(train_parser)
    configure_batchtrain_parser(batchtrain_parser)
    configure_evaluate_parser(evaluate_parser)

    return mainparser



def cli() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args()
    return args


