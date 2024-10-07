import argparse

from woodnet.evaluate import run_evaluation_experiment
from woodnet.parsing import cli
from woodnet.predict import run_prediction_experiment
from woodnet.train import run_training_experiment, run_training_experiment_batch


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



def create_global_torchconf(args: argparse.Namespace) -> dict:
    """Create the thread-specific global PyTorch configuration from the CLI namespace."""
    # if only the first option `torch_num_threads` is set, use this count for the
    # second `torch_num_interop_threads` as well 
    if args.torch_num_threads is not None and args.torch_num_interop_threads is None:
        torch_num_threads = args.torch_num_threads
        torch_num_interop_threads = torch_num_threads
    else:
        torch_num_threads = args.torch_num_threads
        torch_num_interop_threads = args.torch_num_interop_threads

    global_torchconf = {'torch_num_threads' : torch_num_threads,
                        'torch_num_interop_threads' : torch_num_interop_threads}
    return global_torchconf


def main() -> None:
    """Main woodnet package CLI entrypoint."""
    args = cli()
    global_torchconf = create_global_torchconf(args)

    if args.task_verb == 'train':
        run_training_experiment(args.configuration, global_torchconf=global_torchconf)

    elif args.task_verb == 'batchtrain':
        run_training_experiment_batch(args.configurations,
                                      global_torchconf=global_torchconf)

    elif args.task_verb == 'evaluate':
        use_amp = not args.no_amp
        use_inference_mode = not args.use_no_grad
        non_blocking_transfer = not args.blocking_transfer
        no_compile_override = True if args.compilation == 'never' else False
        run_evaluation_experiment(basedir=args.basedir,
                                  transforms_preset=args.transforms,
                                  batch_size=args.batchsize,
                                  device=args.device,
                                  dtype=args.dtype,
                                  use_amp=use_amp,
                                  use_inference_mode=use_inference_mode,
                                  non_blocking_transfer=non_blocking_transfer,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  pin_memory=args.pin_memory,
                                  no_compile_override=no_compile_override,
                                  inject_early_to_device=args.inject_early_to_device,
                                  global_torchconf=global_torchconf
                                  )

    elif args.task_verb == 'predict':
        run_prediction_experiment(args.configuration,
                                  global_torchconf=global_torchconf)
        
    else:
        raise RuntimeError(f'invalid task action \'{args.task}\'')
    

if __name__ == '__main__':
    main()