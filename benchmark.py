import argparse
import rich

from woodnet.benchmarking.bench_threads import run_benchmark

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run benchmarks to evaluate runtime influence '
                                                 'various configuration options.')
    
    parser.add_argument('flavor', type=str, choices={'end2end', 'load_only'}, help='Benchmark flavor.')

    parser.add_argument('--torch_num_threads', '-t', type=int, required=True,
                        help='General PyTorch thread count.')
    
    parser.add_argument('--torch_num_interop_threads', '-i', type=int, required=False,
                        help='General PyTorch thread count.', default=-1)

    parser.add_argument('--tileshape', '-s', type=str, required=False,
                        help='Tileshape for volumetric data chunks.', default='64,64,64')

    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        help='Batch size.', default=2)
    
    parser.add_argument('--num_workers', '-w', type=int, default=0,
                        help='Number of workers in the data loader.')

    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device for inference purposes.')

    parser.add_argument('--size', '-z', type=int, default=None,
                        help='Subsample this number of elements from the '
                             'benchmark dataset. Lower for lower overall benchmark time, '
                             'higher for more stable/informative results.')
    return parser


def cli() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args()
    return args


def main() -> None:
    args = cli()

    if args.torch_num_interop_threads == -1:
        torch_num_interop_threads = args.torch_num_threads
    else:
        torch_num_interop_threads = args.torch_num_interop_threads

    run_benchmark(
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
        tileshape=args.tileshape,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        size=args.size,
        flavor=args.flavor
    )
    

if __name__ == '__main__':
    main()