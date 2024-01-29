import argparse
import rich

from woodnet.configtools.foldgeneration import refold_file

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Tooling for modifying the cross validation '
                                                 'folds.')
    
    parser.add_argument('source', type=str, help='Path to source configuration file.')
    parser.add_argument('target', type=str, help='Path to location where the new '
                                                 'refolded configuration file will be written.')

    parser.add_argument('--strategy', '-s', type=str, required=True,
                        help='Choose the cross validation strategy that produces the splits.')

    parser.add_argument('--foldnum', '-n', type=int, required=True,
                        help='Number of the resulting fold,')
    
    parser.add_argument('--skip-validation', '-v', action='store_true',
                        help='Flag to skip validation via the pydantic model.')

    parser.add_argument('--force-write', '-f', action='store_true',
                        help='Flag to overwrite any preeixting file at the target location.')
    return parser


def cli() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args()
    return args


def main() -> None:
    args = cli()

    validate = not args.skip_validation
    force_write = args.force_write

    configuration = refold_file(source_path=args.source, target_path=args.target,
                                strategy=args.strategy, foldnum=args.foldnum,
                                validate=validate, force_write=force_write)
    
    rich.print(configuration)

    



if __name__ == '__main__':
    main()