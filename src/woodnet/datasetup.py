import argparse
import pathlib
import rich.console

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Configure the data loading system by setting '
                                                 'the location of the data. Note that any preexisting '
                                                 'data configuration paths will be overwritten.')
    parser.add_argument('data_configuration_path', type=str, help='Path to the data configuration file.')
    return parser


def create_env_file(data_conf_path: str | pathlib.Path) -> argparse.ArgumentParser:
    """
    Create or overwrite .env file in repository root and write the data configuration path item to it.
    """
    env_file_path = pathlib.Path(__file__).parents[2] / '.env'
    with open(env_file_path, mode='w') as env_file:
        env_file.write(f'DATA_CONFIGURATION_PATH={data_conf_path}')
    return env_file_path


def cli() -> None:
    parser = create_parser()
    args = parser.parse_args()
    env_file_path = create_env_file(args.data_configuration_path)

    console = rich.console.Console()
    console.rule("[bold red]Configuration Result Report[/bold red]")
    console.print(f'Created environment file at: \'{env_file_path}\'', justify='left')
    console.print(f'Data configuration path set to: \'{args.data_configuration_path}\'', justify='left')
    return args


def main():
    cli()


if __name__ == '__main__':
    main()