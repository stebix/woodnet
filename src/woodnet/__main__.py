import rich
from woodnet.datasets.setup import INSTANCE_MAPPING

def main_info():
    rich.print(INSTANCE_MAPPING)


if __name__ == '__main__':
    main_info()
