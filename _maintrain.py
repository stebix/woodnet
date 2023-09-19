import torch.utils.data.dataloader

from ruamel.yaml import YAML

DataLoader = torch.utils.data.DataLoader




testconf = """
loaders:
  dataset: TileDataset
  tileshape: [256, 256, 256]

  training:
    datasets: [13, 3, 5, 7, 6]
    transformer:
      - name: Normalize3D
        mean: 110
        std: 950
    
      - name: Rotate90
        dims: [1, 2]

      - name: Rotate90
        dims: [1, 2]

  validation:
    datasets: [9, 12, 18, 19]
    transformer:
      - name: Normalize3D
        mean: 110
        std: 950        
"""


def get_builder(configuration: dict) -> type[None]:
    pass


def create_loaders(configuration: dict) -> dict[str, DataLoader]:
    """
    Create the dataloader pair directly from the configuration file.
    """



if __name__ == '__main__':

    yaml = YAML(typ='safe')
    data = yaml.load(testconf)
    print(data)