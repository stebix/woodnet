import torch

from tqdm.auto import tqdm

DataLoader = torch.utils.data.DataLoader


def validate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             validationloader: DataLoader,
             device: torch.device = 'cpu',
             dtype: torch.dtype = torch.float32) -> float:
    """Validate model, duh."""
    model.eval()
    validation_loss_history = []
    wrapped_loader = tqdm(validationloader, unit='bt',
                          desc='ValLoader', leave=False)
    with torch.no_grad():
        for batch in wrapped_loader:
            data, label = batch
            data = data.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=dtype)

            prediction = model(data)
            loss = criterion(prediction, label)

            validation_loss_history.append(loss.item())
    
    return validation_loss_history




def train(model: torch.nn.Module,
          train_iters: int,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          trainloader: DataLoader,
          validationloader: DataLoader,
          validate_every_n_epoch: int,
          dtype: torch.dtype = torch.float32,
          device: torch.device = 'cpu') -> None:


    train_iter = 0
    current_epoch = 0

    train_loss_history = []

    # torch setup
    model.train(True)
    device = torch.device(device)
    total_pbar = tqdm(total=train_iters, unit='iter', desc='TrainIter')

    while train_iter <= train_iters:
        # epoch happens inside this loop
        for batch in tqdm(trainloader, unit='bt', desc='LoaderIter', leave=False):
            # data moving
            data, label = batch
            data = data.to(device=device, dtype=dtype, non_blocking=True)
            label = label.to(device=device, dtype=dtype, non_blocking=True)
            # actual deep learning
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            # bookkeeping
            train_loss_history.append(loss.item())
            total_pbar.update()

        current_epoch += 1

        if current_epoch % validate_every_n_epoch == 0:
            validate(model, validationloader)
