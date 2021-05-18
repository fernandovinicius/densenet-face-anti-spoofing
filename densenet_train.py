from torchvision import datasets
from utils import save_models_metrics
from utils import load_densenet
from utils import TRAIN_DATA_TRANSFORM
from utils import VAL_DATA_TRANSFORM
import pandas as pd
import numpy as np
import argparse
import torch
import yaml
import time
import copy
import os


def main(model_name, pretrained_model_path, output_dir,dataset_dir, num_epochs, batch_size):

    # Check if output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_model_name = model_name + '_antispoofing'
    out_model_path = os.path.join(output_dir,
                                  out_model_name + '.pth')

    print(' ' + '-'*50 + '\n')
    print(' >> Training Info:\n')
    print(f'    Model name:             {model_name}')
    print(f'    Pre-trained model path: {pretrained_model_path}')
    print(f'    Dataset dir:            {dataset_dir}')
    print(f'    Epochs:                 {num_epochs}')
    print(f'    Batch Size:             {batch_size}')
    print(f'    Output model path:      {out_model_path}')
    print(' ' + '-'*50 + '\n')

    # Train current model
    model, hist, (train_labels, val_labels) = finetuning_routine(
        output_model_path=out_model_path,
        model_name=model_name,
        pretrained_model_path=pretrained_model_path,
        dataset_dir=dataset_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Save metrics in txt file
    metrics = [('_train_metrics.txt', train_labels),
               ('_val_metrics.txt', val_labels)]

    for (fname, labels) in metrics:
        metrics_fpath = os.path.join(output_dir,
                                     out_model_name + fname)
        save_models_metrics(
            model_name=out_model_name,
            metrics_filepath=metrics_fpath,
            y_true=labels[0],
            y_pred=labels[1],
        )

    # Save model training hist
    fname = os.path.join(output_dir,
                         out_model_name + '_training_hist.csv')
    df = pd.DataFrame.from_dict(hist)
    df.to_csv(fname, sep=';', index=False)


def finetuning_routine(output_model_path, model_name,
                       pretrained_model_path, dataset_dir,
                       num_epochs=1, batch_size=32):
    """
        DenseNet Finetuning

    Parameters
    ----------
        output_model_path : str
            Path where tuned model will be stored.

        model_name : str
            Pretrained model name.

        pretrained_model_path : str
            Path to pretrained model

        dataset_dir : str
            Training dataset directory.
            Must contain "train" and "val" subdirectories.

        num_epochs : int
            Number of training epochs.

        batch_size : int
            Batch size.

    Returns
    -------
        model_tuned : torchvision model
            Model tuned object.

        model_hist : dict
            Dictionary containing model loss and accuracy history.
            hist = {'loss': [...], 'accuracy': [...],
                    'val_loss': [...], 'val_accuracy': [...]}

        pred_labels : tuple
            Training and Validation labels
            pred_labels = ((y_train_true, y_train_pred),
                           (y_val_true, y_val_pred))
    """
    # Load model and Freeze parameters so we don't backprop through them
    model = load_densenet(path=pretrained_model_path)
    for param in model.parameters():
        param.requires_grad = True

    # Modifying FC layer
    model.classifier = torch.nn.Linear(1024, 2)

    # Hyperparameters train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.3)

    model_tuned, model_hist, (train_labels, val_labels) = train_model(
        model=model,
        dataset_dir=dataset_dir,
        criterion=criterion,
        optimizer=opt,
        scheduler=lr_sched,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
    )

    # Save model
    torch.jit.save(torch.jit.script(model_tuned), output_model_path)
    return model_tuned, model_hist, (train_labels, val_labels)


def train_model(model, dataset_dir, criterion, optimizer, scheduler,
                num_epochs=25, batch_size=32, device='gpu'):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      dataset_dir: Dataset dir
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    print(f'\n >> Training on {device}')

    # Dataset params
    dataloaders, dataset_sizes, _ = get_dataset_params(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=os.cpu_count()
    )
    print('\n >> Dataset:')
    print(f'    Train imgs: {dataset_sizes["train"]}')
    print(f'    Val imgs:   {dataset_sizes["val"]}')
    print(f'    Batch_size: {dataloaders["train"].batch_size}')

    # Start training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, val_loss, val_acc = [],  [], [], []

    since = time.time()
    for epoch in range(num_epochs):

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            phase_start = time.time()

            print(f' -- Phase: {phase}')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save 'loss' and 'acc'
            if device != "cpu":
                epoch_acc = epoch_acc.cpu().item()

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

                # Save best_model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print_time_elapsed(phase_start)
        print()

    time_elapsed = time.time() - since
    min, sec = divmod(time_elapsed, 60)
    print(f'Training complete in {min:.0f}m {sec:.0f}s')
    print(f'Best test Acc: {best_acc:4f}\n\n')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Compute labels for Train and Val datasets
    train_labels, val_labels = compute_dataset_preds(
        model=model,
        dataloader=dataloaders,
        device=device
    )

    # Model history
    model_hist = {
        'epoch': np.arange(num_epochs) + 1,
        'loss': train_loss,
        'accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }

    return model, model_hist, (train_labels, val_labels)


def compute_dataset_preds(model, dataloader, device):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    y_train_true, y_train_pred = np.array([]), np.array([])
    y_val_true, y_val_pred = np.array([]), np.array([])

    for phase in ['train', 'val']:
        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            _labels, _preds = labels.cpu().numpy(), preds.cpu().numpy()
            if phase == 'train':
                y_train_true = np.concatenate((y_train_true, _labels))
                y_train_pred = np.concatenate((y_train_pred, _preds))
            else:
                y_val_true = np.concatenate((y_val_true, _labels))
                y_val_pred = np.concatenate((y_val_pred, _preds))

    return (y_train_true, y_train_pred), (y_val_true, y_val_pred)


def print_time_elapsed(start_time):
    time_elapsed = time.time() - start_time
    min, sec = divmod(time_elapsed, 60)
    print(f'time_elapsed: {min:.0f}m {sec:.0f}s\n')


def get_dataset_params(dataset_dir, batch_size=32, num_workers=8):
    data_transforms = {
        'train': TRAIN_DATA_TRANSFORM,
        'val': VAL_DATA_TRANSFORM
    }

    dataset = {x: datasets.ImageFolder(os.path.join(dataset_dir, x),
                                       data_transforms[x])
               for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(dataset[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
    class_names = dataset['train'].classes

    return dataloaders, dataset_sizes, class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        help='Model name.',
                        default='densenet',
                        type=str)
    parser.add_argument('--pretrained-model-path',
                        help='Pretrained model full path.',
                        default=None,
                        type=str)
    parser.add_argument('--dataset-dir',
                        help='Training dataset directory.' +
                             'Must contain "real" and "spoof" subdirectories.',
                        type=str)
    parser.add_argument('--output-dir',
                        help='Dir where the trained model will be stored.',
                        default='./',
                        type=str)
    parser.add_argument('--num-epochs',
                        help='Training epochs. Default 25.',
                        default=25,
                        type=int)
    parser.add_argument('--batch-size',
                        help='Batch size. Default 32.',
                        default=32,
                        type=int)

    args = parser.parse_args()
    print(args)
    main(
        model_name=args.model_name,
        pretrained_model_path=args.pretrained_model_path,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
