import argparse
import pathlib
from collections import OrderedDict

import random
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from torchvision import transforms

import sigver.datasets.util as util
from sigver.featurelearning.data import TransformDataset
import sigver.featurelearning.models as models



def train(t_base_model: torch.nn.Module,
          s_base_model: torch.nn.Module,
          t_classification_layer: torch.nn.Module,
          s_classification_layer: torch.nn.Module,
          t_forg_layer: torch.nn.Module,
          s_forg_layer: torch.nn.Module,
          d_train_loader: torch.utils.data.DataLoader,
          c_train_loader: torch.utils.data.DataLoader,
          d_val_loader: torch.utils.data.DataLoader,
          c_val_loader: torch.utils.data.DataLoader,
          device: torch.device,
          args: Any,
          logdir: Optional[pathlib.Path]):
   

    # Collect the Student parameters that need to be optimizer
    parameters = list(s_base_model.parameters()) + list(s_classification_layer.parameters())

    # Initialize optimizer and learning rate scheduler
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                          nesterov=True, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             args.epochs // args.lr_decay_times,
                                             args.lr_decay)

    best_acc = 0
    best_acc_std = None
    best_epoch = None
    best_params = get_parameters(s_base_model, s_classification_layer, s_forg_layer)

    print('* Data batch information:')
    print('Distillation dataset number of batches:', len(d_train_loader))
    if args.c_dataset_path is not None:
        print('Continual dataset number of batches:', len(c_train_loader))

    for epoch in range(args.epochs):

        # Train one epoch; evaluate on validation
        train_epoch(d_train_loader, c_train_loader, t_base_model, s_base_model, 
                    t_classification_layer, s_classification_layer, 
                    epoch, optimizer, lr_scheduler, device, args)

        d_val_metrics = test(d_val_loader, s_base_model, s_classification_layer, device)
        d_val_acc, d_val_loss = d_val_metrics

        if args.c_dataset_path is not None:
            c_val_metrics = test(c_val_loader, s_base_model, s_classification_layer, device, 
                                   start=t_classification_layer.out_features)
            c_val_acc, c_val_loss = c_val_metrics


        if args.c_dataset_path is not None:
            val_acc = (d_val_acc + c_val_acc)/2.0
            val_acc_std = np.std([d_val_acc, c_val_acc])
        else:
            val_acc = d_val_acc
            val_acc_std = 0.0


        # Save the best model only on improvement (early stopping)
        if epoch == 0:
            best_acc = val_acc
            best_acc_std = val_acc_std
            best_epoch = epoch
            best_params = get_parameters(s_base_model, s_classification_layer, s_forg_layer)
            if logdir is not None:
                torch.save(best_params, logdir / 'model_best.pth')

        elif val_acc >= best_acc:
            if val_acc > best_acc:
                best_acc = val_acc
                best_acc_std = val_acc_std
                best_epoch = epoch
                best_params = get_parameters(s_base_model, s_classification_layer, s_forg_layer)
                if logdir is not None:
                    torch.save(best_params, logdir / 'model_best.pth')
            else:
                #Tiebreaker by standard deviation
                if val_acc_std <= best_acc_std:
                    best_acc = val_acc
                    best_acc_std = val_acc_std
                    best_epoch = epoch
                    best_params = get_parameters(s_base_model, s_classification_layer, s_forg_layer)
                    if logdir is not None:
                        torch.save(best_params, logdir / 'model_best.pth')


        print('Epoch {}, Distill   set: Val loss: {:.4f}, Val acc: {:.2f}%'.format(epoch, d_val_loss, d_val_acc * 100))
        if args.c_dataset_path is not None:
            print('Epoch {}, Continual set: Val loss: {:.4f}, Val acc: {:.2f}%'.format(epoch, c_val_loss, c_val_acc * 100))
            print('Epoch {}, Total acc: {:.2f}%'.format(epoch, val_acc * 100))

        if logdir is not None:
            current_params = get_parameters(s_base_model, s_classification_layer, s_forg_layer)
            torch.save(current_params, logdir / 'model_last.pth')

    print('Best model was obtained in epoch number {} with total acc: {:.2f}%'.format(best_epoch, best_acc * 100))
    return best_params


def copy_to_cpu(weights: Dict[str, Any]):
    return OrderedDict([(k, v.cpu()) for k, v in weights.items()])


def get_parameters(base_model, classification_layer, forg_layer):
    best_params = (copy_to_cpu(base_model.state_dict()),
                   copy_to_cpu(classification_layer.state_dict()),
                   copy_to_cpu(forg_layer.state_dict()))
    return best_params


def train_epoch(d_train_loader: torch.utils.data.DataLoader,
                c_train_loader: torch.utils.data.DataLoader,
                t_base_model: torch.nn.Module,
                s_base_model: torch.nn.Module,
                t_classification_layer: torch.nn.Module,
                s_classification_layer: torch.nn.Module,
                epoch: int,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                args: Any):
    """ Trains the network for one epoch

        Returns
        -------
        None
        """

    step = 0
    n_steps = len(d_train_loader)

    #Distill from real dataset
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)

    #Continual learning dataset iterator
    if c_train_loader is not None:
        c_iterator = iter(c_train_loader)


    for d_batch in d_train_loader:

        # A) Knowledge distillation loss computation
        d_x, d_y = d_batch[0], d_batch[1]

        d_x = torch.tensor(d_x, dtype=torch.float).to(device)
        d_y = torch.tensor(d_y, dtype=torch.long).to(device)
        d_yforg = torch.tensor(d_batch[2], dtype=torch.float).to(device)

        # Forward propagation (student and teacher)
        with torch.no_grad():
            d_t_features = t_base_model(d_x)
            d_t_logits = t_classification_layer(d_t_features)
        
        d_s_features = s_base_model(d_x)
        d_s_logits = s_classification_layer(d_s_features)

        Scol = d_s_logits.shape[1]
        Tcol = d_t_logits.shape[1]
        Tlin = d_t_logits.shape[0]

        if Tcol < Scol:
            d_t_logits = torch.cat((d_t_logits, torch.zeros(Tlin, Scol - Tcol, device=device)), 1)

        #Compute KL divergence loss
        T = args.temperature
        P = F.log_softmax(d_s_logits / T, dim=1)
        Q = F.softmax(d_t_logits / T, dim=1)

        divergence_loss = kl_loss(P, Q)

        # B) Classification loss (for continual learning)
        if c_train_loader is not None:
        
            c_batch = next(c_iterator)
            c_x, c_y = c_batch[0], c_batch[1]
            c_x = torch.tensor(c_x, dtype=torch.float).to(device)
            c_y = torch.tensor(c_y, dtype=torch.long).to(device)
            c_yforg = torch.tensor(c_batch[2], dtype=torch.float).to(device)

            # Forward propagation (only student)
            c_s_features = s_base_model(c_x)

            # Cross entropy classification loss
            c_s_logits = s_classification_layer(c_s_features)

            with torch.no_grad():
                c_t_features = t_base_model(c_x)
                c_t_logits = t_classification_layer(c_t_features)

            P2 = F.log_softmax(c_s_logits[:,:Tcol] / T, dim=1)
            Q2 = F.softmax(c_t_logits[:,:Tcol] / T, dim=1)
            second_divergence_loss = kl_loss(P2, Q2)

            classification_loss = F.cross_entropy(c_s_logits[:,Tcol:], c_y)

           
        #Compute total loss
        if c_train_loader is not None:
            loss = (args.d_lamb * divergence_loss) + (args.c_lamb * classification_loss) + (args.p_lamb * second_divergence_loss)
        else:
            loss = divergence_loss

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10)

        # Update weights
        optimizer.step()

        step += 1
    
    lr_scheduler.step()


def test(val_loader: torch.utils.data.DataLoader,
         s_base_model: torch.nn.Module,
         s_classification_layer: torch.nn.Module,
         device: torch.device,
         start: int = None) -> Tuple[float, float, float, float]:
    """ Test the model in a validation/test set

    Parameters
    ----------
    val_loader: torch.utils.data.DataLoader
        Iterable that loads the validation set (x, y) tuples
    s_base_model: torch.nn.Module
        The student model architecture that "extract features" from signatures
    s_classification_layer: torch.nn.Module
        The student classification layer (from features to predictions of which user
        wrote the signature)
    device: torch.device
        The device (CPU or GPU) to use for training

    Returns
    -------
    float, float
        The valication accuracy and validation loss

    """
    val_losses = []
    val_accs = []

    for batch in val_loader:
        x, y, yforg = batch[0], batch[1], batch[2]
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        if start is not None:
            y = y + start
        yforg = torch.tensor(yforg, dtype=torch.float).to(device)

        with torch.no_grad():
            features = s_base_model(x)
            logits = s_classification_layer(features[yforg == 0])

            loss = F.cross_entropy(logits, y[yforg == 0])
            pred = logits.argmax(1)
            acc = y[yforg == 0].eq(pred).float().mean()

        val_losses.append(loss.item())
        val_accs.append(acc.item())
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)

    return val_acc, val_loss


def main(args):

    logdir = pathlib.Path(args.logdir)
    if not logdir.exists():
        logdir.mkdir()


    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: {}'.format(device))

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print('* Loading Distillation Data...')

    d_x, d_y, d_yforg, d_usermapping, d_filenames = util.load_dataset(args.d_dataset_path)
    d_data = util.get_subset((d_x, d_y, d_yforg), subset=range(*args.d_users))
    d_data = util.remove_forgeries(d_data, forg_idx=2)

    d_len = len(d_data[0])
    print('Distillation dataset lenght is:', d_len)


    if args.c_dataset_path is not None:
        print('* Loading Continual Learning Data...')

        c_x, c_y, c_yforg, c_usermapping, c_filenames = util.load_dataset(args.c_dataset_path)
        c_data = util.get_subset((c_x, c_y, c_yforg), subset=range(*args.c_users))
        c_data = util.remove_forgeries(c_data, forg_idx=2)

        c_len = len(c_data[0])
        print('Continual dataset lenght is:', c_len)

    else:
        print('Distillation WITHOUT Continual Learning has been performed.')

    if args.c_dataset_path is not None:
        if(d_len > c_len):
            print('WARNING!!: This implementation requires that continual learning dataset MUST BE LONGER than distillation dataset.')
    
    if args.c_dataset_path is not None:
        print('* Setting Up Continual Learning dataset DataLoader...')
        c_train_loader, c_val_loader, c_len_train_set = c_setup_data_loaders(c_data, args.c_batch_size, args.input_size)
    else:
        print('There is NO Continual Learning DataLoader.')
        c_train_loader = None
        c_val_loader = None
        c_len_train_set = None

    print('* Setting Up Distill dataset DataLoader...')
    d_train_loader, d_val_loader = d_setup_data_loaders(d_data, args.d_batch_size, args.input_size, 
                                                         num_samples=c_len_train_set, d_transform=args.d_transform)


    print('* Initializing Teacher Model')

    t_n_classes = 531

    t_base_model = models.available_models[args.t_model]().to(device)
    t_classification_layer = nn.Linear(t_base_model.feature_space_size, t_n_classes).to(device)
    t_forg_layer = nn.Module()  # Stub module with no parameters

    t_base_model_params, t_classification_params, t_forg_params = torch.load(args.t_checkpoint)
    t_base_model.load_state_dict(t_base_model_params)
    t_classification_layer.load_state_dict(t_classification_params)

    print('Teacher weights are loaded from:', args.t_checkpoint)
    t_base_model.eval()
    t_classification_layer.eval()
    t_forg_layer.eval()


    print('* Initializing Student Model')

    if args.c_dataset_path is not None:
        s_n_classes = t_n_classes + len(np.unique(c_data[1]))
    else:
        s_n_classes = t_n_classes

    s_base_model = models.available_models[args.s_model]().to(device)
    print('SigNet architecture based model has been created.')   

    s_classification_layer = nn.Linear(s_base_model.feature_space_size, s_n_classes).to(device)
    s_forg_layer = nn.Module()  # Stub module with no parameters

    s_base_model.train()
    s_classification_layer.train()
    s_forg_layer.train()

    if args.s_checkpoint is not None:
        s_base_model_params, s_classification_params, s_forg_params = torch.load(args.s_checkpoint)
        s_base_model.load_state_dict(s_base_model_params)
    
        if args.c_dataset_path is None:
            s_classification_layer.load_state_dict(s_classification_params)
        else:
            with torch.no_grad():
                s_classification_layer.weight[:531,:] = s_classification_params['weight']
                s_classification_layer.bias[:531] = s_classification_params['bias']

        print('Student weights are loaded from:', args.s_checkpoint)
    else:
        print('Student weights are initialized from scratch.')


    if args.test:
        print('ONLY Testing Student mode.')

        print('Accuracy concerning the distillation dataset:')
        val_acc, val_loss = test(d_val_loader, s_base_model, s_classification_layer,
                                                              device)

        print('Val loss: {:.4f}, Val acc: {:.2f}%'.format(val_loss, val_acc * 100))


        if args.c_dataset_path is not None:
            print('Accuracy concerning the continual learning dataset:')

            val_acc, val_loss = test(c_val_loader, s_base_model, s_classification_layer,
                                                                  device, start=t_classification_layer.out_features)

            print('Val loss: {:.4f}, Val acc: {:.2f}%'.format(val_loss, val_acc * 100))


    else:
        print('Training Student distilling from Teacher ...')
        train(t_base_model, s_base_model, t_classification_layer, s_classification_layer, t_forg_layer, s_forg_layer,
            d_train_loader, c_train_loader, d_val_loader, c_val_loader,
              device, args, logdir)

def d_setup_data_loaders(data, batch_size, input_size, num_samples=None, d_transform=False):
    
    y = data[1] - 350

    data = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(y), torch.from_numpy(data[2]))
    train_size = int(0.9 * len(data))
    sizes = (train_size, len(data) - train_size)
    train_set, test_set = random_split(data, sizes)

    if d_transform:
        print('Transformations have been applied to the distillation set.')
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        train_set = TransformDataset(train_set, train_transforms)
        val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        test_set = TransformDataset(test_set, val_transforms)
    else:
        print('Distillation set is NOT transformed.')

    if num_samples is None:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    else:
        sampler = RandomSampler(train_set, replacement=True, num_samples=num_samples)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)

    val_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader


def c_setup_data_loaders(data, batch_size, input_size):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[1])
    data = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(y), torch.from_numpy(data[2]))
    train_size = int(0.9 * len(data))
    sizes = (train_size, len(data) - train_size)
    train_set, test_set = random_split(data, sizes)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
    ])
    train_set = TransformDataset(train_set, train_transforms)
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    test_set = TransformDataset(test_set, val_transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, len(train_set)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser('Distillation with Continual Learning Support')
    argparser.add_argument('--d-dataset-path', help='Path containing a numpy file with images and labels for Distillation', default='data/data.npz')
    argparser.add_argument('--c-dataset-path', help='Path containing a numpy file with images and labels for Continual learning')
    argparser.add_argument('--d-transform', action='store_true', help='Apply transformations on distillation data')

    argparser.add_argument('--input-size', help='Input size (cropped)', nargs=2, type=int, default=(150, 220))
    
    argparser.add_argument('--d-users', nargs=2, type=int, default=(350, 881))
    argparser.add_argument('--c-users', nargs=2, type=int, default=(2000, 9950))

    argparser.add_argument('--t-model', help='Teacher Model architecture', default='signet', choices=models.available_models)
    argparser.add_argument('--s-model', help='Student Model architecture', choices=models.available_models, required=True)
    
    argparser.add_argument('--d-batch-size', help='Batch size', type=int, default=32)
    argparser.add_argument('--c-batch-size', help='Batch size', type=int, default=32)

    argparser.add_argument('--lr', help='learning rate', default=0.001, type=float)
    argparser.add_argument('--lr-decay', help='learning rate decay (multiplier)', default=0.1, type=float)
    argparser.add_argument('--lr-decay-times', help='number of times learning rate decays', default=3, type=float)
    argparser.add_argument('--momentum', help='momentum', default=0.90, type=float)
    argparser.add_argument('--weight-decay', help='Weight Decay', default=1e-4, type=float)
    argparser.add_argument('--epochs', help='Number of epochs', default=60, type=int)
    
    argparser.add_argument('--t-checkpoint', help='Teacher starting weights (pth file)', default='models/signet.pth')
    argparser.add_argument('--s-checkpoint', help='Student starting weights (pth file)')
    
    argparser.add_argument('--test', action='store_true')

    argparser.add_argument('--seed', default=42, type=int)

    argparser.add_argument('--d-lamb', type=float, default=1.0, help='Weight for distillation')
    argparser.add_argument('--c-lamb', type=float, default=1.0, help='Weight for cross entropy')
    argparser.add_argument('--p-lamb', type=float, default=1.7, help='Weight for minimizing divergence')
    argparser.add_argument('--temperature', type=float, default=1.0, help='Distillation softmax temperature')


    argparser.add_argument('--gpu-idx', default=0, type=int)
    argparser.add_argument('--logdir', help='logdir', required=True)


    argparser.set_defaults(test=False)
    arguments = argparser.parse_args()
    print(arguments)


    main(arguments)
