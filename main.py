import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .data.dataset import ECSSD
from .data.transform import RGB2ChPairs
from .model.model import CoSOV1Network

db_root = "data/salient/ECSSD"

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
      f'Accuracy: {correct}/{len(test_loader.dataset)} '
      f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


def main():
    parser = argparse.ArgumentParser(
        description='CoSOV1 Implementation')
    parser.add_argument('--batch-size',
                        type=int, default=4,
                        help='input batch size for training, default=4')
    parser.add_argument('--epochs',
                        type=int,
                        default=480,
                        help='number of epochs to train, default=480')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help='random seed, default=123')
    parser.add_argument('--log-interval',
                        type=int,
                        default=15,
                        help='# of batches to wait before logging')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=True,
                        help='bool to save the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}

    img_transforms = transforms.Compose([
        transforms.Resize(size=(384, 384)),
        transforms.Normalize(),
        transforms.ToTensor(),
        RGB2ChPairs()
    ])

    train_dataset = ECSSD(db_root,transforms=img_transforms)

    train_loader = DataLoader(train_dataset, 
                            num_workers=1, 
                            batch_size=1, 
                            shuffle=False,
                            transforms=img_transforms)

    model = CoSOV1Network().to(device)
    optimizer = optim.rmsprop(model.parameters(),
                              lr=args.lr,
                              centered=True)

    scheduler = StepLR(optimizer, step_size=240, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "CoSOV1.pt")

if __name__ == "__main__":
    main()
    print("PASS")
