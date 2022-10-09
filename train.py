import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from mobilenet import MobilenetV1 ,MobilenetV1tiny 
from mobilenet2 import MobilenetV2 ,MobilenetV2tiny 
import argparse

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    print('load data')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if(args.version==1):
        network = MobilenetV1tiny(len(classes)).to(device=device)
    else:
        network = MobilenetV2tiny(len(classes)).to(device=device)

    your_model = network.cuda()
    summary(your_model, input_size=(3, 32, 32))

    PATH = './mobilenet{}.pth'.format(args.version)
    print('----------train start----------')
    if(args.train):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(args.epoch):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device=device), data[1].to(device=device)

                optimizer.zero_grad()

                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(network.state_dict(), PATH)
    
    network.load_state_dict(torch.load(PATH))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device=device), data[1].to(device=device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',type=int, default=1)
    parser.add_argument('--train',type=bool, default=False)
    parser.add_argument('--epoch',type=int, default=2)

    args = parser.parse_args()
    main(args)