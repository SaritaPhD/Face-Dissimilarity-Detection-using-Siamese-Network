import torch
from torch import optim
from src.network import SiameseNetwork, ContrastiveLoss
from src.dataset import SiameseNetworkDataset
from torch.utils.data import DataLoader

def train_model(train_dataloader, num_epochs=100, learning_rate=0.0005):
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(num_epochs):
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()

            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss_contrastive.item()}")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    return net, loss_history, counter
