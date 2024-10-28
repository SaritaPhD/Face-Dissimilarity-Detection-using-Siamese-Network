import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from src.utils import imshow

def test_model(net, test_dataloader):
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    
    for i in range(10):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = net(x0.cuda(), x1.cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
