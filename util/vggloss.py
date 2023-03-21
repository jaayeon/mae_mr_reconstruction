from torchvision.models import vgg19
import torch.nn as nn
import torchvision.transforms as transforms


class perceptualloss(nn.Module):
    def __init__(self):
        super(perceptualloss, self).__init__()
        vgg19_model = vgg19(pretrained=True).to('cuda')
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()
        self.transform = transforms.Compose([transforms.Resize(224)])
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        N = x.size(0)
        x = self.normalize(x)
        y = self.normalize(y)
        out_x = self.feature_extractor(x)
        out_y = self.feature_extractor(y)
        loss = self.criterion(out_x, out_y)/N
        return loss

    def normalize(self, x):
        if x.size(1)==1:
            x=x.repeat(1,3,1,1)
        x=self.transform(x)
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for i, (m,s) in enumerate(zip(mean, std)):
            x[:,i:i+1,:,:] = (x[:,i:i+1,:,:] - m)/s
        return x