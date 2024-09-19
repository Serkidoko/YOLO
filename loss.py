import torch.nn.functional as F

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        
    def forward(self, preds, targets):
        
        
    