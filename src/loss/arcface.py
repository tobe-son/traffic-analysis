import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLayer(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss Layer
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        # 学習可能なクラス中心ベクトル (Weight)
        # bias=False は必須 (角度のみで判定するため)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # [cite_start]1. Normalize Features & Weights (超球面上への射影) [cite: 38, 197-198]
        # features: (Batch, Dim)
        # weight: (NumClasses, Dim)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        
        # 数値安定性のためのクリップ
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 2. Additive Angular Margin
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        
        # One-hotエンコーディングで正解クラスのみマージンを適用
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = one_hot * target_logits + (1.0 - one_hot) * cosine
        
        # 3. Scaling
        output *= self.s
        
        return output