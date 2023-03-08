import torch
import torch.nn as nn
import torch.nn.functionl as F
from .ema import EMA

class guidedMAE(nn.Module):

    def __init__(self, online:nn.Module, cfg) -> None:
        super(guidedMAE).__init__()
        self.online = online

        # EMA vs individual block ?
        self.ema = EMA(self.online) # next; exclude decoder, only encoder

        self.regression_head = self._build_regression_head()

        self.cfg = cfg
        self.ema_decay = self.cfg.model.ema_decay
        self.ema_end_decay = self.cfg.model.ema_end_decay
        self.ema_anneal_end_step = self.cfg.model.ema_anneal_end_step

    def _build_regression_head(self):