import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class SimpleInfillingModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=512)