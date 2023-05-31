import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class FinalCustomActivationFunction(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=3):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.available_activations = [nn.Mish(), nn.ReLU(), nn.GELU()]

        # Layer normalization
        self.normlist = nn.ModuleList([nn.LayerNorm(d_model) for i in range(num_layers)])

        # Q, K, V transformations
        self.q_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.k_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.v_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])

        self.mha_layers = nn.ModuleList([MultiheadAttention(embed_dim=d_model, num_heads=num_heads) for _ in range(num_layers)])

        self.activation_layers = nn.ModuleList([self.available_activations[i % len(self.available_activations)] for i in range(num_layers)])

        # Learnable weights for each layer
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers, requires_grad=True)

    def forward(self, x):
        outputs = []
        
        # Apply Q, K, V transformations and multihead attention layer-wise
        for q_layer, k_layer, v_layer, mha_layer, act_layer, norm_layer in zip(self.q_layers, self.k_layers, self.v_layers, self.mha_layers, self.activation_layers, self.normlist):
            q = q_layer(x)
            k = k_layer(x)
            v = v_layer(x)
            
            # Pass through multi-head attention and activation
            output, _ = mha_layer(q, k, v)
            output = act_layer(output)
            output = norm_layer(output)
            outputs.append(output)
        
        # Combine outputs from all layers
        x = torch.stack(outputs, dim=0)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (x * weights).sum(dim=0)
        
        return x
