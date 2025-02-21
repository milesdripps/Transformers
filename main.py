# Transformer

import torch
import torch.nn as nn

#Transformer NN
d_model = 512
number_of_heads = 8
num_encoding_layers = 6
num_decoding_layers = 6
dimension_feedforward = 2048
dropout = 1
activation = "gelu"
#
#
#
#
#

#Encoding layer
encoding_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead = number_of_heads)
transformer_encoder = nn.TransformerEncoder(encoder_layer=encoding_layer, num_layers=6)

src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
print(out)
#Transformer
transformer = nn.Transformer(
    d_model=d_model,
    nhead = number_of_heads,
    num_encoder_layers=num_encoding_layers,
    num_decoder_layers=num_decoding_layers,
    dim_feedforward=dimension_feedforward,
    dropout=dropout,
    activation = activation
)


