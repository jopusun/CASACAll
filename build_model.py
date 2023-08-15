import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, input):
        encoder_outputs = self.encoder(input)
        output = self.decoder(encoder_outputs)
        return output

def build_model(encoder, decoder, device):
  # 建構模型
  model = Seq2Seq(encoder, decoder)
  #print(model)

  model = model.to(device)
  return model

