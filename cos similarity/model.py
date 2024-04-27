import torch
import torchvision
from torch import nn

from lightly.models import utils
if False:
    from .NecessaryCode import masked_autoencoder
else:
    import masked_autoencoder as masked_autoencoder
from lightly.transforms.mae_transform import MAETransform

class MAE(nn.Module):
    def __init__(self, vit, seq_length, out_dim):
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.5
        self.patch_size = vit.patch_size
        self.sequence_length = seq_length+1
        self.out_dim = out_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit, out_dim)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=seq_length+1,
            num_layers=4,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=out_dim,
            dropout=0,
            attention_dropout=0,
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        images = images.reshape((batch_size, self.sequence_length-1, self.out_dim))
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(images, idx_mask - 1)
        # unpatchify the x_pred into same-shape tensor to imshow the prediction
        visual_prediction = utils.set_at_index(images, idx_mask - 1, x_pred).reshape((batch_size, self.sequence_length-1, self.out_dim//2, 2))
        visual_target = images.reshape((batch_size, self.sequence_length-1, self.out_dim//2, 2))

        
        return x_pred, target, visual_prediction, visual_target
