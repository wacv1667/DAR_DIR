import torch
from torch import nn, Tensor
from typing import Dict, List, Optional
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

class AttentionFusionModule(nn.Module):
    """
    Fuse embeddings through weighted sum of the corresponding linear projections.
    Linear layer for learning the weights.
    Copied from: https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/attention.py

    Args:
        channel_to_encoder_dim: mapping of channel name to the encoding dimension
        encoding_projection_dim: common dimension to project the encodings to.
        defaults to min of the encoder dim if not set

    """
    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        encoding_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        attn_in_dim = sum(channel_to_encoder_dim.values())
        self.attention = nn.Sequential(
            nn.Linear(attn_in_dim, len(channel_to_encoder_dim)),
            nn.Softmax(-1),
        )
        if encoding_projection_dim is None:
            encoding_projection_dim = min(channel_to_encoder_dim.values())

        encoding_projection = {}
        for channel in sorted(channel_to_encoder_dim.keys()):
            encoding_projection[channel] = nn.Linear(
                channel_to_encoder_dim[channel], encoding_projection_dim
            )
        self.encoding_projection = nn.ModuleDict(encoding_projection)

    def forward(self, embeddings: Dict[str, Tensor]) -> Tensor:
        concatenated_in = torch.cat(
            [embeddings[k] for k in sorted(embeddings.keys())], dim=-1
        )
        attention_weights = self.attention(concatenated_in)
        projected_embeddings: List[Tensor] = []
        for channel, projection in self.encoding_projection.items():
            projected_embedding = projection(embeddings[channel])
            projected_embeddings.append(projected_embedding)

        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = (
                attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
            )

        fused = torch.sum(torch.stack(projected_embeddings), dim=0)
        return fused


class TwoStreamAttentionFusion(nn.Module):
    def __init__(self,
                 train_ds,
                 model_ckpt =  "MCG-NJU/videomae-base-finetuned-kinetics", 
                 num_classes=5):
        super().__init__()
        self.inside_vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=train_ds.label2id, id2label=train_ds.id2label, ignore_mismatched_sizes=True,)
        self.inside_vmae.classifier = torch.nn.Identity()
        self.outside_vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=train_ds.label2id, id2label=train_ds.id2label, ignore_mismatched_sizes=True,)
        self.outside_vmae.classifier = torch.nn.Identity()
        self.attention_fusion = AttentionFusionModule({"inside":768, "outside":768}, 768)
        self.classifier = nn.Linear(768,num_classes)
        
    def forward(self, x):
        x1 = self.inside_vmae(x["inside"].permute(0,2,1,3,4)).logits
        x2 = self.outside_vmae(x["outside"].permute(0,2,1,3,4)).logits
        x = self.attention_fusion({"inside": x1, "outside": x2})
        x = self.classifier(x)
        return x


class TwoStreamConcatFusion(nn.Module):
    def __init__(self,
                 train_ds,
                 model_ckpt =  "MCG-NJU/videomae-base-finetuned-kinetics", 
                 num_classes=5):
        super().__init__()
        self.inside_vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=train_ds.label2id, id2label=train_ds.id2label, ignore_mismatched_sizes=True,)
        self.inside_vmae.classifier = torch.nn.Identity()
        self.outside_vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=train_ds.label2id, id2label=train_ds.id2label, ignore_mismatched_sizes=True,)
        self.outside_vmae.classifier = torch.nn.Identity()
        self.classifier = nn.Linear(768*2,num_classes)
        
    def forward(self, x):
        x1 = self.inside_vmae(x["inside"].permute(0,2,1,3,4)).logits
        x2 = self.outside_vmae(x["outside"].permute(0,2,1,3,4)).logits
        x = torch.cat((x1, x2), -1)
        x = self.classifier(x)
        return x


class OneStream(nn.Module):
    def __init__(self,
                 train_ds,
                 model_ckpt =  "MCG-NJU/videomae-base-finetuned-kinetics",
                 stream_name = "inside"):
        super().__init__()
        self.vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=train_ds.label2id, id2label=train_ds.id2label, ignore_mismatched_sizes=True,)
        self.stream = stream_name
    def forward(self, x):
        x = self.vmae(x[self.stream].permute(0,2,1,3,4)).logits
        return x
