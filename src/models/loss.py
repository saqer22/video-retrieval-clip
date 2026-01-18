import torch
import torch.nn.functional as F


def info_nce(similarity: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    labels = torch.arange(similarity.size(0), device=similarity.device)
    logits = similarity / temperature
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2
