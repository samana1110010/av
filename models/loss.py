import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedInfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video_embeddings, audio_embeddings, labels):
        # Normalize
        vid_emb = F.normalize(video_embeddings, dim=1)
        aud_emb = F.normalize(audio_embeddings, dim=1)
        
        # Similarity matrix (B, B)
        sim = torch.matmul(vid_emb, aud_emb.T) / self.temperature
        
        # Mask of positives (B, B)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Compute log-softmax over rows (video -> audio)
        log_prob_v_a = F.log_softmax(sim, dim=1)
        
        # Compute log-softmax over columns (audio -> video)
        log_prob_a_v = F.log_softmax(sim.T, dim=1)
        
        # Average log-prob of all positive pairs
        # We sum the log_probs masked by positives, and divide by number of positives per row/col
        positives_per_row = mask.sum(dim=1)
        positives_per_col = mask.sum(dim=0)
        
        loss_v_a = -(mask * log_prob_v_a).sum(dim=1) / positives_per_row
        loss_a_v = -(mask * log_prob_a_v).sum(dim=0) / positives_per_col
        
        return (loss_v_a.mean() + loss_a_v.mean()) / 2
