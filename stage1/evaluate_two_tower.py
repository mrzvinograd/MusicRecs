import torch
import numpy as np


def recall_at_k(model, dataset, embeddings, K=10, num_samples=1000):
    device = next(model.parameters()).device

    model.eval()

    hits = 0
    total = 0

    # 🔥 ускорение: берём subset треков
    idx = np.random.choice(len(embeddings), 5000, replace=False)
    all_tracks = torch.tensor(embeddings[idx], dtype=torch.float32).to(device)

    with torch.no_grad():

        for i, (playlist, target) in enumerate(dataset):

            if i >= num_samples:
                break

            playlist = playlist.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            # --- encode playlist ---
            p_vec = model.playlist_encoder(playlist)

            # --- similarity ---
            scores = torch.matmul(p_vec, all_tracks.T)

            topk = torch.topk(scores, K).indices[0]

            # --- правильный трек ---
            target_score = torch.matmul(target, all_tracks.T)
            target_idx = torch.argmax(target_score)

            if target_idx in topk:
                hits += 1

            total += 1

    return hits / total if total > 0 else 0
