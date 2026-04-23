import torch


def rerank_candidates(model, features, device):
    model.eval()

    X = []
    track_ids = []

    for f in features:
        X.append([
            f["stage1_score"],
            f["transformer_score"],
            f["popularity"],
        ])
        track_ids.append(f["track_id"])

    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        scores = model(X).squeeze(-1).cpu().numpy()

    ranked = sorted(
        zip(track_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [{"track_id": t, "score": float(s)} for t, s in ranked]
