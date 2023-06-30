import torch


def top_p_sample(logits, top_p):
    # Top-P Sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # Set first token to false to always keep it
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    logits[indices_to_remove] = float("-inf")

    # sample from the output distribution
    sample = torch.multinomial(
        torch.softmax(logits[:, -1], dim=-1),
        num_samples=1,
    )

    return sample
