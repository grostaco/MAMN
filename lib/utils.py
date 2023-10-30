import torch


def wdmc(h_cp: torch.Tensor, a_starts: tuple[int, ...], a_ends: tuple[int, ...], window_size: int):
    """_summary_

    Args:
        h_cp (torch.Tensor): _description_
        a_range (tuple[tuple[int, int], ...]): _description_
        window_size (int): _description_

    Returns:
        _type_: _description_
    """
    tensors = []

    for a_s, a_e in zip(a_starts, a_ends):
        r = []
        n = h_cp.size(1)

        if a_s - window_size/2 > 0:
            d_fwd = torch.arange(a_s - window_size/2, 0, -1)
            d_fwd = (1 - d_fwd/n)

            r.append(d_fwd)

        r.append(torch.ones(min(a_s, window_size//2) +
                            min(n - a_e - 1, window_size//2) + a_e - a_s + 1))

        if a_e + window_size/2 + 1 < n:
            d_bwd = torch.arange(1, n - a_e - window_size/2)
            d_bwd = (1 - d_bwd/n)
            r.append(d_bwd)

        tensors.append(torch.cat(r).view(-1, 1).repeat(1, 1, h_cp.size(-1)))

    return torch.cat(tensors).to(h_cp.device) * h_cp
