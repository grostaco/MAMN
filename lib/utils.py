import torch


def wdmc(h_cp: torch.Tensor, a_range: tuple[tuple[int, int], ...], window_size: int):
    tensors = []

    for a_s, a_e in a_range:
        n = h_cp.size(1)

        d = torch.arange(window_size / 2, max(a_s, n - a_e) - 1) + 1

        d_weighted = 1 - (d - (window_size/2))/n

        r_s = int(a_s - window_size / 2)
        r_e = int(n - a_e - window_size / 2 - 1)

        tensors.append(torch.cat((d_weighted[:r_s].flip(-1), torch.ones(
            window_size + a_e - a_s + 1), d_weighted[-r_e:])).view(-1, 1).repeat(1, 1, h_cp.size(-1)))
    return torch.cat(tensors) * h_cp
