import torch
from geomloss import SamplesLoss

device = "cuda"
tdtype = torch.float


x = torch.randn((3, 8, 2), dtype=torch.float, device=device)
y = torch.randn((3, 15, 2), dtype=torch.float, device=device)


P = [1, 2]
Debias = [True, False]
potential = False

for p in P:
    for debias in Debias:
        L_tensorized = SamplesLoss(
            "sinkhorn",
            p=p,
            blur=0.5,
            potentials=potential,
            debias=debias,
            backend="tensorized",
        )
        # a, b= L_tensorized(x, y)
        A = L_tensorized(x, y)

        L_online = SamplesLoss(
            "sinkhorn",
            p=p,
            blur=0.5,
            potentials=potential,
            debias=debias,
            backend="online",
        )
        # c, d= L_online(x, y)
        B = L_tensorized(x, y)

        # print(a, b)
        # print(c, d)
        print(torch.norm(A - B))
