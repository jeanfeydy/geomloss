import torch
from geomloss import SamplesLoss

device = "cpu"
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



xs = [torch.randn((8, 2), dtype=torch.float, device=device), torch.randn((7, 2), dtype=torch.float, device=device), torch.randn((6, 2), dtype=torch.float, device=device),]
ys = [torch.randn((3, 2), dtype=torch.float, device=device), torch.randn((4, 2), dtype=torch.float, device=device), torch.randn((5, 2), dtype=torch.float, device=device),]

L_online = SamplesLoss(
    "sinkhorn",
    p=p,
    blur=0.5,
    potentials=potential,
    debias=False,
    backend="online",
)

distances = torch.as_tensor([L_online(x, y) for x, y in zip(xs, ys)], device=device)

ptr_x = torch.tensor([0, 8, 15, 21], device=device)
ptr_y = torch.tensor([0, 3, 7, 12], device=device)
xs_batched = torch.cat(xs, dim=0)
ys_batched = torch.cat(ys, dim=0)

distances_batched = L_online(xs_batched, ys_batched, ptr_x=ptr_x, ptr_y=ptr_y)

# not sure where the small difference comes from. Is it just numerical error?
print( torch.norm(distances - distances_batched) )
