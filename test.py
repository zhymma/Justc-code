import torch

x = torch.tensor([[0, 1, 0],
                  [2, 0, 3]])

result_tuple = torch.nonzero(x, as_tuple=True)
result_tensor = torch.nonzero(x)



print("Result as tensor:")
print(result_tensor)