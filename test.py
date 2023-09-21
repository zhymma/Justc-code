import torch

x = torch.tensor([[0, 1, 0],
                  [2, 0, 3]])

result_tuple = torch.nonzero(x, as_tuple=True)
result_tensor = torch.nonzero(x)

# 将result_tensor转化为下标list
result_list = result_tensor.tolist()
result_list = [(i,j) for i,j in result_list] 
# 通过result_list访问x中的元素
for i in range(len(result_list)):
    y = (x[result_list[i]])
    print(y)
   #  result_list[i].append(y.item())


print("Result as tensor:")
print(result_tensor)
print(result_list)