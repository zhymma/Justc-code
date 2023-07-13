import numpy as np
def compute_iterations(P, alpha, beta, threshold):
    iterations = 0
    while P < threshold:
        P = P * alpha + P * (1-alpha) * beta + (1 - P) * alpha * beta
        iterations += 1
    return iterations

P = 0.88
threshold = 0.95
max_iterations = 10

# 初始的最小值为1，因为α和β的最大可能值为1
min_alpha = 1
min_beta = 1

step = 0.5  # 调整步长以改变搜索的精度和速度

# 对每一个可能的α和β值进行搜索
for alpha in np.arange(0.6, 1+step, step):
    for beta in np.arange(0.6, 1+step, step):
        iterations = compute_iterations(P, alpha, beta, threshold)
        # 如果在指定的迭代次数内达到阈值，并且α和β的值比当前找到的最小值还要小，那么就更新最小值
        if iterations <= max_iterations and alpha * beta < min_alpha * min_beta:
            min_alpha = alpha
            min_beta = beta

print("最小的α值为：", min_alpha)
print("最小的β值为：", min_beta)
