import numpy as np
import matplotlib.pyplot as plt
from time import time

def solve_laplace(nx, ny, omega=1.0, tol=1e-6, max_iter=10000, verbose=False):
    """
    使用 SOR 方法求解二维拉普拉斯方程稳态温度场
    nx, ny: 网格点数量 (不包括边界)
    omega: 松弛因子
    tol: 收敛阈值
    max_iter: 最大迭代次数
    返回: 温度场, 迭代次数, 运行时间
    """
    # 板尺寸
    Lx, Ly = 0.15, 0.12  # 单位 m
    dx = Lx / (nx + 1)
    dy = Ly / (ny + 1)

    # 初始化温度场，包括边界
    T = np.zeros((ny+2, nx+2))
    # 边界条件
    T[-1, :] = 100.0
    T[:, 0] = 20.0
    T[:, -1] = 20.0
    T[0, :] = 20.0

    # 迭代求解
    count = 0
    start = time()
    dx2, dy2 = dx*dx, dy*dy
    denom = 2*(dx2 + dy2)
    
    while count < max_iter:
        max_diff = 0.0
        for j in range(1, ny+1):
            for i in range(1, nx+1):
                T_old = T[j, i]
                # 离散拉普拉斯算子 (中央差分)
                T_new = ((dy2*(T[j, i-1] + T[j, i+1]) + dx2*(T[j-1, i] + T[j+1, i])) / denom)
                # SOR 更新
                T[j, i] = T_old + omega*(T_new - T_old)
                diff = abs(T[j, i] - T_old)
                if diff > max_diff:
                    max_diff = diff
        count += 1
        if verbose and count % 500 == 0:
            print(f"迭代 {count}, 最大变化 {max_diff:.2e}")
        if max_diff < tol:
            break
    elapsed = time() - start
    return T, count, elapsed

# (1) 计算并绘制等温线
nx, ny = 50, 40
T, iters, t_elapsed = solve_laplace(nx, ny, omega=1.0, tol=1e-6, verbose=True)
print(f"网格: {nx}x{ny}, Gauss-Seidel 迭代次数: {iters}, 用时: {t_elapsed:.2f}s")

X = np.linspace(0, 0.15, nx+2)
Y = np.linspace(0, 0.12, ny+2)
XX, YY = np.meshgrid(X, Y)

plt.figure(figsize=(8,5))
cs = plt.contour(XX, YY, T, levels=10)
plt.clabel(cs, inline=1, fontsize=10)
plt.title('板内稳定温度场等温线 (GS)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# (2) 不同松弛因子比较收敛速度
omegas = np.linspace(1.0, 1.9, 10)
iters_list = []
for w in omegas:
    _, it, _ = solve_laplace(nx, ny, omega=w, tol=1e-6)
    iters_list.append(it)
    print(f"omega={w:.2f}, 迭代次数={it}")

plt.figure()
plt.plot(omegas, iters_list, 'o-')
plt.title('不同松弛因子下的迭代次数')
plt.xlabel('ω')
plt.ylabel('迭代次数')
plt.grid(True)
plt.show()

# (3) 不同网格尺度下最佳松弛因子变化
grid_sizes = [(30,24), (50,40), (70,56), (100,80)]
best_omegas = []
for nx_g, ny_g in grid_sizes:
    best_it = np.inf
    best_w = None
    for w in omegas:
        _, it, _ = solve_laplace(nx_g, ny_g, omega=w, tol=1e-6)
        if it < best_it:
            best_it, best_w = it, w
    best_omegas.append(best_w)
    print(f"网格 {nx_g}x{ny_g}, 最佳 omega = {best_w:.2f} (迭代 {best_it})")

sizes = [nx*ny for nx,ny in grid_sizes]
plt.figure()
plt.plot(sizes, best_omegas, 's-')
plt.title('不同网格尺度下的最佳松弛因子')
plt.xlabel('网格点总数 (nx×ny)')
plt.ylabel('最佳 ω')
plt.grid(True)
plt.show()
