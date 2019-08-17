import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_kernel(x1: np.array, x2: np.array, param: np.array) -> float:
    """ ガウスカーネルを計算する。
        k(x1, x2) = exp(-|x - x2|^2/param[0])
    
    Args:
        x1 (np.array)   : 入力値1
        x2 (np.array)   : 入力値2
        param (np.array): ガウスカーネルのパラメータ
    
    Returns:
        float: ガウスカーネルの値
    """

    return np.exp(-np.linalg.norm(x1 - x2)**2 / param[0])

def exp_kernel(x1: np.array, x2: np.array, param: np.array) -> float:
    """ 指数カーネルを計算する。
        k(x1, x2) = exp(-|x - x2|/param[0])
    
    Args:
        x1 (np.array)   : 入力値1
        x2 (np.array)   : 入力値2
        param (np.array): 指数カーネルのパラメータ
    
    Returns:
        float: 指数カーネルの値
    """

    return np.exp(-np.linalg.norm(x1 - x2) / param[0])

def periodic_kernel(x1: np.array, x2: np.array, param: np.array) -> float:
    """ 周期カーネルを計算する。
        k(x1, x2) = exp(param[0]*cos(|x - x2|/param[0]))
    
    Args:
        x1 (np.array)   : 入力値1
        x2 (np.array)   : 入力値2
        param (np.array): 周期カーネルのパラメータ
    
    Returns:
        float: 周期カーネルの値
    """

    return np.exp(param[0]*np.cos(np.linalg.norm(x1 - x2) / param[1]))

def calc_kernel_matrix(x: np.array, kernel: 'function', kparam: np.array) -> np.array:
    """ カーネル行列を計算して返す。
        カーネル行列Kの(i, j)成分はkernel(x[i], x[j])
    
    Args:
        x (np.array)      : 入力値
        kernel (function) : カーネル関数
        kparam (np.array) : カーネル関数のパラメータ
    
    Returns:
        np.array: カーネル行列
    """

    # カーネル行列の初期化
    mkernel = np.zeros((x.shape[0], x.shape[0]))

    for i_row in range(x.shape[0]):
        for i_col in range(i_row, x.shape[0]):
            mkernel[i_row, i_col] = kernel(x[i_row], x[i_col], kparam)
            mkernel[i_col, i_row] = mkernel[i_row, i_col]
    
    return mkernel


def gp_sampling(x: np.array, lm: float, mean: np.array, kernel: 'function', 
                kparam: np.array) -> np.array:
    """ ガウス過程からサンプルした結果を返す。
    
    Args:
        x (np.array)      : 入力値
        lm (float)        : 共分散行列の定数λ
        mean (np.array)   : ガウス過程の平均ベクトル
        kernel (function) : カーネル関数
        kparam (np.array) : カーネル関数のパラメータ
    
    Returns:
        np.array: サンプルした値
    """

    # カーネル行列の計算
    mkernel = calc_kernel_matrix(x, kernel, kparam)

    # 共分散行列の計算
    K = lm**2 * np.dot(mkernel, mkernel.T)
    
    # サンプリング
    return np.random.multivariate_normal(mean, K)

if __name__ == "__main__":

    # 乱数シードの固定
    np.random.seed(2)

    fig = plt.figure(figsize=(18, 5))

    # ガウスカーネル-1次元入力
    x = np.linspace(0, 10, 200, endpoint=False)
    samples = np.array([gp_sampling(x, 1.0, np.zeros(len(x)), gaussian_kernel, np.array([0.5])) for i in range(3)])
    
    ax = fig.add_subplot(1, 3, 1)
    for sample in samples:
        ax.plot(x, sample)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gaussian Kernel')

    # 指数カーネル-1次元入力
    x = np.linspace(0, 10, 200, endpoint=False)
    samples = np.array([gp_sampling(x, 1.0, np.zeros(len(x)), exp_kernel, np.array([0.5])) for i in range(3)])
    
    ax = fig.add_subplot(1, 3, 2)
    for sample in samples:
        ax.plot(x, sample)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Exponential Kernel')
    
    # 周期カーネル-1次元入力
    x = np.linspace(0, 10, 200, endpoint=False)
    samples = np.array([gp_sampling(x, 0.01, np.zeros(len(x)), periodic_kernel, np.array([1.0, 0.5])) for i in range(3)])
    
    ax = fig.add_subplot(1, 3, 3)
    for sample in samples:
        ax.plot(x, sample)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Periodic Kernel')
    
    fig.tight_layout()
    fig.savefig('result1.png')

    # ガウスカーネル-2次元入力
    x = np.linspace(0, 10, 100, endpoint=False)
    y = np.linspace(0, 10, 100, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    shape_input = xx.shape
    xy = np.c_[xx.reshape(-1), yy.reshape(-1)] # 2次元ベクトルの入力
    kparam = np.array([.50])
    z = gp_sampling(xy, 1.0, np.zeros(len(xy)), gaussian_kernel, kparam).reshape(shape_input)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, z, cmap='CMRmap', linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Gaussian Kernel')
    fig.tight_layout()
    fig.savefig('result2.png')

    