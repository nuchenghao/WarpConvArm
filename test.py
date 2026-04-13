import torch


def _create_voxels_data(
    B: int,
    min_N: int,
    max_N: int,
    C: int,
    device: str = "cpu",
    voxel_size: float = 0.01,
):
    """Helper function to create voxels data with given parameters."""
    torch.manual_seed(0)
    Ns = torch.randint(
        min_N, max_N, (B,)
    )  # 先为 B 个样本分别随机生成点数，每个样本的点数在 min_N 到 max_N 之间。
    print(Ns)
    # 给每个样本生成 N x 3 的随机三维坐标。torch.rand 先生成 [0, 1) 的浮点数，除以 voxel_size 后放大，再转成 int，
    # 相当于把连续空间坐标量化成离散 voxel 坐标。
    coords = [(torch.rand((int(N.item()), 3)) / voxel_size).int() for N in Ns]
    # 给每个坐标点配一个长度为 C 的随机特征向量。
    features = [torch.rand((int(N.item()), C)) for N in Ns]

    print(coords)

    print(features)


_create_voxels_data(
    B=3,
    min_N=1,
    max_N=5,
    C=7,
    device="cpu",
    voxel_size=0.01,
)
