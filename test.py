import torch
from typing import List



def block_diag_left_matmul(blocks_A: List[torch.Tensor], B: torch.Tensor) -> torch.Tensor:
    """
    Compute Y = A @ B where A = block_diag(blocks_A).
    
    - blocks_A: [A0, A1, ..., Ap-1], each Ai of shape (di, di).
    - B: Tensor of shape (D, M), where D = sum(di).
    
    Returns:
        Y of shape (D, M) given by:
        [A0 @ B0; A1 @ B1; ...; Ap-1 @ Bp-1],
        where Bi is the slice of B corresponding to rows of Ai.
    """
    # 1. Determine split sizes along rows of B
    split_sizes = [Ai.shape[0] for Ai in blocks_A]
    # 2. Split B into matching row slices
    B_slices = torch.split(B, split_sizes, dim=0)
    # 3. Multiply each block independently
    Y_slices = [Ai @ Bi for Ai, Bi in zip(blocks_A, B_slices)]
    # 4. Concatenate back into full result
    return torch.cat(Y_slices, dim=0)

def block_diag_right_matmul(A: torch.Tensor, blocks_B: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute Y = A @ B where B = block_diag(blocks_B).
    
    - A: Tensor of shape (N, D), where D = sum(dj).
    - blocks_B: [B0, B1, ..., Bp-1], each Bj of shape (dj, dj).
    
    Returns:
        Y of shape (N, D) given by:
        [A0 @ B0, A1 @ B1, ..., Ap-1 @ Bp-1],
        where Aj is the slice of A corresponding to columns of Bj.
    """
    # 1. Determine split sizes along columns of A
    split_sizes = [Bj.shape[0] for Bj in blocks_B]
    # 2. Split A into matching column slices
    A_slices = torch.split(A, split_sizes, dim=1)
    # 3. Multiply each slice with its block
    Y_slices = [Ai @ Bj for Ai, Bj in zip(A_slices, blocks_B)]
    # 4. Concatenate back into full result
    return torch.cat(Y_slices, dim=1)


def block_diag_left_matmul_compact(A_compact, B, p):
    """
    Y = A @ B,  A = block_diag(A0,...,A_{p-1}),
    A_compact: (D, d) with D = p*d
    B:         (D, M)
    """
    D, d = A_compact.shape
    # 1) 把 A_compact 视为 (p, d, d)
    A3 = A_compact.view(p, d, d)
    # 2) 把 B 视为 (p, d, M)
    B3 = B.view(p, d, -1)
    # 3) 批量乘法
    Y3 = torch.matmul(A3, B3)        # (p, d, M)
    # 4) 重塑回 (D, M)
    return Y3.reshape(D, B.shape[1])

def block_diag_right_matmul_compact(A, B_compact, p):
    """
    Y = A @ B,  B = block_diag(B0,...,B_{p-1}),
    A:         (N, D) with D = p*d
    B_compact: (d, D)
    """
    N, D = A.shape
    d = D // p
    # 1) 把 B_compact 视为 (p, d, d)
    B3 = B_compact.view(d, p, d).permute(1, 0, 2)  # (p, d, d)
    # 2) 把 A 视为 (N, p, d)
    A3 = A.view(N, p, d)
    # 3) 分块相乘并水平拼回
    Y_slices = [A3[:, i, :] @ B3[i] for i in range(p)]  # 每块 (N, d)
    return torch.cat(Y_slices, dim=1)  # (N, D)


if __name__ == "__main__":
    # 测试 block_diag_left_matmul
    A = torch.randn(6, 6)
    blocks_A = [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)]
    B = torch.randn(6, 4)
    result = block_diag_left_matmul(blocks_A, B)
    print("block_diag_left_matmul result shape:", result.shape)  # 应该是 (6, 4)

    # 测试 block_diag_right_matmul
    A = torch.randn(4, 6)
    blocks_B = [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)]
    result = block_diag_right_matmul(A, blocks_B)
    print("block_diag_right_matmul result shape:", result.shape)  # 应该是 (4, 6)

    # 测试 block_diag_left_matmul_compact
    A_compact = torch.randn(12, 3)
    B = torch.randn(12, 4)
    p = 4
    result = block_diag_left_matmul_compact(A_compact, B, p)
    print("block_diag_left_matmul_compact result shape:", result.shape)  # 应该是 (12, 4)

    # 测试 block_diag_right_matmul_compact
    A = torch.randn(4, 12)
    B_compact = torch.randn(3, 12)
    p = 4
    result = block_diag_right_matmul_compact(A, B_compact, p)
    print("block_diag_right_matmul_compact result shape:", result.shape)  # 应该是 (4, 12)