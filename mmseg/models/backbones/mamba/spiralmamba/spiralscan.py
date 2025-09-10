import torch
import torch.nn as nn
import numpy as np
import time

class SpiralScan(nn.Module):
    """
    (V3: With Static Index Cache)
    This version adds a class-level cache to store pre-computed spiral indices.
    When multiple SpiralScan modules with the same (H, W) dimensions are
    created, the expensive index calculation is performed only once, and the
    resulting tensor is shared among all instances.

    This significantly speeds up model initialization and reduces memory usage
    in models with many such layers.
    """
    # 1. 定义一个类级别的静态字典作为缓存/注册器
    _INDEX_CACHE = {}

    def __init__(self, H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        
        # 定义当前实例的尺寸key
        key = (H, W)

        # 2. 检查缓存中是否已有该尺寸的索引
        if key not in SpiralScan._INDEX_CACHE:
            # 如果没有，则计算索引并存入缓存
            print(f"Cache miss for key {key}. Building spiral index...")
            index = self.build_spiral_index_optimized(H, W)
            SpiralScan._INDEX_CACHE[key] = index
        else:
            # 如果有，则直接从缓存读取
            print(f"Cache hit for key {key}. Reusing existing index.")
            index = SpiralScan._INDEX_CACHE[key]
        
        # 3. 将索引注册为当前模块的缓冲区
        # 注意：这里传递的是张量的引用，所以内存是共享的
        self.register_buffer("spiral_index", index)

    def build_spiral_index_optimized(self, H: int, W: int) -> torch.LongTensor:
        """
        (Optimized Version from V2)
        Builds the 1D indices for a spiral path on an H x W grid.
        """
        if not (H > 0 and W > 0):
            return torch.empty(0, dtype=torch.long)

        coords = []
        r, c = H // 2, W // 2
        dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # R, U, L, D
        dir_idx, steps_per_leg, steps_walked, turns = 0, 1, 0, 0

        # 使用 while 循环直到收集够 H * W 个点
        while len(coords) < H * W:
            # **核心修正**: 只有当 (r, c) 在边界内时，才将其加入路径
            # 这个检查确保了即使是第一个中心点，如果H或W为0，也不会被添加
            if 0 <= r < H and 0 <= c < W:
                coords.append((r, c))

            # 和之前一样的移动和转向逻辑
            dr, dc = dirs[dir_idx]
            r, c = r + dr, c + dc
            steps_walked += 1
            if steps_walked == steps_per_leg:
                steps_walked = 0
                dir_idx = (dir_idx + 1) % 4
                turns += 1
                if turns % 2 == 0:
                    steps_per_leg += 1
        
        # 将坐标列表转换为1D索引
        coords_torch = torch.tensor(coords, dtype=torch.long)
        indices = coords_torch[:, 0] * W + coords_torch[:, 1]
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            B, C, _, _ = x.shape
            x_flat = x.contiguous().view(B, C, -1)
        else:
            B, C, L = x.shape
            x_flat = x.contiguous().view(B, C, -1)
        if self.H * self.W != x_flat.shape[-1]:
            try:
                indices = self.spiral_index_new[None, None, :].expand(B, C, -1)
                scanned_seq = torch.gather(x_flat, 2, indices)
            except:
                index_new = self.build_spiral_index_optimized(int(x_flat.shape[-1] ** 0.5), int(x_flat.shape[-1] ** 0.5)).to(x.device)
                self.register_buffer("spiral_index_new", index_new)
                indices = self.spiral_index_new[None, None, :].expand(B, C, -1)
                scanned_seq = torch.gather(x_flat, 2, indices)
            return scanned_seq

        indices = self.spiral_index[None, None, :].expand(B, C, -1)
        scanned_seq = torch.gather(x_flat, 2, indices)
        return scanned_seq

    def recover(self, y: torch.Tensor) -> torch.Tensor:
        B, C, L = y.shape
        if self.H * self.W != L:
            indices = self.spiral_index_new[None, None, :].expand(B, C, -1)
            x_flat = torch.zeros(B, C, L, device=y.device, dtype=y.dtype)
            x_flat = x_flat.scatter(2, indices, y)
            return x_flat.view(B, C, int(L ** 0.5), int(L ** 0.5))

        indices = self.spiral_index[None, None, :].expand(B, C, -1)
        x_flat = torch.zeros(B, C, self.H * self.W, device=y.device, dtype=y.dtype)
        x_flat = x_flat.scatter(2, indices, y)
        return x_flat.view(B, C, self.H, self.W)


def verify_module(scanner_instance: SpiralScan, B: int, C: int, H: int, W: int):
    """
    一个用于验证 SpiralScan 模块正确性的函数。
    它会检查 forward -> recover 的过程是否能完美重构原始输入。
    """
    print(f"--- Verifying module for size H={H}, W={W} ---")
    
    # 1. 创建一个可识别的、有序的输入张量
    # 使用 arange 可以确保每个元素都是唯一的，方便排查错误
    try:
        device = next(scanner_instance.parameters()).device
    except StopIteration: # 如果模块没有参数，则检查缓冲区
        try:
            device = next(scanner_instance.buffers()).device
        except StopIteration:
            device = 'cpu' # 默认为cpu

    input_tensor = torch.arange(B * C * H * W, dtype=torch.float32, device=device).view(B, C, H, W)
    print(f"Input tensor created on device: '{device}' with shape {input_tensor.shape}")

    # 2. 执行前向传播 (scan)
    scanned_output = scanner_instance(input_tensor)
    
    # 3. 执行恢复操作 (recover)
    recovered_flat = scanner_instance.recover(scanned_output)
    recovered_tensor = recovered_flat.view(B, C, H, W)
    
    # 4. 比较原始张量和恢复后的张量
    is_correct = torch.allclose(input_tensor, recovered_tensor)
    
    if is_correct:
        print("✅ Verification successful: Input and recovered tensors match perfectly.")
    else:
        print("❌ Verification FAILED: Input and recovered tensors DO NOT match.")
    
    # 检查scanner实例的索引是否在正确的设备上
    print(f"Scanner index is on device: '{scanner_instance.spiral_index.device}'")
    print("-" * 40)


# --- 使用示例来验证缓存效果 ---
if __name__ == '__main__':
    print("--- Verifying Static Cache ---")
    
    # 创建第一个 12x12 的实例
    print("\nCreating first 12x12 instance...")
    scanner1 = SpiralScan(12, 12)

    # 创建第二个 12x12 的实例
    print("\nCreating second 12x12 instance...")
    scanner2 = SpiralScan(12, 12)

    # 创建一个不同尺寸的实例
    print("\nCreating a 32x32 instance...")
    scanner3 = SpiralScan(32, 32)
    
    # 创建第三个 12x12 的实例
    print("\nCreating third 12x12 instance...")
    scanner4 = SpiralScan(12, 12)

    print("\n--- Cache Verification ---")
    # 验证 scanner1 和 scanner2 的 spiral_index 是否是同一个对象
    # data_ptr() 返回张量第一个元素的内存地址
    is_shared = scanner1.spiral_index.data_ptr() == scanner2.spiral_index.data_ptr()
    print(f"Are indices for scanner1 and scanner2 shared in memory? -> {is_shared}")
    
    is_different = scanner1.spiral_index.data_ptr() == scanner3.spiral_index.data_ptr()
    print(f"Are indices for scanner1 and scanner3 different? -> {not is_different}")


    print("-" * 40)
    B, C = 2, 4
    H1, W1 = 12, 12
    H2, W2 = 32, 32
    # 2. 使用新增的函数来验证每个实例
    verify_module(scanner1, B, C, H1, W1)
    verify_module(scanner2, B, C, H1, W1)
    verify_module(scanner3, B, C, H2, W2)

    print("-" * 40)
    B, C, H, W =2, 4, 12, 12

    scanner = SpiralScan(H, W)
    # a. 创建一个原始的二维数据
    original_data = torch.arange(B * C * H * W).reshape(B, C, H, W)
    print("原始 5x5 数据:")
    print(original_data[0, 0, :, :])
    
    # b. 模拟扫描过程，得到一维序列
    # original_data = original_data.flip(-1)
    original_data = original_data.transpose(2, 3).flip(-1)
    flattened_sequence = scanner(original_data)
    print("\n螺旋扫描后的一维序列:")
    print(flattened_sequence[0, 0, :])

    # c. 从一维序列中恢复二维数据
    recovered_data = scanner.recover(flattened_sequence)
    print("\n从序列中恢复的 5x5 数据:")
    print(recovered_data[0, 0, :, :])
    print(recovered_data.shape)
    # d. 验证恢复是否正确
    print("\n恢复是否成功:", torch.allclose(original_data, recovered_data))