import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def diag_scan_kernel(
    x_ptr, y_ptr, H, W, C,
    stride_c, stride_h, stride_w,
    out_stride_c, out_stride_hw,
    BLOCK_C: tl.constexpr, d: tl.constexpr, mode: tl.constexpr
):
    c_idx = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c_idx < C
    out_offset = 0
    for i in range(H):
        j = d - i if mode == 0 else i - (H - 1 - d)
        # if 0 <= j < W:
        if (j >= 0) and (j < W):
            x_offset = i * stride_h + j * stride_w + c_idx * stride_c
            y_offset = out_offset * out_stride_hw + c_idx * out_stride_c
            val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            tl.store(y_ptr + y_offset, val, mask=mask)
            out_offset += 1


def build_diag_index(H, W, mode='rd'):
    index_map = []
    for d in range(H + W - 1):
        for i in range(H):
            j = d - i if mode == 'rd' else i - (H - 1 - d)
            if 0 <= j < W:
                index_map.append(i * W + j)
    return torch.tensor(index_map, dtype=torch.long)


class DiagScanFunction(Function):
    @staticmethod
    def forward(ctx, x, rd_index_map, ld_index_map):
        B, C, H, W = x.shape
        BLOCK_C = 64
        x = x.contiguous()

        y_rd = torch.empty((B, C, H * W), device=x.device, dtype=x.dtype)
        y_ld = torch.empty((B, C, H * W), device=x.device, dtype=x.dtype)

        for b in range(B):
            x_b = x[b]
            stride_c, stride_h, stride_w = x_b.stride()
            out_stride_c, out_stride_hw = y_rd[b].stride()

            write_pos = 0
            for d in range(H + W - 1):
                num_warps = (C + BLOCK_C - 1) // BLOCK_C
                diag_scan_kernel[num_warps,](
                    x_b, y_rd[b][:, write_pos:], H, W, C,
                    stride_c, stride_h, stride_w,
                    out_stride_c, out_stride_hw,
                    BLOCK_C=BLOCK_C, d=d, mode=0
                )
                write_pos += min(d + 1, H, W, H + W - 1 - d)

            write_pos = 0
            for d in range(H + W - 1):
                num_warps = (C + BLOCK_C - 1) // BLOCK_C
                diag_scan_kernel[num_warps,](
                    x_b, y_ld[b][:, write_pos:], H, W, C,
                    stride_c, stride_h, stride_w,
                    out_stride_c, out_stride_hw,
                    BLOCK_C=BLOCK_C, d=d, mode=1
                )
                write_pos += min(d + 1, H, W, H + W - 1 - d)

        ctx.save_for_backward(rd_index_map.to(x.device), ld_index_map.to(x.device))
        ctx.input_shape = (B, C, H, W)
        return y_rd, y_ld

    @staticmethod
    def backward(ctx, grad_rd, grad_ld):
        rd_index_map, ld_index_map = ctx.saved_tensors
        B, C, H, W = ctx.input_shape
        grad_input = torch.zeros((B, C, H * W), device=grad_rd.device, dtype=grad_rd.dtype)
        grad_input.index_add_(2, rd_index_map, grad_rd)
        grad_input.index_add_(2, ld_index_map, grad_ld)
        return grad_input.view(B, C, H, W), None, None


class DiagScanModule(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.register_buffer("rd_index_map", build_diag_index(H, W, 'rd'))
        self.register_buffer("ld_index_map", build_diag_index(H, W, 'ld'))

    def forward(self, x):
        return DiagScanFunction.apply(x, self.rd_index_map, self.ld_index_map)

    def recover(self, y, mode='rd'):
        index_map = self.rd_index_map if mode == 'rd' else self.ld_index_map
        B, C, L = y.shape
        x = torch.zeros((B, C, self.H * self.W), dtype=y.dtype, device=y.device)
        x.index_copy_(2, index_map.to(y.device), y)
        return x#.view(B, C, self.H, self.W)

class FastDiagScan(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.register_buffer("rd_index", self.build_diag_index(H, W, mode='rd'))  # [L]
        self.register_buffer("ld_index", self.build_diag_index(H, W, mode='ld'))  # [L]

    def build_diag_index(self, H, W, mode='rd'):
        index_map = []
        for d in range(H + W - 1):
            for i in range(H):
                j = d - i if mode == 'rd' else i - (H - 1 - d)
                if 0 <= j < W:
                    index_map.append(i * W + j)
        return torch.tensor(index_map, dtype=torch.long)

    def forward(self, x):  # [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        rd = torch.gather(x_flat, 2, self.rd_index[None, None, :].expand(B, C, -1))
        ld = torch.gather(x_flat, 2, self.ld_index[None, None, :].expand(B, C, -1))
        return rd, ld

    def recover(self, y, mode='rd'):
        B, C, L = y.shape
        index_map = self.rd_index if mode == 'rd' else self.ld_index
        x = torch.zeros(B, C, self.H * self.W, device=y.device, dtype=y.dtype)
        return x.scatter(2, index_map[None, None, :].expand(B, C, -1), y)#.view(B, C, self.H, self.W)


class FastDirectionalScan(nn.Module):
    """Gather/scatter indices for eight complementary 2D scan directions.

    The forward set is ``(h_lr, diag_ur, v_tb, diag_dr)`` and the reverse
    set is its exact sequence-wise inverse ``(h_rl, diag_dl, v_bt, diag_ul)``.
    Every index visits each spatial position exactly once.
    """

    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W

        h_lr = self._horizontal_index(H, W)
        v_tb = self._vertical_index(H, W)
        diag_ur = self._diag_up_right_index(H, W)
        diag_dr = self._diag_down_right_index(H, W)

        indices = dict(
            h_lr=h_lr,
            h_rl=torch.flip(h_lr, dims=[0]),
            v_tb=v_tb,
            v_bt=torch.flip(v_tb, dims=[0]),
            diag_ur=diag_ur,
            diag_dl=torch.flip(diag_ur, dims=[0]),
            diag_dr=diag_dr,
            diag_ul=torch.flip(diag_dr, dims=[0]),
        )
        for name, index in indices.items():
            self.register_buffer(f'{name}_index', index)

    @staticmethod
    def _horizontal_index(H, W):
        return torch.arange(H * W, dtype=torch.long)

    @staticmethod
    def _vertical_index(H, W):
        return torch.arange(H * W, dtype=torch.long).reshape(H, W).t().reshape(-1)

    @staticmethod
    def _diag_up_right_index(H, W):
        """Scan every / diagonal from lower-left towards upper-right."""
        grid = torch.arange(H * W, dtype=torch.long).reshape(H, W)
        index = []
        for col in range(W):
            row, cur_col = H - 1, col
            while row >= 0 and cur_col < W:
                index.append(grid[row, cur_col])
                row, cur_col = row - 1, cur_col + 1
        for start_row in range(H - 2, -1, -1):
            row, cur_col = start_row, 0
            while row >= 0 and cur_col < W:
                index.append(grid[row, cur_col])
                row, cur_col = row - 1, cur_col + 1
        return torch.stack(index)

    @staticmethod
    def _diag_down_right_index(H, W):
        """Scan every \\ diagonal from upper-left towards lower-right."""
        grid = torch.arange(H * W, dtype=torch.long).reshape(H, W)
        index = []
        for col in range(W):
            row, cur_col = 0, col
            while row < H and cur_col < W:
                index.append(grid[row, cur_col])
                row, cur_col = row + 1, cur_col + 1
        for start_row in range(1, H):
            row, cur_col = start_row, 0
            while row < H and cur_col < W:
                index.append(grid[row, cur_col])
                row, cur_col = row + 1, cur_col + 1
        return torch.stack(index)

    def scan(self, x, direction):
        B, C, _, _ = x.shape
        index = getattr(self, f'{direction}_index')
        x = x.flatten(2)
        return torch.gather(x, 2, index[None, None, :].expand(B, C, -1))

    def recover(self, y, direction):
        B, C, _ = y.shape
        index = getattr(self, f'{direction}_index')
        x = y.new_zeros(B, C, self.H * self.W)
        return x.scatter(2, index[None, None, :].expand(B, C, -1), y)



if __name__ == "__main__":
    B, C, H, W = 6, 256, 128, 128
    # x = torch.randn(B, C, H, W).cuda().requires_grad_(True)
    x = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W).cuda().requires_grad_(True)
    # print('x:',x,'\n')
    model = DiagScanModule(H, W).cuda()
    fast_model = FastDiagScan(H, W).cuda()
    y_rd, y_ld = model(x)
    y_rd_fast, y_ld_fast = fast_model(x)
    # print('y_rd:',y_rd,'\n')
    # print('y_ld:',y_ld,'\n')
    print(y_ld.shape)
    print(torch.allclose(y_rd, y_rd_fast))
    print(torch.allclose(y_ld, y_ld_fast))

    y_rd = model.recover(y_rd, 'rd')
    y_ld = model.recover(y_ld, 'ld')
    y_ld_fast = fast_model.recover(y_ld_fast, 'ld')
    y_rd_fast = fast_model.recover(y_rd_fast, 'rd')
    # print('x_rd:',y_rd,'\n')
    # print('x_ld:',y_ld,'\n')
    print(y_rd.shape)
    print(torch.allclose(y_rd, y_rd_fast))
    print(torch.allclose(y_ld, y_ld_fast))


    loss = (y_rd + y_ld).sum()
    loss.backward()

    print("x.grad shape:", x.grad.shape)
    from time import time
    
    t0 = time(); y1, y2 = model(x); torch.cuda.synchronize(); print("Old:", time() - t0)
    t0 = time(); y3, y4 = fast_model(x); torch.cuda.synchronize(); print("New:", time() - t0)


    # class A:
    #     def __init__(self, scan):
    #         self.scan = scan

    # shared = DiagScanModule(128, 128)
    # a1 = A(scan=shared)
    # a2 = A(scan=shared)

    # print(id(shared) == id(a1.scan) == id(a2.scan))  # ✅ True：都是同一个
