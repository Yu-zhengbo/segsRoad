import torch
import torch.nn as nn

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
        B, C, L = x.shape
        rd = torch.gather(x, 2, self.rd_index[None, None, :].expand(B, C, -1))
        ld = torch.gather(x, 2, self.ld_index[None, None, :].expand(B, C, -1))
        return rd, ld

    def recover(self, y, mode='rd'):
        B, C, L = y.shape
        index_map = self.rd_index if mode == 'rd' else self.ld_index
        x = torch.zeros(B, C, self.H * self.W, device=y.device, dtype=y.dtype)
        return x.scatter(2, index_map[None, None, :].expand(B, C, -1), y).view(B, C, self.H, self.W)

class FastDiagScanv2(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.register_buffer("rd_index", self.build_diag_index(H, W, mode='rd'))  # [L]
        

    def build_diag_index(self, H, W, mode='rd'):
        index_map = []
        for d in range(H + W - 1):
            for i in range(H):
                j = d - i if mode == 'rd' else i - (H - 1 - d)
                if 0 <= j < W:
                    index_map.append(i * W + j)
        return torch.tensor(index_map, dtype=torch.long)

    def forward(self, x):  # [B, C, H, W]
        B, C, L = x.shape
        if self.H * self.W != x.shape[-1]:
            try:
                rd = torch.gather(x, 2, self.rd_index_new[None, None, :].expand(B, C, -1))
                return rd
            except:
                rd_index_new = self.build_diag_index(int(L ** 0.5), int(L ** 0.5), mode='rd').to(x.device)
                self.register_buffer("rd_index_new", rd_index_new)  # [L]
                rd = torch.gather(x, 2, self.rd_index_new[None, None, :].expand(B, C, -1))
                return rd
        rd = torch.gather(x, 2, self.rd_index[None, None, :].expand(B, C, -1))
        return rd

    def recover(self, y, mode='rd'):
        B, C, L = y.shape
        if self.H * self.W != y.shape[-1]:
            index_map = self.rd_index_new if mode == 'rd' else self.ld_index_new
            x = torch.zeros(B, C, L, device=y.device, dtype=y.dtype)
            return x.scatter(2, index_map[None, None, :].expand(B, C, -1), y).view(B, C, int(L ** 0.5), int(L ** 0.5))
        index_map = self.rd_index if mode == 'rd' else self.ld_index
        x = torch.zeros(B, C, self.H * self.W, device=y.device, dtype=y.dtype)
        return x.scatter(2, index_map[None, None, :].expand(B, C, -1), y).view(B, C, self.H, self.W)


if __name__ == "__main__":
    B, C, H, W = 6, 4, 12, 12
    # x = torch.randn(B, C, H, W).cuda().requires_grad_(True)
    original_data = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H * W).cuda().requires_grad_(True)
    # print('x:',x,'\n')
    fast_model = FastDiagScan(H, W).cuda()
    fast_model_v2 = FastDiagScanv2(H, W).cuda()
    y_rd, y_ld = fast_model(original_data)
    # print('y_rd:',y_rd,'\n')
    # print('y_ld:',y_ld,'\n')

    y_rd_recover = fast_model.recover(y_rd, 'rd')
    y_ld_recover = fast_model.recover(y_ld, 'ld')
    

    print("原始 5x5 数据:")
    print(original_data[0, 0, :])
    
    # b. 模拟扫描过程，得到一维序列
    # original_data = original_data.flip(-1)
    print("\n螺旋扫描后的一维序列:")
    print(y_rd[0, 0, :])

    # c. 从一维序列中恢复二维数据
    print("\n从序列中恢复的 5x5 数据:")
    y_rd_recover = y_rd_recover.flatten(2)
    print(y_rd_recover[0, 0, :])
    print(original_data.shape, y_rd_recover.shape)
    # d. 验证恢复是否正确
    print("\n恢复是否成功:", torch.allclose(original_data, y_rd_recover))


    original_data = original_data.reshape(B, C, H, W).transpose(2, 3).flatten(2)
    y_rd_v2 = fast_model_v2(original_data)
    y_rd_recover_v2 = fast_model_v2.recover(y_rd_v2)
    print("\n螺旋扫描后的一维序列:")
    print(y_rd_v2[0, 0, :])

    print("\n从序列中恢复的 5x5 数据:")
    y_rd_recover_v2 = y_rd_recover_v2.flatten(2)
    print(y_rd_recover_v2[0, 0, :])
    print(original_data.shape, y_rd_recover_v2.shape)
    # d. 验证恢复是否正确
    print("\n恢复是否成功:", torch.allclose(original_data, y_rd_recover_v2))