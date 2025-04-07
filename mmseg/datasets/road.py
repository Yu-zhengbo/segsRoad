import numpy as np
from PIL import Image
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class RoadDataset(BaseSegDataset):
    """Custom dataset class for RGB format segmentation labels."""
    
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]])

    # # 定义颜色到类别的映射
    # COLOR_MAP = {
    #     (0, 0, 0): 0,          # 背景
    #     (255, 255, 255): 1     # 道路
    # }

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        # self.is_255 = kwargs.get('is_255', False)

    # def load_annotations(self, ann_file):
    #     """加载 RGB 格式标签并将其动态转换为单通道类别索引。"""
    #     ann_image = Image.open(ann_file)#.convert('RGB')
    #     if self.is_255:
    #         ann_array = np.array(ann_image)/255.0
    #     else:
    #         ann_array = np.array(ann_image)

    #     # # 创建单通道类别索引图
    #     # single_channel_label = np.zeros((ann_array.shape[0], ann_array.shape[1]), dtype=np.uint8)
        
    #     # for rgb, idx in self.COLOR_MAP.items():
    #     #     mask = np.all(ann_array == rgb, axis=-1)
    #     #     single_channel_label[mask] = idx

    #     return ann_array.astype(np.int)