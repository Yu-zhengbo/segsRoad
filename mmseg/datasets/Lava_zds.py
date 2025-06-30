import numpy as np
from PIL import Image
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union
import mmengine
import mmengine.fileio as fileio


@DATASETS.register_module()
class LavaDataset(BaseSegDataset):
    """Custom dataset class for RGB format segmentation labels."""
    
    METAINFO = dict(
        classes=('background', 'lava'),
        palette=[[0, 0, 0], [255, 255, 255]])

    # # 定义颜色到类别的映射
    # COLOR_MAP = {
    #     (0, 0, 0): 0,          # 背景
    #     (255, 255, 255): 1     # 道路
    # }

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 dem_suffix='.tif',
                 reduce_zero_label=False,
                 data_prefix: dict = dict(img_path='', seg_map_path='',dem_path=''),
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_prefix=data_prefix,
            **kwargs)
        
        self.dem_suffix = dem_suffix
        


    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        dem_dir = self.data_prefix.get('dem_path', None)


        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            if dem_dir is not None:
                dem_name = img[:-_suffix_len] + '.tif'
                data_info['dem_path'] = osp.join(dem_dir, dem_name)
            if ann_dir is not None:
                seg_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list