# SAM with mmdet

复现结果失败，情况类似

https://github.com/liuyanyi/sam-with-mmdet/issues/2



更换上一个commit版本

https://github.com/liuyanyi/sam-with-mmdet/tree/f8252d14fdbcbbae7c04ab6099ff06bcba1b7667

目前不知道怎么修改.config

# 复现

回退到上一个[commit](https://github.com/liuyanyi/sam-with-mmdet/tree/f8252d14fdbcbbae7c04ab6099ff06bcba1b7667) 的代码

[liuyanyi/sam-with-mmdet at f8252d14fdbcbbae7c04ab6099ff06bcba1b7667 (github.com)](https://github.com/liuyanyi/sam-with-mmdet/tree/f8252d14fdbcbbae7c04ab6099ff06bcba1b7667)

下载vit_b

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

下载.pth

```
mim download mmdet --config rtmdet_l_8xb32-300e_coco --dest .
```

修改merge_sam_det.py和demo.py中的路径。



解决报错：

```
mmengine.config.utils.ConfigParsingError: The configuration file type in the inheritance chain must match the current configuration file type, either "lazy_import" or non-"lazy_import".You got this error since you use the syntax like `_base_ = "_base_"` in your config.You should use `with read_base(): ... to` mark the inherited config file.See more information in https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html
```

经测试，将rtm_l_sam_b.py文件中的from functools import partial修改成

```python
from mmengine.config import read_base

with read_base():
	from functools import partial
```

后面操作不变即可复现结果



