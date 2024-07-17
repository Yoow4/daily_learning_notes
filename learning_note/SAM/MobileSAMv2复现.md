# MobileSAMv2复现

## Python3.8环境适配

代码环境：python3.8

报错:TypeError: 'type' object is not subscriptable

这个错误通常出现在Python 3.8及更早版本中，因为在这些版本中，你不能直接使用`dict[str, torch.Tensor]`这样的语法来进行类型注解。你需要使用`typing`模块中的`Dict`类型。

在Python 3.9及更高版本，你可以直接使用`dict[str, any]`这样的语法。但在Python 3.8及更低版本，你需要使用`typing`模块中的`Dict`类型。



efficientvit/models/utils/network.py

修改：

```
from typing import List, Dict, Any

#44行
    scale_factor: List[float] or None = None,
    
#61行
def build_kwargs_from_config(config: dict, target_func: callable) -> Dict[str, Any]:

#71行
def load_state_dict_from_file(file: str, only_state_dict=True) -> Dict[str, torch.Tensor]:
```



MobileSAMv2/efficientvit/models/nn/norm.py

```
from typing import  Dict

REGISTERED_NORM_DICT: Dict[str, type] = {
```



MobileSAMv2/efficientvit/apps/trainer/base.py

MobileSAMv2/efficientvit/models/nn/ops.py

MobileSAMv2/efficientvit/models/nn/drop.py

MobileSAMv2/efficientvit/models/efficientvit/backbone.py

MobileSAMv2/efficientvit/models/efficientvit/cls.py

MobileSAMv2/efficientvit/models/efficientvit/sam.py

MobileSAMv2/efficientvit/models/efficientvit/seg.py

同理

```
from typing import List, Dict, Any,Tuple
```









efficientvit/models/utils/random.py

修改

```
from typing import List, Union,Any

#32行
def torch_shuffle(src_list: List[Any], generator: Union[torch.Generator, None] = None) -> List[Any]:

#44行
src_list: List[Any],

#47
weight_list: List[float] or None = None,
```



efficientvit/models/nn/act.py

修改：

```
from typing import Dict, Type
#17行
REGISTERED_ACT_DICT: Dict[str, Type] = {
```





efficientvit/apps/data_provider/augment/bbox.py

```
from typing import Tuple
#16行
) -> Tuple[int, int, int, int]:
```



MobileSAMv2/efficientvit/apps/data_provider/augment/color_aug.py

```
from typing import Dict, Tuple,Any
#58行
def __init__(self, config: Dict[str, Any], mean: Tuple[float, float, float], key="data"):
```



MobileSAMv2/efficientvit/apps/data_provider/random_resolution/controller.py

```
from typing import List,Tuple	
#27行
def get_candidates() -> List[Tuple[int, int]]:
```



MobileSAMv2/efficientvit/apps/data_provider/base.py

```
from typing import Tuple,Any
#line 17
def parse_image_size(size: int or str) -> Tuple[int, int]:

def data_shape(self) -> Tuple[int, ...]:

 def build_valid_transform(self, image_size: Tuple[int, int] or None = None) -> any:
 
 def build_train_transform(self, image_size: Tuple[int, int] or None = None) -> any:
 
 def build_datasets(self) -> Tuple[any, any, any]:
 
 def sample_val_dataset(self, train_dataset, valid_transform) -> Tuple[Any, Any]:
```



MobileSAMv2/efficientvit/apps/utils/ema.py

```
from typing import Dict

from typing import Dictdef update_ema(ema: nn.Module, new_state_dict: Dict[str, torch.Tensor], decay: float) -> None:


    def state_dict(self) -> Dict[float, Dict[str, torch.Tensor]]:
    
    def load_state_dict(self, state_dict: Dict[float, Dict[str, torch.Tensor]]) -> None:
```





MobileSAMv2/efficientvit/apps/utils/lr.py

```
from typing import List

   decay_steps: int or List[int],
   
   def get_lr(self) -> List[float]:
```



MobileSAMv2/efficientvit/apps/utils/opt.py

```
from typing import Dict,Tuple,Any

REGISTERED_OPTIMIZER_DICT: Dict[str, Tuple[type, Dict[str, Any]]] = {
```



MobileSAMv2/efficientvit/apps/trainer/run_config.py

```

```





## 环境依赖

报错：File "/home/user/SAM/MobileSAM/MobileSAMv2/efficientvit/apps/trainer/base.py", line 9, in <module>
    import torchpack.distributed as dist
ModuleNotFoundError: No module named 'torchpack'

```
pip install torchpack
```





报错：  File "/home/user/SAM/MobileSAM/MobileSAMv2/efficientvit/apps/utils/export.py", line 8, in <module>
    import onnx
ModuleNotFoundError: No module named 'onnx'

```
pip install onnx==1.12.0
pip install onnxruntime==1.13.1
```

File "/home/user/SAM/MobileSAM/MobileSAMv2/efficientvit/apps/utils/export.py", line 11, in <module>
    from onnxsim import simplify as simplify_func
ModuleNotFoundError: No module named 'onnxsim'

```
pip install onnx-simplifier
```





# 代码修改

MobileSAMv2/Inference.py

增加检查输出文件是否存在

```
if not os.path.exists(output_dir):
            os.makedirs(output_dir)
```

清理内存

```
# 处理输入框的循环
    for (boxes,) in batch_iterator(320, input_boxes):  # 可能需要调整320的值
        with torch.no_grad():
            # 动态调整张量大小
            image_embedding = image_embedding[:boxes.shape[0]]
            prompt_embedding = prompt_embedding[:boxes.shape[0]]

            # ... (保留mask预测的代码)

            # 清理内存
            del sparse_embeddings, dense_embeddings, low_res_masks, sam_mask_pre
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

保存图片后也要关闭图像清理内存,否则会额外增加不止3G的占用

```
if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig("{}".format(output_dir+image_name), bbox_inches='tight', pad_inches = 0.0) 
        plt.close()  # 关闭当前图像
```

