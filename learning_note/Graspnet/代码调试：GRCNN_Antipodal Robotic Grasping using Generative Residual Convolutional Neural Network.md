# 代码调试：Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

- [skumra](https://github.com/skumra)/
- [robotic-grasping](https://github.com/skumra/robotic-grasping)

## 训练：

python train_network.py --dataset jacquard --dataset-path <Path To Dataset> --description training_jacquard --use-dropout 0 --input-size 300

```
python train_network.py --dataset jacquard --dataset-path '/home/ubuntu/data0/hyw/Jacquard' --description training_jacquard --use-dropout 0 --input-size 300
```

## 评估：

python evaluate.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --iou-eval --use-dropout 0 --input-size 300

```
python evaluate.py --network '/home/ubuntu/data0/hyw/robotic-grasping/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93' --dataset jacquard --dataset-path '/home/ubuntu/data0/hyw/Jacquard' --iou-eval --input-size 300

python evaluate.py --network '/home/ubuntu/data0/hyw/robotic-grasping/logs/240312_1016_training_jacquard/epoch_26_iou_0.91' --dataset jacquard --dataset-path '/home/ubuntu/data0/hyw/Jacquard' --iou-eval --input-size 300
```

## 报错

**错误1：**ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package

```
pip install imagecodecs-lite
```

**错误2：**AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.

解决方法：降级你的numpy版本到1.23.5

```
pip install numpy==1.23.5
```



**错误3：**更换GPU运行时报错：RuntimeError: Expected all tensors to be on the same device, but found at leasttwo devices, cuda:0 and cuda:1! (when checking argument for argument weight in method wrapper cudnn convolution)



解决方法：**设置全局默认设备**：

```
    torch.cuda.set_device(1)  # 设置全局默认设备为 cuda:1
    device = torch.device('cuda:1')
```

