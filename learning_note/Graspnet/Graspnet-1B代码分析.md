# graspnet-1B代码分析

```python
GraspNet(
  (view_estimator): GraspNetStage1(
    (backbone): Pointnet2Backbone(
      (sa1): PointnetSAModuleVotes(
        (grouper): QueryAndGroup()
        (mlp_module): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
      (sa2): PointnetSAModuleVotes(
        (grouper): QueryAndGroup()
        (mlp_module): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
      (sa3): PointnetSAModuleVotes(
        (grouper): QueryAndGroup()
        (mlp_module): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
      (sa4): PointnetSAModuleVotes(
        (grouper): QueryAndGroup()
        (mlp_module): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer2): Conv2d(
            (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
      (fp1): PointnetFPModule(
        (mlp): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
      (fp2): PointnetFPModule(
        (mlp): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): ReLU(inplace=True)
          )
        )
      )
    )
    (vpmodule): ApproachNet(
      (conv1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      (conv2): Conv1d(256, 302, kernel_size=(1,), stride=(1,))
      (conv3): Conv1d(302, 302, kernel_size=(1,), stride=(1,))
      (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): BatchNorm1d(302, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (grasp_generator): GraspNetStage2(
    (crop): CloudCrop(
      (mlps): SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace=True)
        )
        (layer1): Conv2d(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace=True)
        )
        (layer2): Conv2d(
          (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU(inplace=True)
        )
      )
    )
    (operation): OperationNet(
      (conv1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      (conv2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      (conv3): Conv1d(128, 36, kernel_size=(1,), stride=(1,))
      (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (tolerance): ToleranceNet(
      (conv1): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
      (conv2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      (conv3): Conv1d(128, 12, kernel_size=(1,), stride=(1,))
      (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
)
```

