# OmniIsaacGymEnvs复现

github地址：**[OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs)**

# 环境配置

创建conda环境

```
conda create -n hyw_isaacGym python=3.10
```

下载代码

```
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
```

此处是isaac-sim-*，github上写错了	

```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac-sim-*/python.sh
```



忽然发现conda环境是另外的配置方法，上面的命令没用，查看~/.local/share/ov/pkg/isaac-sim-*/python.sh文件发现

`echo "If conda is desired please source setup_conda_env.sh in your python 3.10 conda env and run python normally"`

故运行：（以后可能每次运行都得先用这句给定代码需要的环境路径）

```
source ~/.local/share/ov/pkg/isaac-sim-4.0.0/setup_conda_env.sh
```

接着，回到OmniIsaacGymEnvs路径：

```
python -m pip install -e .
```

此时会像github所说报错：

`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.`

无视它



## 运行example

由于是conda 环境，在OmniIsaacGymEnvs/omniisaacgymenvs路径下运行

```
python scripts/rlgames_train.py task=Cartpole
```

此时需要等待几分钟完成加载，不用退出GUI，等就行。

为了获得最佳性能，加参数

```
python scripts/rlgames_train.py task=Ant headless=True
```

此时，没有图形界面，会在终端自己训练

```
saving next best rewards:  [5771.3906]
=> saving checkpoint '/home/midea/hyw/code/OmniIsaacGymEnvs/omniisaacgymenvs/runs/Ant/nn/Ant.pth'
fps step: 261574 fps step and policy inference: 230705 fps total: 181660 epoch: 376/500 frames: 24576000
fps step: 266194 fps step and policy inference: 232788 fps total: 185195 epoch: 377/500 frames: 24641536
fps step: 288666 fps step and policy inference: 253068 fps total: 197888 epoch: 378/500 frames: 24707072
.
.
.
fps step: 238445 fps step and policy inference: 208654 fps total: 163718 epoch: 499/500 frames: 32636928
fps step: 234007 fps step and policy inference: 205768 fps total: 173824 epoch: 500/500 frames: 32702464
=> saving checkpoint '/home/midea/hyw/code/OmniIsaacGymEnvs/omniisaacgymenvs/runs/Ant/nn/last_Ant_ep_500_rew_5943.272.pth'
saving next best rewards:  [5943.272]
=> saving checkpoint '/home/midea/hyw/code/OmniIsaacGymEnvs/omniisaacgymenvs/runs/Ant/nn/Ant.pth'
=> saving checkpoint '/home/midea/hyw/code/OmniIsaacGymEnvs/omniisaacgymenvs/runs/Ant/nn/last_Ant_ep_500_rew__5943.272_.pth'
MAX EPOCHS NUM!
2024-07-23 07:28:31 [411,663ms] [Warning] [omni.usd] Unexpected reference count of 2 for UsdStage 'anon:0x5e6ae89b91f0:World0.usd' while being closed in UsdContext (this may indicate it is still resident in memory).
[411.694s] Simulation App Shutting Down
2024-07-23 07:28:31 [411,774ms] [Warning] [carb.audio.context] 1 contexts were leaked
```

训练结束。

# Extension workflow

扩展工作流程提供了一个简单的用户界面，用于创建和启动 RL 任务。要为扩展工作流程启动 Isaac Sim，请运行：

```
./<isaac_sim_root>/isaac-sim.gym.sh --ext-folder </parent/directory/to/OIGE>
```

注意：`isaac_sim_root `应与 `python.sh `位于同一目录中。





# Demo

创建一个有64个机器人的demo，看了下显存占用才6GB

```
python scripts/rlgames_demo.py task=AnymalTerrain num_envs=64 checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/anymal_terrain.pth
```

在此演示中，您可以单击场景中的任何 ANYmals 进入第三人称模式并使用键盘手动控制机器人，

如下所示：     

向上箭头：向前线速度指令    

下箭头：向后线速度指令    

左箭头：向左线速度指令    

右箭头：向右线速度指令    

Z：逆时针偏航角速度指令    

X：顺时针偏航角速度指令    

C：在第三人称和场景视图之间切换摄像机视图，同时保持手动控制    

ESC：取消选择选定的 ANYmal 并进行手动控制 

请注意，此演示将场景中 ANYmal 的最大数量限制为 128。





