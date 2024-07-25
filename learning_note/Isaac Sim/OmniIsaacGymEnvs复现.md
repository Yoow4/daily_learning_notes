# OmniIsaacGymEnvs复现

github地址：**[OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs)**

# conda环境配置

创建conda环境

```
conda create -n hyw_isaacGym python=3.10
```

下载代码

```
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
```

激活虚拟环境

```
conda activate hyw_isaacGym
```

进入目录，

```
cd .local/share/ov/pkg/isaac-sim-4.0.0/
```

修改setup_python_env.sh，复制一份

```
cp setup_python_env.sh setup_python_env_conda.sh
gedit setup_python_env_conda.sh
```

修改最后一行，删除`$SCRIPT_DIR/kit/python/lib/python3.10/site-packages:`

备份setup_conda_env.sh

```
cp setup_conda_env.sh setup_conda_env_bak.sh
gedit setup_conda_env.sh
```

把setup_conda_env.sh文件中最后一行的`. ${MY_DIR}/setup_python_env.sh`修改为`. ${MY_DIR}/setup_python_env_conda.sh`

```
source ~/.local/share/ov/pkg/isaac-sim-4.0.0/setup_conda_env.sh
```

接着，回到OmniIsaacGymEnvs路径：

```
python -m pip install -e .
```

# 运行example

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









# conda环境配置_导致isaacsim环境错误

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

无视它,详细报错：

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
nvidia-srl-usd 0.14.0 requires usd-core<24.00,>=21.11, which is not installed.
nvidia-srl-usd-to-urdf 0.6.0 requires usd-core<24.00,>=21.11, which is not installed.
selenium 4.14.0 requires trio~=0.17, which is not installed.
selenium 4.14.0 requires trio-websocket~=0.9, which is not installed.
matplotlib 3.8.4 requires contourpy>=1.0.1, which is not installed.
matplotlib 3.8.4 requires fonttools>=4.22.0, which is not installed.
msal 1.23.0 requires PyJWT[crypto]<3,>=1.0.0, which is not installed.
aiobotocore 2.12.1 requires botocore<1.34.52,>=1.34.41, but you have botocore 1.34.68 which is incompatible.
boto3 1.26.63 requires botocore<1.30.0,>=1.29.63, but you have botocore 1.34.68 which is incompatible.

```

此错误虽然不影响该代码的运行，但会影响isaac sim的正常运行，因为`source ~/.local/share/ov/pkg/isaac-sim-4.0.0/setup_conda_env.sh`该操作后，pip安装的东西会更改isaac sim的python环境，并不是更改了conda出来的虚拟环境。



修复原来isaac sim环境过程

`./python.sh -m pip list |grep numpy`查看发现numpy不见了

安装

```
./python.sh -m pip install numpy==1.26.0
```

报错

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
nvidia-srl-usd 0.14.0 requires tqdm<5.0.0,>=4.63.0, which is not installed.
nvidia-srl-usd 0.14.0 requires usd-core<24.00,>=21.11, which is not installed.
nvidia-srl-usd-to-urdf 0.6.0 requires usd-core<24.00,>=21.11, which is not installed.
matplotlib 3.8.4 requires contourpy>=1.0.1, which is not installed.
matplotlib 3.8.4 requires fonttools>=4.22.0, which is not installed.
gymnasium 0.28.1 requires cloudpickle>=1.2.0, which is not installed.
gymnasium 0.28.1 requires farama-notifications>=0.0.1, which is not installed.
gymnasium 0.28.1 requires jax-jumpy>=1.0.0, which is not installed.

```

安装

```
./python.sh -m pip install tqdm==4.66.4
./python.sh -m pip install hydra-core==1.3.2

./python.sh -m pip install cloudpickle==3.0.0
./python.sh -m pip install jax-jumpy==1.0.0
./python.sh -m pip install farama-notifications==0.0.4

./python.sh -m pip install urllib3==2.2.1

./python.sh -m pip install contourpy==1.0.1
./python.sh -m pip install fonttools==4.22.0
```



原来isaacsim4.0.0的python环境

```
~/.local/share/ov/pkg/isaac-sim-4.0.0$ ./python.sh -m pip list
Package                  Version
------------------------ ------------
aiobotocore              2.12.1
aiodns                   3.1.1
aiofiles                 23.2.1
aiohttp                  3.9.3
aioitertools             0.11.0
aiosignal                1.3.1
annotated-types          0.6.0
anyio                    4.3.0
asteval                  0.9.32
async-timeout            4.0.3
attrs                    23.2.0
azure-core               1.28.0
azure-identity           1.13.0
azure-storage-blob       12.17.0
boto3                    1.26.63
botocore                 1.34.68
cchardet                 2.1.7
certifi                  2024.2.2
cffi                     1.16.0
charset-normalizer       3.3.2
click                    8.1.7
construct                2.10.68
coverage                 7.4.4
cryptography             42.0.7
cycler                   0.11.0
exceptiongroup           1.2.1
fastapi                  0.110.0
filelock                 3.9.0
frozenlist               1.4.1
fsspec                   2024.3.1
gunicorn                 22.0.0
gymnasium                0.28.1
h11                      0.14.0
httptools                0.6.1
idna                     3.7
idna-ssl                 1.1.0
imageio                  2.22.2
isodate                  0.6.1
Jinja2                   3.1.3
jmespath                 1.0.1
kiwisolver               1.4.4
llvmlite                 0.42.0
lxml                     4.9.3
MarkupSafe               2.1.5
matplotlib               3.8.4
mpmath                   1.3.0
msal                     1.23.0
msal-extensions          1.0.0
multidict                6.0.5
nest-asyncio             1.5.6
networkx                 3.2.1
numba                    0.59.1
numpy                    1.26.0
numpy-quaternion         2023.0.3
nvidia-cublas-cu11       11.11.3.6
nvidia-cuda-cupti-cu11   11.8.87
nvidia-cuda-nvrtc-cu11   11.8.89
nvidia-cuda-runtime-cu11 11.8.89
nvidia-cudnn-cu11        8.7.0.84
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.3.0.86
nvidia-cusolver-cu11     11.4.1.48
nvidia-cusparse-cu11     11.7.5.86
nvidia-lula-no-cuda      0.10.1
nvidia-nccl-cu11         2.19.3
nvidia-nvtx-cu11         11.8.86
nvidia_srl_base          0.10.0
nvidia_srl_math          0.9.0
nvidia_srl_usd           0.14.0
nvidia_srl_usd_to_urdf   0.6.0
nvsmi                    0.4.2
oauthlib                 3.2.2
opencv-python-headless   4.9.0.80
osqp                     0.6.5
packaging                23.0
pillow                   10.2.0
Pint                     0.20.1
pip                      21.2.1+nv1
plotly                   5.3.1
portalocker              2.7.0
psutil                   5.9.8
pycares                  4.3.0
pycparser                2.22
pydantic                 2.7.0
pydantic_core            2.18.1
pyparsing                3.0.9
pyperclip                1.8.0
pypng                    0.20220715.0
python-dateutil          2.9.0.post0
python-multipart         0.0.9
pytz                     2024.1
PyYAML                   6.0.1
qdldl                    0.1.7.post1
qrcode                   7.4.2
requests                 2.31.0
requests-oauthlib        1.3.1
s3transfer               0.6.1
scipy                    1.10.1
selenium                 4.14.0
sentry-sdk               1.43.0
setuptools               65.5.1
setuptools-scm           8.0.4
six                      1.16.0
sniffio                  1.3.1
starlette                0.36.3
sympy                    1.12
toml                     0.10.2
tomli                    2.0.1
torch                    2.2.2+cu118
torchaudio               2.2.2+cu118
torchvision              0.17.2+cu118
tornado                  6.2
typing_extensions        4.10.0
urllib3                  2.2.1
uvicorn                  0.29.0
watchdog                 4.0.0
webbot                   0.34
websockets               12.0
wrapt                    1.16.0
yarl                     1.9.4

```

