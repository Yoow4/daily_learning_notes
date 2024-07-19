# omniverse笔记

## 登陆

localhost然后弹出了输入账号密码的页面

帐号密码都为omniverse











# Isaac Sim笔记

# Isaac Examples

## Manipulation

Follow Target 

simple stack





# 快捷键

Q:选择

W：移动

E：旋转

R：缩放



# 界面

## Property







Add——>Physics

Articulation Root 

选择Ariculation Enabled控制机器人关节根节点保证机器人控制精准和流畅

## Stage 



层级化管理





## Window

### viewport



### Extensions

搜索animation timeline window 设置为enabled可以调出timeline12







## Isaac Utils

### Generate Extension Templates

创建一个Template

去Window——>Extensions——>Settings中设置路径

在THIRD PARTY中找到自己创建的extension并设置为enable

在顶部菜单栏可看到自己的extension





## Create

### Physics

Physics Scene物理场景

Ground Plane地面

#### 关节

选中两个物体，

Revolute Joint旋转关节，要对齐旋转轴，不然会乱转



# 保存

save as 

save Flattened as



把想保存的东西选择Set as Default Prim再保存时就不会将光照环境等东西保存下来，方便到时候直接导入



# 打开

open ：可以修改结构

add reference:这个选项不能更改组织内的树状结构

直接拖动也是add reference操作





# 代码魔改 Examples\Hello_world

## Hello_world.py

### setup_scene

代码加载一些需要的模块、属性、物体等

`world = self.get_world()`

这个get_world继承自base_sample

可以考虑用world.instance()

```python
from omni.isaac.core import World
import numpu as np
world = World.instance()

#加入方块
fancy_cube = world.scene.add(
	DynamicCuboid(
    	prim_path = "/World/random_cube"
        name = 
        position = np.array([0,0,1.0])
        scale = np.array([])
        color = np.array([0,0,1.0])
    )
)
```

### setup_post_load

点击load之后的操作



setup_pre_reset重置前的操作：如增加保存

## hello_world_extension.py

修改start_extension中的

name、title、overview等





## add_physics_callback函数

每次按下物理仿真按钮就会使用该回调函数

该函数在omni/isaac/core/simulation_context/simulation_context.py中

调用方法：

```
self.world.add_physics_callback("description of you function",callback_fn=self.yourfunction)#callback names have to be unique
```

## 创建自己的Example

注意：一定要在user_examples里修改，否则GUI内不会显示



修改文件名

__ init __.py中的关系

```
from omni.isaac.examples.my_test_world.test_world import TestWorld
from omni.isaac.examples.my_test_world.test_world_extension import TestWorldExtension
```

以及其他.py文件中涉及的各个函数内涉及到的函数名





## standalone example

先复制一份主函数.py文件

```
cp /home/yoow/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.examples/omni/isaac/examples/hello_world/hello_world.py /home/yoow/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.examples/omni/isaac/examples/hello_world/standalone_helloworld.py
```

移除原来的class,以及别的函数

1、Initialization 初始化:初始化模拟应用程序

```
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
```



2、RESET 重置世界

```
from omni.isaac.core import World
import numpy as np
from omni.isaac.core.objects import DynamicCuboid

world = world()
world.scene.add_default_ground_plane()


#加入方块
fancy_cube = world.scene.add(
	DynamicCuboid(
    	prim_path = "/World/random_cube"
        name = 
        position = np.array([0,0,1.0])
        scale = np.array([])
        co
    )
)

world.reset()

```

记住要做world.reset()



3、Step 模拟执行步骤

```
for i in range(500):
    position,orientation = fancy_cube.get_world_pose()
    linear_velocity = fancy_cube.get_linear_velocity()
    print("position is :"+str(position))
    print("orientation is :"+str(orientation))
    print("linear_velocity is :"+str(linear_velocity))
    
    world.step(render=True)

simulation.close()
```



## 添加机器人





### 轮式机器人

使用WhellRobot类的好处

1、简化初始过程：

- 不需要手动添加引用和创建Robot对象
- 直接使用WheeledRobot类将机器人添加到场景中，简化了代码。

2、更高层次的控制方法：

- 提供了专门用于控制有轮机器人的方法，如apply_wheel_actions，简化了控制逻辑。
- 不需要手动获取和调节关节控制器的方法，更加方便。

3、减少手动操作：

- 不需要手动获取关节控制器，直接调用高层次的方法即可



## Controller









## 机械臂