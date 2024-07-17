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