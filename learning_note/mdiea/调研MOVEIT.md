# 调研MOVEIT

能否直接通过C++调用

如果不能，能否在不安装ROS的情况下调用





MoveIt官方文档：基于ROS的调用

## Using MoveIt Directly Through the C++ API[](https://moveit.picknik.ai/main/doc/examples/examples.html#using-moveit-directly-through-the-c-api)

https://moveit.picknik.ai/main/doc/examples/examples.html#using-moveit-directly-through-the-c-api







## 可以自己修改?

如：[swri-robotics/iterative_spline_parameterization: MoveIt's iterative_spline_parameterization utility without the ROS- and MoveIt-dependent stuff (github.com)](https://github.com/swri-robotics/iterative_spline_parameterization)

这是 MoveIt 迭代样条参数化轨迹处理工具的修改版本，消除了对 ROS 和 MoveIt 特定库的依赖。它仍然使用 Eigen 库来表示向量，尽管这在技术上可能不是必需的并且将来可以简化。



主要代码定义了一个名为`iterative_spline_parameterization`的命名空间，其中包含一个名为`IterativeSplineParameterization`的类。这个类主要用于根据给定的关节轨迹数据，通过迭代优化方法来计算每个关节的时间戳，以确保轨迹在速度和加速度约束内。

1. 类中定义了几个静态成员函数，如`fit_cubic_spline`、`adjust_two_positions`、`init_times`、`fit_spline_and_adjust_times`和`global_adjustment_factor`，这些函数分别用于拟合三次样条曲线、调整两个位置值、初始化时间间隔、调整时间间隔以满足约束条件以及计算全局调整因子。

2. `TrajectoryState`结构体用于表示单个关节的轨迹状态，包括位置、速度和加速度，以及时间戳。

3. `IterativeSplineParameterization`类的构造函数接受一个布尔参数`add_points`，表示是否需要在轨迹的两端添加额外的点。

4. `computeTimeStamps`成员函数接受多个参数，包括关节轨迹、最大速度、最大加速度、最大速度缩放因子和最大加速度缩放因子。该函数首先对输入进行错误检查，然后根据输入参数计算每个关节的时间戳，确保轨迹在速度和加速度约束内。

5. 在`computeTimeStamps`函数中，首先将输入的关节轨迹转换为适合处理的形式，然后使用迭代优化方法来计算每个关节的时间戳。最后，将计算出的时间戳转换回原始的关节轨迹形式，并返回结果。



1. `fit_cubic_spline` 函数：
   - 输入参数包括：点的数量n、时间差数组dt、位置数组x、一阶导数数组x1和二阶导数数组x2。
   - 使用追赶法（Tridiagonal algorithm）求解三阶多项式插值，得到一阶和二阶导数。
   - 计算并返回一个全局调整因子tfactor，以使轨迹在速度和加速度范围内。

2. `adjust_two_positions` 函数：
   - 调整两个特定位置的时间间隔，使得轨迹的二阶导数在指定范围内。

3. `init_times` 函数：
   - 根据最大和最小速度初始化每个时间间隔。

4. `fit_spline_and_adjust_times` 函数：
   - 拟合三次样条曲线，并根据速度和加速度限制调整时间间隔。

5. `global_adjustment_factor` 函数：
   - 计算全局调整因子，确保整个轨迹在速度和加速度范围内。

6. `globalAdjustment` 函数：
   - 对整个轨迹进行全局调整，使其在速度和加速度范围内。

### Extremely Basic Usage Example

这个函数`do_parameterization`的主要作用是将给定的关节轨迹（trajectory_msgs::msg::JointTrajectory）进行参数化处理，将其转换为更适合计算机模拟和控制系统的格式。具体步骤如下：

1. 创建一个名为`waypoints`的`std::vector`，用于存储轨迹中的关键点（即关节角度）。

2. 遍历输入的`trajectory_msgs::msg::JointTrajectory`中的每一个点（`traj_in.points`），将每个点的关节角度从消息格式（vector<double>）转换为Eigen矩阵（Eigen::Matrix<double, 6, 1>），因为迭代样条参数化算法需要这种格式的数据。

3. 将每个关键点的信息（包括关节角度、速度和加速度以及时间戳）封装到一个`iterative_spline_parameterization::TrajectoryState`对象中，并将其添加到`waypoints`向量中。 

4. 创建一个`iterative_spline_parameterization::IterativeSplineParameterization`对象`isp`，并设置其参数（这里假设为false）。

5. 调用`isp.computeTimeStamps(waypoints, 1.0, 1.0)`方法来计算关键点之间的时间戳，使得轨迹具有连续性和平滑性。这里的参数1.0表示每个关键点之间的平均时间间隔，可以根据实际需求进行调整。

6. 再次遍历`waypoints`向量，将每个关键点的信息从`iterative_spline_parameterization::TrajectoryState`对象转换回`trajectory_msgs::msg::JointTrajectoryPoint`格式，并将结果添加到输出轨迹`traj_out`的点列表中。

7. 最后，将每个`trajectory_msgs::msg::JointTrajectoryPoint`对象的时间戳转换为ROS时间格式（rclcpp::Duration），并将关节角度、速度和加速度从Eigen矩阵映射到ROS消息格式（vector<double>）。



查阅函数发现还是需要最后转换到ROS消息格式下运行



## 讨论robotics.stackexchange.com

[robotic arm - Can MoveIT framework be used separately from ROS, i.e. "as a standalone library"? - Robotics Stack Exchange](https://robotics.stackexchange.com/questions/23880/can-moveit-framework-be-used-separately-from-ros-i-e-as-a-standalone-library)

只需重写使用 ROS 消息和 ROS 传输机制的函数，这样您就不需要使用 ROS。





## Robowflex: Robot Motion Planning with MoveIt Made Easy

论文：https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj_toHA8_OGAxVgslYBHfHuC9w4ChAWegQIIhAB&url=https%3A%2F%2Fwww.kavrakilab.org%2Fpublications%2Fkingston2022-robowflex.pdf&usg=AOvVaw0splGqm98-WNrZ8PDxa3fz&opi=89978449

源码:[KavrakiLab/robowflex: Making MoveIt Easy! (github.com)](https://github.com/KavrakiLab/robowflex)

还是用ROS的,只是变简单了-。-









# 不用MoveIt：

[10 Alternatives to ROS for robots | Kshitij Tiwari Ph.D.](https://kshitijtiwari.com/all-resources/mobile-robots/alternatives-to-ros/)

- [PyRobot ](https://pyrobot.org/)– this is an open-source robotics framework which is particularly well-suited for academic and research projects. PyRobot provides a unified API to control robots from different vendors and allows for easy development and deployment of experiments. Offers functionality for implementing manipulation, navigation and pick-and-place demonstrations.

  

- [Orca ](https://orca-robotics.sourceforge.net/)– this is a C++ framework for robotics development which is designed for embedded systems and low-power applications. It provides a software stack for controlling low-level peripherals, as well as support for higher-level control algorithms such as motion planning and machine learning.

  

- [Yet Another Robot Platform (YARP)](https://yarp.it/latest/) – this is an open-source robot operating system (ROS), written in C++ and capable of running on Linux and Windows. It is flexible and extensible and provides a range of tools for automation, robotics, and machine vision.

  

- [Open Real-time Control System (OROCOS)](https://www.orocos.org/rtt/) – this is an open-source real-time control system (RTS) designed for robotics applications. It is written in C++ and targets the Linux platform and is capable of distributed execution with support for real-time scheduling.

  

- [genom3 ](https://git.openrobots.org/projects/genom3)– this is a distributed robotics control system which is designed for real-time, safety-critical robotics applications. It is written in C++ and provides a range of features for real-time scheduling, message passing, and distributed control.

  

- [Mobile Robot Programming Toolkit (MRPT)](https://docs.mrpt.org/reference/latest/) – this is an open-source C++ library for robotics applications. It provides features for mapping, localization, path planning, navigation, and computer vision and is easily extensible and scalable.

  

- Integrated Development Environments (IDEs)– IDEs like [NAOqi](http://doc.aldebaran.com/2-5/index_dev_guide.html) are suited for Aldebaran robots such as Pepper and NAO, while [V-REP](https://www.coppeliarobotics.com/) is suitable for a wide range of robots, including those running on ROS.

  
  Similarly, [Webots ](https://cyberbotics.com/)is a powerful and popular robotics development environment designed to facilitate the development of robotics applications. It provides an integrated development environment (IDE) for writing and debugging code, a physics engine for simulating dynamic systems, and a visual programming language for creating control logic. Webots also includes libraries for a wide range of sensors, actuators and controllers, as well as a number of tools for 3D modeling.

  

- [Player Project](https://playerstage.sourceforge.net/) – this is an open-source frameworks for mobile robotics research and development which is capable of controlling a range of robots. It provides easy-to-use APIs for controlling a wide range of robots, as well as a number of other features such as obstacle avoidance and path planning.

# Orca框架

[Orca ](https://orca-robotics.sourceforge.net/)– this is a C++ framework for robotics development which is designed for embedded systems and low-power applications. It provides a software stack for controlling low-level peripherals, as well as support for higher-level control algorithms such as motion planning and machine learning.

Orca – 这是一个用于机器人开发的 C++ 框架，专为嵌入式系统和低功耗应用而设计。它提供了用于控制低级外设的软件堆栈，以及对运动规划和机器学习等高级控制算法的支持。

查阅[Orca Robotics (sourceforge.net)](https://orca-robotics.sourceforge.net/group__hydro__libs.html)发现是2D的运动规划-。-



# OROCOS——KDL库

官网：[The Orocos Project | Smarter control in robotics & automation!](https://www.orocos.org/index.html)

源码：[orocos/orocos_kinematics_dynamics: Orocos Kinematics and Dynamics C++ library (github.com)](https://github.com/orocos/orocos_kinematics_dynamics)

## trajectory.cpp

这段代码是C++实现的一个库，用于处理运动轨迹。主要功能包括读取和写入轨迹数据，以及创建不同类型的轨迹对象。

1. `Trajectory* Trajectory::Read(std::istream& is)`：这是一个静态成员函数，用于从输入流（如文件或网络流）中读取轨迹数据并创建相应的轨迹对象。函数首先读取一个字符串，判断其是否为"SEGMENT"，如果是，则表示接下来是一个轨迹段（Trajectory_Segment）。

   - 使用`scoped_ptr`来管理动态分配的内存，确保在读取完成后自动释放内存。
   - 调用`Path::Read(is)`读取轨迹段的几何路径，然后调用`VelocityProfile::Read(is)`读取运动速度规划。
   - 读取结束标志'['，并返回创建的`Trajectory_Segment`对象。

2. 在其他部分，如`Path::Read(is)`和`VelocityProfile::Read(is)`，分别用于读取路径和运动速度规划的具体内容。

总的来说，这段代码实现了从文本格式的输入流中读取轨迹数据，并将其转换为相应的轨迹对象。



## trajectory_stationary.cpp

这段代码是C++实现的一个库，用于处理静止轨迹。主要功能包括将静止轨迹对象写入输出流（如文件或网络流）。

1. `void Trajectory_Stationary::Write(std::ostream& os) const`：这是一个成员函数，用于将静止轨迹对象的内容写入输出流。

   - 首先，输出字符串"STATIONARY["，后面跟着静止轨迹的持续时间`duration`。
   - 然后，输出轨迹的当前位置`pos`。
   - 最后，输出结束标志']'。

整个函数的主要目的是将静止轨迹的信息以特定格式写入输出流，以便于后续的处理或保存。



## trajectory_segment.cpp

这段代码是C++中的一个库，用于处理运动轨迹。`Trajectory_Segment`类是一个表示轨迹段的数据结构，它包含一个路径（Path）和一个速度规划（VelocityProfile）。

在构造函数中，如果用户没有提供速度规划，则根据给定的路径长度和持续时间自动设置速度规划。同时，还可以指定是否需要聚合（aggregate）资源，即是否在析构时释放路径和速度规划的内存。

`Duration()`方法返回轨迹段的持续时间。

`Pos(double time)`、`Vel(double time)`和`Acc(double time)`方法分别返回在给定时间`time`时的位置、速度和加速度。

`Write(std::ostream& os)`方法将轨迹段的信息写入输出流。

`GetPath()`和`GetProfile()`方法分别返回路径和速度规划对象。

总之，这个库提供了创建和操作轨迹段的功能，包括计算轨迹段的时间、位置、速度和加速度等属性，以及将轨迹段信息写入输出流的功能。

## trajectory_composite.cpp

这个头文件定义了一个名为`Trajectory_Composite`的类，它继承自`Trajectory`类。这个类主要用于表示由多个子轨迹（Trajectory）组成的复合轨迹。

在`Trajectory_Composite`类中，定义了两个私有成员变量：

1. `VectorTraj vt`：一个包含`Trajectory`指针的向量，用于存储构成复合轨迹的所有子轨迹。
2. `VectorDouble vd`：一个包含每个子轨迹结束时间的向量，用于记录每个子轨迹在复合轨迹中的持续时间。
3. `double duration`：复合轨迹的总持续时间。

类的构造函数`Trajectory_Composite()`初始化这些成员变量。

提供了以下公共成员函数：

1. `double Duration() const`：返回复合轨迹的总持续时间。
2. `Frame Pos(double time) const`：根据给定的时间`time`计算并返回复合轨迹在指定时间处的位置。
3. `Twist Vel(double time) const`：根据给定的时间`time`计算并返回复合轨迹在指定时间处的速度。
4. `Twist Acc(double time) const`：根据给定的时间`time`计算并返回复合轨迹在指定时间处的加速度。
5. `void Add(Trajectory* elem)`：将指定的子轨迹`elem`添加到复合轨迹的末尾。
6. `void Destroy()`：销毁并释放复合轨迹中所有子轨迹的内存。
7. `void Write(std::ostream& os) const`：将复合轨迹的信息写入输出流`os`。
8. `Trajectory* Clone() const`：创建并返回复合轨迹的一个副本。
9. `virtual ~Trajectory_Composite()`：析构函数，释放资源。

总之，这个类用于表示和操作由多个子轨迹组成的复合轨迹，提供了添加、获取信息、克隆和销毁子轨迹的功能。



# MoveIt源码

[moveit/moveit: :robot: The MoveIt motion planning framework (github.com)](https://github.com/moveit/moveit)



## 对比MoveIt中的robot_trajectory.cpp

这段代码定义了一个名为`RobotTrajectory`的类，它用于表示机器人运动轨迹。该类包含了一系列方法，如获取轨迹组名、计算总时长、平均每段时长、交换轨迹信息、将轨迹追加到现有轨迹、反转轨迹、解包连续关节、将轨迹转换为ROS消息格式等。

在`RobotTrajectory`类中，`getRobotTrajectoryMsg`方法用于将当前轨迹对象转换为ROS消息格式，而`setRobotTrajectoryMsg`方法则用于从ROS消息格式中设置轨迹信息。此外，还提供了`findWayPointIndicesForDurationAfterStart`、`getWayPointDurationFromStart`和`getStateAtDurationFromStart`等方法，用于根据给定的时间点获取轨迹中的特定状态。

最后，`path_length`函数用于计算给定轨迹的长度。



以下是`RobotTrajectory`类中每个函数的详细功能介绍：

1. `RobotTrajectory(const moveit::core::RobotModelConstPtr& robot_model)`：构造函数，初始化一个空的`RobotTrajectory`对象，其中`robot_model_`指向传入的机器人模型，`group_`为空指针。

2. `RobotTrajectory(const moveit::core::RobotModelConstPtr& robot_model, const std::string& group)`：构造函数，初始化一个指定组名的`RobotTrajectory`对象，其中`robot_model_`指向传入的机器人模型，`group_`指向该组名对应的关节模型组。

3. `RobotTrajectory(const moveit::core::RobotModelConstPtr& robot_model, const moveit::core::JointModelGroup* group)`：构造函数，初始化一个指定关节模型组的`RobotTrajectory`对象，其中`robot_model_`指向传入的机器人模型，`group_`指向传入的关节模型组。

4. `RobotTrajectory(const RobotTrajectory& other, bool deepcopy)`：拷贝构造函数，根据传入的`RobotTrajectory`对象`other`创建一个新的`RobotTrajectory`对象。如果`deepcopy`为真，则进行深拷贝；否则进行浅拷贝。

5. `const std::string& getGroupName() const`：返回轨迹组名。

6. `double getDuration() const`：返回整个轨迹的总时长。

7. `double getAverageSegmentDuration() const`：返回每段轨迹的平均时长。

8. `void swap(RobotTrajectory& other)`：交换两个`RobotTrajectory`对象的内容。

9. `RobotTrajectory& append(const RobotTrajectory& source, double dt, size_t start_index, size_t end_index)`：将另一个轨迹对象`source`从索引`start_index`到`end_index`的部分追加到当前轨迹中，并添加一段从当前轨迹末尾到新轨迹开始的时间差`dt`。

10. `RobotTrajectory& reverse()`：反转当前轨迹，即将所有轨迹点逆序，同时将速度向量也进行逆序。

11. `RobotTrajectory& unwind()`：解包连续关节，确保连续关节的值不会超过其上下界。

12. `RobotTrajectory& unwind(const moveit::core::RobotState& state)`：与`unwind()`类似，但使用给定的参考状态来解包连续关节。

13. `void getRobotTrajectoryMsg(moveit_msgs::RobotTrajectory& trajectory, const std::vector<std::string>& joint_filter) const`：将当前轨迹转换为ROS消息格式，其中`trajectory`是输出的消息，`joint_filter`是一个字符串列表，用于指定需要包含的关节。

14. `RobotTrajectory& setRobotTrajectoryMsg(const moveit::core::RobotState& reference_state, const trajectory_msgs::JointTrajectory& trajectory)`：从ROS消息格式的`trajectory`中设置当前轨迹，其中`reference_state`是参考状态。

15. `RobotTrajectory& setRobotTrajectoryMsg(const moveit::core::RobotState& reference_state, const moveit_msgs::RobotTrajectory& trajectory)`：从ROS消息格式的`trajectory`中设置当前轨迹，其中`reference_state`是参考状态。

16. `RobotTrajectory& setRobotTrajectoryMsg(const moveit::core::RobotState& reference_state, const moveit_msgs::RobotState& state, const moveit_msgs::RobotTrajectory& trajectory)`：从ROS消息格式的`trajectory`中设置当前轨迹，其中`reference_state`是参考状态，`state`是参考状态。

17. `void findWayPointIndicesForDurationAfterStart(const double duration, int& before, int& after, double& blend) const`：根据给定的时间点`duration`，找到在该时间点之后最近的两个轨迹点，并计算出这两个轨迹点之间的混合比例`blend`。

18. `double getWayPointDurationFromStart(std::size_t index) const`：返回指定索引处的轨迹点相对于起点的时间。

19. `double getWaypointDurationFromStart(std::size_t index) const`：与`getWayPointDurationFromStart`相同。

20. `bool getStateAtDurationFromStart(const double request_duration, moveit::core::RobotStatePtr& output_state) const`：根据给定的时间点`request_duration`，获取该时间点处的轨迹状态，并将其存储在`output_state`中。

