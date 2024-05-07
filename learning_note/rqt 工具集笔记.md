# rqt 工具集笔记

## Node Graph

```
rqt_graph
```

或

```
rosrun rqt_gui rqt_gui
```

选择plugins中的introspection下的Node Graph

## TF Tree

```
rosrun rqt_tf_tree rqt_tf_tree
```

或

```
rosrun rqt_gui rqt_gui
直接用也可以
rqt
```

选择plugins中的Visualization下的TF Tree

## Plot

```
rqt_plot
```

命令行查看输出可以用

```
rostopic echo /<话题名>
```

## Dynamic Reconfigure

```
rqt
```

选择plugins中的Robot Tools下的Dynamic Reconfigure

## Image View

```
rqt
```

选择plugins中的Visualization下的Image View

## Bag

```
rosbag record -a 							记录所有话题
rosbag play <name.bag> 						播放一次
rosbag play -r num <name.bag> 				倍速播放
rosbag play <name.bag>--topic /topic_name 	只播放某个话题
```

