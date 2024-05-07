[TOC]



# Git指令

## 基本操作

初始化仓库

```
git init
```

查看仓库状态

```
git status
```

向暂存区中添加文件

```
git add
```

保存仓库的历史记录

```
git commit -m "修改内容"
git commit              #此时为打开编辑器写修改内容
```

合在一起：

```
git commit --am "修改内容"
```

查看提交日志

```
git log
参数：
--graph  #树图形方式查看分支
--pretty=short #只显示提交信息的第一行
--目录名   #只显示该目录、文件下的日志
-p       #显示文件的改动
```

查看更改前后的差别

```
git diff
git diff HEAD #查看与最新提交的差别
```

## 分支操作

显示分支一览表

```
git branch
```

创建、切换分支

```
git checkout -b	#创建并切换分支
git checkout	#创建分支
git branch -a	#显示全部分支
```

合并分支

```
git merge --no-ff	#合并并提交
```

## 更改提交的操作

回溯历史版本

```
git reset --hard 哈希值
```

推进至某一个历史状态

```
git reflog
```

合并时冲突部分

```
=======以上的部分是HEAD的内容，以下的部分是要合并的内容
```

修改提交信息

```
git commit --amend
```

压缩历史

```
git rebase -i HEAD~2	#选定当前分支中包含HEAD（最新提交）在内的两个最新历史记录为对象
```

## 推送至远程仓库

添加远程仓库

```
git remote add
git remote add origin git@github.com:yoow4/az.git  
#Git将远程仓库的名称设置为origin（标识符）
```

推送至远程仓库

```
git push -u origin master
#-u参数可在推送的同时将origin仓库的master分支设置为本地仓库当前分支的upstream
```

推送至master以外的分支

```
git push -u origin feature-D
```

## 从远程仓库获取

从远程仓库获取

```
git clone git@github.com:yoow4/az.git 
```

获取远程的feature-D分支

```
git checkout -b feature-D origin/feature-D
```

推送feature-D分支

```
git push
```

获取最新的远程仓库分支

```
git pull origin feature-D
```

