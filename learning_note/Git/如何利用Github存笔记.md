# 如何利用Github存笔记

参考链接[[全栈算法工程师\] 如何把github当做个人笔记记录站点，git与github（本地与远端，local/remote(origin)）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1uG4y1a7gz/?spm_id_from=333.337.search-card.all.click&vd_source=bf7b9535de982f1d288138463991a3f7)



创建一个新的仓库

![image-20240507104455207](.\如何利用Github存笔记.assets\image-20240507104455207.png)



Pycharm 克隆下来

![image-20240507104416171](.\如何利用Github存笔记.assets\image-20240507104416171.png)





如果提示没有权限克隆,检查本地ssh key是否存在,通常在C:\Users\Yoow\.ssh文件中，看是否有id_rsa.pub，没有就

```
ssh-keygen -t rsa -b 2048 -C "你自己的邮箱地址"
```

创建一个

然后去增加SSH keys

![image-20240507105153285](.\如何利用Github存笔记.assets\image-20240507105153285.png)



接着打开终端按照给出的提示操作就好

```
echo "# daily_learning_notes" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Yoow4/daily_learning_notes.git
git push -u origin main
```



修改操作，三连

```
git add （改成自己的文件）
git commit -m "（修改备注）"
git push -u origin main（提交）
```

