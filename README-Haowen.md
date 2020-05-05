Deep Learning with Python

$ pwd /Users/ML/deeplearning/dl-with-python

Start Jupyter: 
$ jupyter notebook

Local directory: /Users/ML/deeplearning/dl-with-python 
Github: https://github.com/hanyun2019/deep-learning-with-python.git


Steps：

git init //初始化仓库

# git add .(文件name) //添加文件到本地仓库

git commit -m "first commit" //添加文件描述信息

git remote add origin https://github.com/hanyun2019/deep-learning-with-python.git //链接远程仓库，创建主分支

git pull origin master // 把本地仓库的变化连接到远程仓库主分支

# git push -u origin master //把本地仓库的文件推送到远程仓库

使用强制push的方法： $ git push -u origin master -f

########################################################################

Push an existing repository from the command line

$ git remote add origin https://github.com/hanyun2019/deep-learning-with-python.git

$ git add .

$ git commit -m "update for git commit test"

$ git push origin master