# wearglasses
3dface-wearglaesses

git init
4.1 输入git init，如下图所示，这个意思是在当前项目的目录中生成本地的git管理（会发现在当前目录下多了一个.git文件夹，这个文件为隐藏文件）
这里写图片描述

4.2 输入git add .，这个是将项目上所有的文件添加到仓库中的意思，如果想添加某个特定的文件，只需把.换
成这个特定的文件名即可。

4.3 输入git commit -m 'first commit'，表示你对这次提交的注释，双引号里面的内容可以根据个人的需要
改。

4.4 输入git remote add origin https://github.com/sdoften/wearglasses（上面有说到） 将本地的仓库关联到github上，如果执行git remote add origin https://github.com/EndaLi/First_Test.git，
出现错误：　
fatal: remote origin already exists

则执行以下语句：

git remote rm origin

再次执行
git remote add origin https://github.com/sdoften/wearglasses
即可。

4.5 输入git push -u origin master，这是把代码上传到github仓库的意思。执行完后，如果没有异常，会等待几秒，然后跳出一个让你输入Username和Password的窗口，你只要输入github的登录账号和密码就行了。

账号密码都正确的话，会看到下面这么一个东西，进度还会跳，这个是上传过程中的进度，这个过程可能有点慢，有时候得等个10几分钟，这时候去github上面看，还是什么都没有的，所以大兄弟别着急，先做点其他的事，晚点再来看看。