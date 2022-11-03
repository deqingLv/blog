---
title: "Make blog with hugo and github-pages"
date: 2020-06-24T12:27:43+08:00

keywords: ["Hugo", "Blog"]
summary: "基于GitHub Pages + Hugo搭建的个人博客，给大家介绍下搭建过程"

---

基于GitHub Pages + Hugo搭建的个人博客，给大家介绍下搭建过程。


# 序言

博客还是一个开发者很好的名片，不仅可以记录点滴，还能让更多人认识你了解你，为你赢来潜在的机会。
开发者另个好名片就是代码了，以GitHub上开源的代码项目最为著名，可以主导或者参与到优秀的项目中，提升知名度。
为了和代码一起，我选择GitHub-pages来搭建个人博客，由于现在钟爱Golang，所以作为Hugo作为博客搭建的工具。
下面记录下基于Hugo和GitHub-pages搭建个人博客的方法


# 前提
1. GitHub账号： [点此创建](https://github.com/join)
2. Hugo安装: 安装方法：[macOS](https://gohugo.io/getting-started/installing#macos) , [Windows](https://gohugo.io/getting-started/installing#windows) , [Linux](https://gohugo.io/getting-started/installing#linux).

因为我使用的是macOS，所以下面的操作环境都在mac下。
mac 执行安装Hugo
```
brew install hugo
```

# 利用Hugo建博客

## 新增站点

选择一个存在站点的目录，我的是在 /Users/Archer/Personal/Github 目录下

在终端（terminal）上操作，cd到存放站点的目录，新建站点，我的站点名就叫blog。


```
cd /Users/Archer/Personal/Github

hugo new site blog

Congratulations! Your new Hugo site is created in /Users/Archer/Personal/Github/blog.

Just a few more steps and you're ready to go:

1. Download a theme into the same-named folder.
   Choose a theme from https://themes.gohugo.io/ or
   create your own with the "hugo new theme <THEMENAME>" command.
2. Perhaps you want to add some content. You can add single files
   with "hugo new <SECTIONNAME>/<FILENAME>.<FORMAT>".
3. Start the built-in live server via "hugo server".

Visit https://gohugo.io/ for quickstart guide and full documentation.

```

cd 到新生成的目录下，通过git init初始化

```
cd blog && git init
```

## 安装主题

从[主题市场](https://themes.gohugo.io/tags/portfolio/)选择一个喜欢的主题，快速搭建好看的站点吧。

不同主题的使用方法也不同，我选择了一个GitHub风格的主题：[github-style](https://themes.gohugo.io/github-style/) ,感谢[MeiK](https://github.com/MeiK2333)的创作

![](https://d33wubrfki0l68.cloudfront.net/9fa6ba1ad22946393b69d2ddb0c7ca58d43409b5/b21f8/github-style/screenshot-github-style_hu8fd5979415d309ffffa537b9d823e1c2_209006_750x500_fill_catmullrom_top_2.png)

在站点目录下

```
# git submodule add <LINK_TO_THEME_REPO> themes/<THEME_NAME>. 库下载地址必须是https

git submodule add https://github.com/MeiK2333/github-style.git themes/github-style

# echo 'theme = "<THEME_NAME>"' >> config.toml

echo 'theme = "github-style"' >> config.toml


```
config.toml像这样

```
cat config.toml

baseURL = "http://example.org/"
languageCode = "en-us"
title = "My New Hugo Site"
theme = "github-style"
```



## 测试站点

通过hugo启动本地服务
```
hugo server
```

在浏览器访问 [http://localhost:1313](http://localhost:1313)

至此本地的个人博客站点就完成了。


可以通过
```
hugo new posts/first-post.md
```
发布你的第一篇博客



# 利用GitHub Pages提供免费的网页服务

## 创建博客用的代码库
首先在GitHub建立一个空的公开库，由于之前已经在博客目录做了git init，所以可以将博客目录内容推送到GitHub仓库上

```
git remote add origin git@github.com:deqingLv/blog.git
git add .
git commit -m 'init blog'
git push --set-upstream origin master

```
这样一来，博客的所有内容就都在GitHub库的master分支上了。

## 开启GitHub Pages服务

在GitHub的库页面上通过 **Settings > Options > Github Pages** 

![](https://res.cloudinary.com/practicaldev/image/fetch/s--hobhtfbi--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://thepracticaldev.s3.amazonaws.com/i/ke0v4am83j8j5fwc2qtx.png)

由于GitHub Pages默认是基于[Jekyll](https://jekyllrb.com/)来建站的，而Hugo并不是，GitHub Pages也可以支持纯静态网站资源来提供服务。


**划重点**：在GitHub Pages的源分支选择"master branches /docs folder", 只使用/docs目录来提供网页服务。


Hugo可以通过`hogo`命令进行发布构建，发布结果可指定/docs目录。
修改config.toml

```
echo 'publishDir = "docs"' >> config.toml

cat config.toml

baseURL = "http://example.org/"
languageCode = "en-us"
title = "My New Hugo Site"
theme = "github-style"
publishDir = "docs"

```

执行`hugo` 构建， 得到/docs目录，再把/docs所有内容进行提交

```
hugo

git add .
git commit -m 'init docs'
git push

```

GitHub仓库收到提交会自动构建Pages服务。在**Code**可以找到**Environments**

![](https://res.cloudinary.com/practicaldev/image/fetch/s--r-vTxGC4--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://thepracticaldev.s3.amazonaws.com/i/rpegy86d92hwm0b9ogj4.png)

通过View deployment就能访问个人博客了。 这个地址和域名还可以在GitHub Pages设置里进行修改。

## 博客发布
上面配置完后，博客的写作和发布过程如下

```
# 创建
hugo new posts/xxxx.md

# 写博客
vim xx/xx/xxxx.md

# 本地测试
hugo server

# 构建
hugo

# 提交到GitHub
git add .
git commit -m 'xxx'
git push

# 浏览GitHub Pages，查看发布效果


```


下面是我的个人博客: [https://deqinglv.github.io/blog](https://deqinglv.github.io/blog/)

欢迎勾搭









