---
title: "Git awesome tips"
date: 2025-12-15T22:24:00+08:00

keywords: ["git", "tips"]
summary: "some awesome tips of git"

---

Git awesome tips

# vi ~/.gitconfig


add alias
```
[alias]
    st =  status
	br = branch
	co = checkout
	cm = commit -m 
    aa = add .
	amend = commit --amend --reuse-message=HEAD # 合并当前缓冲内容到上一次的提交并复用提交信息
	uncommit = reset --soft HEAD~1 # 撤销上次提交
	lg = log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
    ll = log --oneline
	last = log -1 HEAD --stat
	se = !git rev-list --all | xargs git grep -F
```

