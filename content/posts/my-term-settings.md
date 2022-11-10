iTerm2 + zsh + oh-my-zsh The Most Power Full Terminal on my mac.

# Install



```
# Install iterm2  or go to download page: https://www.iterm2.com/downloads.html
brew install --cask iterm2

# Install zsh
brew install zsh

# Install oh-my-zsh

sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"


```

# Decorate Our iTerm2 

## Getting rid of the title bar 


Go to iTerm2 > Appearance > General > Theme: Dark



## iTerm2 theme

Go to iTerm2 > Preferences > Profiles > Colors Tab 

Get the iTerm color settings:

- [https://github.com/topics/iterm2-theme](https://github.com/topics/iterm2-theme)

- [https://iterm2colorschemes.com](https://iterm2colorschemes.com)

[Snazzy.itermcolors](https://github.com/sindresorhus/iterm2-snazzy/raw/main/Snazzy.itermcolors) looks goot to me



## iTerm2 BackgroundIamge

Go to iTerm2 -> Preferences -> Profiles -> window -> Background Image.
Choose your favorite wallpaper and set Blending.


## Zsh Theme: Powerlevel10k


### Install [Powerlevel10k](https://github.com/romkatv/powerlevel10k)

```
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k
echo 'source ~/powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc

```

Prompt Style use 'Pure'


## Hide Username & Hostname

```
echo 'prompt_context() {}' >> ~/.zshrc
```


# Manage Plugins



## zsh-syntax-highlighting


1. Clone this repository in oh-my-zsh's plugins directory:
	
	```
	git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
	```
2. Activate the plugin in ~/.zshrc:

	```
	plugins=( [plugins...] zsh-syntax-highlighting)
	```
	Note that zsh-syntax-highlighting must be the last plugin sourced.
	
3. Restart zsh (such as by opening a new instance of your terminal emulator).

	```
	exec zsh
	```



## zsh-autosuggestions


1. Clone this repository in oh-my-zsh's plugins directory:

	```
	git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
	
	```

2. Activate the plugin in ~/.zshrc:

	```
	plugins=( 
	    # other plugins...
	    zsh-autosuggestions
	)
	```

3. Restart zsh (such as by opening a new instance of your terminal emulator).
	
	```
	exec zsh
	```

## autojump


```
brew install autojump

```


# Common Tips and Shortcuts

Tips:

- Selection is copied
- command + d :Vertical split screen
- command + shift + d :Horizontal split screen
- command + shift + h :Open the Clipboard (Copy History)

# Alias
My alias:

```
# navigation aliases
alias dev="cd ~/dev/"
alias personal="cd ~/persional"
alias work="cd ~/work"

# kubectl aliases
alias k="kubectl"
alias kt="sudo ktctl"

# docker aliases
alias d="docker"

```