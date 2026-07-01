# ghostty 极客推荐配置

# ==============================
# 字体与排版：打造呼吸感
# ==============================
# 强烈推荐深受开发者喜爱的 JetBrainsMono 字体
font-family = "JetBrainsMono-Regular"
font-style-bold = "Medium"
font-style-bold-italic = "Medium Italic"
font-size = 16

# 字体加粗渲染，让代码在高清屏上看起来更饱满扎实
font-thicken = true
grapheme-width-method = "unicode"
adjust-cell-width = 0%

# 减少水平边距，节省屏幕空间；适当增加垂直边距，不至于太拥挤
window-padding-x = 15
window-padding-y = 15
# 永远保存窗口状态（记住你上次关闭时的窗口大小和位置）
window-save-state = "always"

# ==============================
# 沉浸式 UI：透明与高斯模糊
# ==============================
# 背景不透明度设为 80%，半透效果最高级
background-opacity = 0.8
# 背景模糊半径，配合透明度产生绝美的磨砂玻璃 (亚克力) 质感
background-blur-radius = 25

# ==============================
# 实用细节：光标与内存优化
# ==============================
# 使用醒目的黄色作为光标颜色，更容易在密集的日志中找到位置
cursor-color = e5c07b
cursor-style = "bar"
# 开启光标闪烁
cursor-style-blink = true

# 减少滚动历史记录（保留 2w 行），为追求极致性能节省内存
scrollback-limit = 20000

# ==============================
# 配色方案：护眼 OneDark 优化版
# ==============================
# 稍微加深背景色以增加代码高亮对比度
background = 1e222a
# 提高前景色亮度，文本更清晰易读
foreground = dcdfe4

# 调整选择区域的前后景色反转，提高复制时的辨识度
selection-background = 405060
selection-foreground = dcdfe4
selection-invert-fg-bg = true

# ANSI 16 色调色板映射（定制红绿黄蓝紫色系，长久看代码不伤眼）
palette = 0=#1e222a
palette = 1=#e06c75
palette = 2=#98c379
palette = 3=#e5c07b
palette = 4=#61afef
palette = 5=#c678dd
palette = 6=#56b6c2
palette = 7=#dcdfe4
palette = 8=#545862
palette = 9=#e06c75
palette = 10=#98c379
palette = 11=#e5c07b
palette = 12=#61afef
palette = 13=#c678dd
palette = 14=#56b6c2
palette = 15=#ffffff

# ==============================
# 快捷键：全局 Quake 模式呼出
# ==============================
# 这是灵魂操作！按下 Alt + ` 键（波浪号键）立刻像下拉菜单一样呼出/隐藏终端
keybind = global:alt+grave_accent=toggle_quick_terminal
使用如上配置，变成如下界面：

1ea46b0b06cb75ce563017e5945e4b3f_MD5

原始配置：

font-family = "ZedMono NFM Extd"

font-style-bold = "Medium"
font-style-bold-italic = "Medium Italic"
font-size = 13.4
font-thicken = true
grapheme-width-method = "unicode"

adjust-cell-width = -5%
palette = 0=#212733
palette = 1=#f08778
palette = 2=#53bf97
palette = 3=#fdcc60
palette = 4=#60b8d6
palette = 5=#ec7171
palette = 6=#98e6ca
palette = 7=#fafafa
palette = 8=#686868
palette = 9=#f58c7d
palette = 10=#58c49c
palette = 11=#ffd165
palette = 12=#65bddb
palette = 13=#f17676
palette = 14=#9debcf
palette = 15=#ffffff
background = #1f2430
foreground = #cbccc6

selection-invert-fg-bg = true

cursor-style = "bar"
cursor-style-blink = true
scrollback-limit = 100000
window-padding-x = 20
window-padding-y = 2,10
window-save-state = "always"
可选：

1、添加背景图片

• background-image
如：

background-image = "/Users/lca/Pictures/背景图片/GJ75CAxbUAAfEH5.jpeg"
# 控制窗口背景透明度
background-opacity = 0.8
# 控制缩放策略：contain、cover、stretch、none
background-image-fit= cover
2、隐藏标题栏及 mac 的红黄蓝按扭

macos-titlebar-style = hidden
