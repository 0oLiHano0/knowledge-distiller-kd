import tkinter as tk

# 创建主窗口
window = tk.Tk()
window.title("KD Tool - Phase 1") # 设置窗口标题
window.geometry("300x200") # 设置窗口大小

# 创建一个标签控件
label = tk.Label(window, text="Hello, KD Project!")
label.pack(pady=20) # 将标签放置到窗口中，并增加一些垂直边距

# 启动Tkinter事件循环 (这会让窗口一直显示，直到你关闭它)
window.mainloop()