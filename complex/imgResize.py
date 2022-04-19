import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image


root = tk.Tk()
root.withdraw()

Folderpath = filedialog.askdirectory()  #获得选择好的文件夹
print(Folderpath)
filenames = os.listdir(Folderpath)
for filename in filenames:
    image = Image.open(Folderpath + '/' + filename)
    image = image.resize((384, 512))
    image.save(r"G:\dataset\2022.4.9\DownSample\turb2" + '/' + filename)
