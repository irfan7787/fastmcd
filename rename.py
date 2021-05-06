import cv2
import os

path = "results/PTZ/continuousPan/"
file_list = sorted(os.listdir(path))


for filename in file_list:

    newName = filename.replace("in", "bin")
    print(newName)

    os.rename(path + filename, path + newName)
