from os import walk

from PIL import Image


def resize_it(path):
    img = Image.open(path)
    img.resize((28, 28)).convert('L').save(path)


for i in list(walk('datasets2/images/triangle'))[0][2]:
    resize_it(f'datasets2/images/triangle/{i}')