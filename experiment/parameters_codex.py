# Code = font + size + color

fonts = {
    "0": "Arial",
    "1": "Arial bold",
    "2": "MS Sans Serif",
    "3": "MS Sans Serif bold",
    "4": "Let's Go Digital"
    }

# mm sizes for PC-LEN-E580
sizes = {
    "0": "Big", # 6 mm
    "1": "Medium", # 4 m
    "2": "Small" # 2 mm
    }

colors = {
    "0": "black on white",
    "1": "white on black",
    "2": "black on red",
    "3": "black on green"
    }

def repr(image_code):

    font = fonts[image_code[0]]
    size = sizes[image_code[1]]
    color = colors[image_code[2]]

    print(
        f"The code {image_code} represents a number with the following attributes:\n",
        f"Font is {font}\n",
        f"Size is {size}\n",
        f"Font color is {color} background\n"
        )

