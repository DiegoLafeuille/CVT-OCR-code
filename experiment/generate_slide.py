import parameters_codex as param
from PIL import Image, ImageDraw, ImageFont
import random
import csv
import pandas as pd


def generate_number_image(number, font_size, background_color=(255, 255, 255), text_color=(0, 0, 0), font_path=None):
    
    # Create a new image with the specified background color
    image_size = (500, 300)
    image = Image.new('RGB', image_size, background_color)
    
    # Load a font (change the font_path to the path of the font file on your system)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.truetype("arial.ttf", font_size)
    
    # Calculate the position to center the number
    draw = ImageDraw.Draw(image)
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    
    # Draw the number on the image
    draw.text((x, y), str(number), font=font, fill=text_color)
    
    return image


def main():

    ground_truths = []
        
    # for font in param.fonts:
    #     for size in param.sizes:
    #         for length in param.lengths:
    #             for decimal in param.decimals:
    #                 for color in param.colors:
    #                     for i in range(5):    
    #                         if length == "0":
    #                             number = random.randint(100, 999)
    #                         if decimal == "1":
    #                             number = number / 100
    #                             number = "{:.2f}".format(number)
    #                         print(number)

    #                         if color == "0":
    #                             text_color = (0, 0, 0)
    #                             background_color = (255, 255, 255)
    #                         elif color == "1":
    #                             text_color = (255, 255, 255)
    #                             background_color = (0, 0, 0)
    #                         elif color == "2":
    #                             text_color = (0, 0, 0)
    #                             background_color = (255, 0, 0)
    #                         elif color == "3":
    #                             text_color = (0, 0, 0)
    #                             background_color = (0, 255, 0)

    #                         image = generate_number_image(number, 128, background_color, text_color, font_path=None)


    #                         img_code = font + size + length + decimal + color + str(i)
    #                         image.save("slides/" + img_code + ".png")

    #                         ground_truths.append((img_code, number))


    for length in param.lengths:
        for decimal in param.decimals:
            for color in param.colors:
                for i in range(3):  
                    
                    if length == "0":
                        number = random.randint(100, 999)
                    elif length == "1":
                        number = random.randint(10000, 99999)

                    if decimal == "1":
                        number = number / 100
                        number = "{:.2f}".format(number)
                    print(number)

                    if color == "0":
                        text_color = (0, 0, 0)
                        background_color = (255, 255, 255)
                    elif color == "1":
                        text_color = (255, 255, 255)
                        background_color = (0, 0, 0)
                    elif color == "2":
                        text_color = (0, 0, 0)
                        background_color = (255, 0, 0)
                    elif color == "3":
                        text_color = (0, 0, 0)
                        background_color = (0, 255, 0)

                    image = generate_number_image(number, 70, background_color, text_color, font_path=None)
                    img_code = length + decimal + color + str(i)
                    
                    image.save("experiment/slides/" + img_code + ".png")
                    print("Picture saved")
                    ground_truths.append((img_code, number))

    print(f"Total number of slides is {len(ground_truths)}")

    # Create a DataFrame from the list
    df = pd.DataFrame(ground_truths, columns=['Code', 'Ground truth'])

    # Save the DataFrame to a CSV file
    df.to_csv('experiment/ground_truths.csv', index=False)
    print("Ground truths saved")




if __name__ == '__main__':
    main()