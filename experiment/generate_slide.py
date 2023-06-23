import parameters_codex as param
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd
import copy



def generate_number_image(number, font_path, font_size, background_color, text_color):
    
    # Create a new image with the specified background color
    image_size = (500, 300)
    image = Image.new('RGB', image_size, background_color)
    
    # Load a font
    font = ImageFont.truetype(font_path, font_size)
    
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
    digits = list(range(10))
        
    for font in param.fonts:
        for size in param.sizes:
            for color in param.colors: 
                    
                # Font parameter
                if font == "0":
                    font_path = "experiment/fonts/arial.ttf"
                elif font == "1":
                    font_path = "experiment/fonts/arialbd.ttf"
                elif font == "2":
                    font_path = "experiment/fonts/micross.ttf"
                elif font == "3":
                    # Verify that MS Reference Sans Serif and MS Sans Serif are the same
                    font_path = "experiment/fonts/microssbd.ttf"  
                elif font == "4":
                    font_path = "experiment/fonts/Let_s_go_Digital_Regular.ttf"

                # Size parameter
                if size == "0":
                    font_size = 70
                elif size == "1":
                    font_size = 50
                elif size == "2":
                    font_size = 30

                # Create a number made up of digits 0 to 9 in a random number
                shuffled_digits = copy.copy(digits)
                random.shuffle(shuffled_digits)

                # Insert decimal point at the randomly chosen index
                decimal_index = random.randint(1, len(shuffled_digits) - 1)
                shuffled_digits.insert(decimal_index, '.')
                number = ''.join(str(num) for num in shuffled_digits)

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

                image = generate_number_image(number, font_path, font_size, background_color, text_color)
                img_code = font + size + color
                
                image.save("experiment/slides/" + img_code + ".png")
                ground_truths.append((img_code, number))



    print(f"Total number of slides is {len(ground_truths)}")

    # Create a DataFrame from the list
    df = pd.DataFrame(ground_truths, columns=['Code', 'Ground truth'])

    # Save the DataFrame to a CSV file
    df.to_csv('experiment/ground_truths.csv', index=False)
    print("Ground truths saved")


if __name__ == '__main__':
    main()