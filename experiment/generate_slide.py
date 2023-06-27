import parameters_codex as param
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd
import copy


def generate_number_image(number, font_path, font_size, background_color, text_color):
    
    # Create a new image
    image_size = (600, 700)
    image = Image.new('RGB', image_size)
    
    # Fill the top 200 pixels with the specified background color for the number
    top_background = Image.new('RGB', (image_size[0], 200), background_color)
    image.paste(top_background, (0, 0))
    
    # Fill the bottom 500 pixels with white background for the Charuco board
    bottom_background = Image.new('RGB', (image_size[0], 500), (255, 255, 255))
    image.paste(bottom_background, (0, 200))
    
    # Load a font
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate the position to center the number
    draw = ImageDraw.Draw(image)
    text = str(number)
    _, _, w, h  = draw.textbbox((0, 0), text, font=font)
    x = (image_size[0] - w) // 2
    y = 100 - h // 2
    
    # Draw the number on the image
    draw.text((x, y), str(number), font=font, fill=text_color)

    # Load the charuco board image
    charuco_board_path = "aruco_patterns/charuco_boards/charuco_4x4_DICT_4X4_1000_sl20_ml14.png"
    charuco_board = Image.open(charuco_board_path)
    charuco_board = charuco_board.resize((400, 400))
    
    # Calculate the position to place the charuco board
    board_width, board_height = charuco_board.size
    board_x = (image_size[0] - board_width) // 2
    board_y = 450 - board_height // 2
    
    # Paste the charuco board at the bottom and centered
    image.paste(charuco_board, (board_x, board_y))
    
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
                    font_size = 45
                elif size == "1":
                    font_size = 30
                elif size == "2":
                    font_size = 15

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