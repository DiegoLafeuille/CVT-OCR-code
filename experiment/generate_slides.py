import parameters_codex as param
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd
import copy


def generate_1_number_image(number, font_path, font_size, stroke_size, background_color, text_color):
    
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
    draw.text((x, y), str(number), font=font, fill=text_color, stroke_width=stroke_size)

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

def generate_3_numbers_image(numbers, font_path, background_color, text_color):
    
    # Create a new image
    image_size = (600, 700)
    image = Image.new('RGB', image_size)
    
    # Fill the top 200 pixels with the specified background color for the number
    top_background = Image.new('RGB', (image_size[0], 200), background_color)
    image.paste(top_background, (0, 0))
    
    # Fill the bottom 500 pixels with white background for the Charuco board
    bottom_background = Image.new('RGB', (image_size[0], 500), (255, 255, 255))
    image.paste(bottom_background, (0, 200))

    # Size parameter (sizes for second screen)
    size_big = 25 # 5.5mm
    size_medium = 18 # 4mm
    size_small = 12 # 2.5mm

    # # Size parameter (sizes for pc)
    # size_big = 35 # 5.5mm
    # size_medium = 25 # 4mm
    # size_small = 15 # 2.5mm
    
    # Load fonts with different sizes
    fonts = [ImageFont.truetype(font_path, font_size) for font_size in [size_big, size_medium, size_small]]
    
    # Calculate the position to center the number
    draw = ImageDraw.Draw(image)
    texts = [str(number) for number in numbers]
    text_boxes = [draw.textbbox((0, 0), text, font=font) for text, font in zip(texts, fonts)]
    ws = [text_box[2] for text_box in text_boxes]
    hs = [text_box[3] for text_box in text_boxes]

    xs = [(image_size[0] - w) // 2 for w in ws]
    ys = []
    h_delta = [-45, 0, 35]
    for h, delta in zip(hs, h_delta):
        ys.append(100 - h // 2 + delta)
    
    # Draw the number on the image
    for x, y, number, font in zip(xs, ys, numbers, fonts):
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



def read_numbers():

    # Read the existing ground_truths.csv file
    df = pd.read_csv('experiment/ground_truths.csv', dtype=str)

    # Iterate over the DataFrame
    for index, row in df.iterrows():
        
        img_code, number = row['Code'], row['Ground truth']

        # Extract font, size, color from the image code
        font = img_code[0]
        size = img_code[1]
        color = img_code[2]

        # Font parameter
        stroke_size = 0           
        if font == "0":
            font_path = "experiment/fonts/arial.ttf"
        elif font == "1":
            font_path = "experiment/fonts/arialbd.ttf"
        elif font == "2":
            font_path = "experiment/fonts/micross.ttf"
        elif font == "3":
            # font_path = "experiment/fonts/micross.ttf" 
            # stroke_size = 1
            continue
        elif font == "4":
            font_path = "experiment/fonts/Let_s_go_Digital_Regular.ttf"

        # # Size parameter (sizes for second screen)
        # if size == "0":
        #     font_size = 25 # 5.5mm
        # elif size == "1":
        #     font_size = 18 # 4mm
        # elif size == "2":
        #     font_size = 12 # 2.5mm

        # Size parameter (sizes for pc)
        if size == "0":
            font_size = 35 # 5.5mm
        elif size == "1":
            font_size = 25 # 4mm
        elif size == "2":
            font_size = 15 # 2.5mm

        # Set font and background colors
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

        image = generate_1_number_image(number, font_path, font_size, stroke_size, background_color, text_color)
        image.save("experiment/slides/" + img_code + ".png")

    print(f"Total number of slides is {len(df)}")

def create_numbers():

    ground_truths = []
    digits = list(range(10))
        
    for font in param.fonts:
        for size in param.sizes:
            for color in param.colors: 


                # Font parameter
                stroke_size = 0           
                if font == "0":
                    font_path = "experiment/fonts/arial.ttf"
                elif font == "1":
                    font_path = "experiment/fonts/arialbd.ttf"
                elif font == "2":
                    font_path = "experiment/fonts/micross.ttf"
                elif font == "3":
                    # font_path = "experiment/fonts/micross.ttf" 
                    # stroke_size = 1
                    continue
                elif font == "4":
                    font_path = "experiment/fonts/Let_s_go_Digital_Regular.ttf"

                # Size parameter (sizes for second screen)
                if size == "0":
                    font_size = 25 # 5.5mm
                elif size == "1":
                    font_size = 18 # 4mm
                elif size == "2":
                    font_size = 12 # 2.5mm

                # Create a number made up of digits 0 to 9 in a random number
                shuffled_digits = copy.copy(digits)
                random.shuffle(shuffled_digits)

                # Insert decimal point at the randomly chosen index
                decimal_index = random.randint(1, len(shuffled_digits) - 1)
                shuffled_digits.insert(decimal_index, '.')
                number = ''.join(str(num) for num in shuffled_digits)
                
                # Set font and background colors
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

                image = generate_1_number_image(number, font_path, font_size, stroke_size, background_color, text_color)
                img_code = font + size + color
                
                image.save("experiment/slides/" + img_code + ".png")
                ground_truths.append((img_code, number))



    print(f"Total number of slides is {len(ground_truths)}")

    # Create a DataFrame from the list
    df = pd.DataFrame(ground_truths, columns=['Code', 'Ground truth'])

    # Save the DataFrame to a CSV file
    df.to_csv('experiment/ground_truths.csv', index=False)
    print("Ground truths saved")

def create_slides_walkiria():
    
    ground_truths = []
    digits = list(range(10))
        
    for font in param.fonts:
        for size in param.sizes:
            for color in param.colors: 


                # Font parameter
                stroke_size = 0           
                if font == "0":
                    font_path = "experiment/fonts/arial.ttf"
                elif font == "1":
                    font_path = "experiment/fonts/arialbd.ttf"
                elif font == "2":
                    # font_path = "experiment/fonts/micross.ttf"
                    continue
                elif font == "3":
                    # font_path = "experiment/fonts/micross.ttf" 
                    # stroke_size = 1
                    continue
                elif font == "4":
                    # font_path = "experiment/fonts/Let_s_go_Digital_Regular.ttf"
                    continue

                # # Size parameter (sizes for second screen)
                # if size == "0":
                #     font_size = 25 # 5.5mm
                # elif size == "1":
                #     font_size = 18 # 4mm
                # elif size == "2":
                #     font_size = 12 # 2.5mm

                # Size parameter (sizes for pc)
                if size == "0":
                    font_size = 35 # 5.5mm
                elif size == "1":
                    font_size = 25 # 4mm
                elif size == "2":
                    font_size = 15 # 2.5mm

                # Create a number made up of digits 0 to 9 in a random number
                shuffled_digits = copy.copy(digits)
                random.shuffle(shuffled_digits)

                # Insert decimal point at the randomly chosen index
                decimal_index = random.randint(1, len(shuffled_digits) - 1)
                shuffled_digits.insert(decimal_index, '.')
                number = ''.join(str(num) for num in shuffled_digits)
                
                # Set font and background colors
                if color == "0":
                    text_color = (0, 0, 0)
                    background_color = (255, 255, 0)
                elif color == "1":
                    text_color = (255, 255, 0)
                    background_color = (0, 0, 0)
                elif color == "2":
                    text_color = (255, 255, 0)
                    background_color = (0, 0, 255)
                elif color == "3":
                    text_color = (0, 0, 0)
                    background_color = (255, 0, 0)
                image = generate_1_number_image(number, font_path, font_size, stroke_size, background_color, text_color)
                img_code = font + size + color
                
                image.save("experiment/slides_walkiria/" + img_code + ".png")
                ground_truths.append((img_code, number))

    print(f"Total number of slides is {len(ground_truths)}")

    # Create a DataFrame from the list
    df = pd.DataFrame(ground_truths, columns=['Code', 'Ground truth'])

    # Save the DataFrame to a CSV file
    df.to_csv('experiment/ground_truths_walkiria.csv', index=False)
    print("Ground truths saved")

def create_slides_3by3():
    
    # Read the existing ground_truths.csv file
    df = pd.read_csv('experiment/ground_truths.csv', dtype=str)

    counter = 0
    for font in param.fonts:
        for color in param.colors: 

            # Font parameter
            if font == "0":
                font_path = "experiment/fonts/arial.ttf"
            elif font == "1":
                font_path = "experiment/fonts/arialbd.ttf"
            elif font == "2":
                font_path = "experiment/fonts/micross.ttf"
            elif font == "3":
                # font_path = "experiment/fonts/micross.ttf" 
                # stroke_size = 1
                continue
            elif font == "4":
                font_path = "experiment/fonts/Let_s_go_Digital_Regular.ttf"
            
            # Set font and background colors
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

            # Get number from existing ground_truths
            numbers = []
            for size in param.sizes:
                numbers.extend(df.loc[df["Code"] == font + size + color]["Ground truth"].to_list())

            img_code = font + color
            image = generate_3_numbers_image(numbers, font_path, background_color, text_color)
            counter += 1
            image.save("experiment/slides_3_big/" + img_code + ".png")

    print(f"Total number of slides is {counter}")

            

def main():
    
    # mode = "create"
    # mode = "read"
    # mode = "walkiria"
    mode = "3by3"

    if mode == "create":
        create_numbers()
    elif mode == "read":
        read_numbers() 
    elif mode == "walkiria":
        create_slides_walkiria() 
    elif mode == "3by3":
        create_slides_3by3()
    

if __name__ == '__main__':
    main()