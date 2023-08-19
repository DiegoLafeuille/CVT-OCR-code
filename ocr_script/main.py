import tkinter as tk
from ocr_gui import OCR_GUI


def main():

    root = tk.Tk()
    root.state('zoomed') 
    root.bind('<Escape>', lambda e: root.quit())
    gui = OCR_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()