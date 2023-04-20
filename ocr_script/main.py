import tkinter as tk
from ocr_gui import OCR_GUI


def main():
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    root.geometry = ("1080x720")
    root.config(bg="skyblue")
    gui = OCR_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()