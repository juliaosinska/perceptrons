import tkinter as tk
import pickle
import numpy as np
from classifier import classify

# Załaduj perceptrony
with open('perceptrons.pkl', 'rb') as f:
    perceptrons = pickle.load(f)

WIDTH = 5
HEIGHT = 7
CELL_SIZE = 50

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Digit Classifier")
        self.root.configure(bg='#f0f0f0')  # Jasne tło
        
        # Tytuł
        title_label = tk.Label(root, text="Digit Classifier", font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Kontener dla canvas i prawej strony
        content_frame = tk.Frame(root, bg='#f0f0f0')
        content_frame.pack(pady=10, fill=tk.NONE, expand=False)
        
        # Lewa strona: canvas
        self.canvas = tk.Canvas(content_frame, width=WIDTH*CELL_SIZE, height=HEIGHT*CELL_SIZE, bg='white', relief='sunken', bd=2)
        self.canvas.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Prawa strona: frame dla button i label
        right_frame = tk.Frame(content_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.reset_button = tk.Button(right_frame, text="Reset Grid", font=("Arial", 12), bg="#7D2BCA", fg='white', relief='raised', command=self.reset_grid)
        self.reset_button.pack(pady=15)
        
        self.result_label = tk.Label(right_frame, text="Draw a digit and see prediction", font=("Arial", 14), bg='#f0f0f0', fg='#333', width=25, anchor='w')
        self.result_label.pack(pady=15)
        
        self.grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.rects = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]  # Przechowuj ID prostokątów
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.toggle_pixel)
    
    def draw_grid(self):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                color = 'black' if self.grid[i][j] else 'white'
                if self.rects[i][j] is None:
                    self.rects[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')
                else:
                    self.canvas.itemconfig(self.rects[i][j], fill=color)
    
    def toggle_pixel(self, event):
        j = event.x // CELL_SIZE
        i = event.y // CELL_SIZE
        if 0 <= i < HEIGHT and 0 <= j < WIDTH:
            self.grid[i][j] = 1 - self.grid[i][j]
            self.draw_grid()
            self.classify_digit()  # Dynamiczna klasyfikacja po każdej zmianie
    
    def reset_grid(self):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.grid[i][j] = 0
                self.canvas.itemconfig(self.rects[i][j], fill='white')  # Bez migania
        self.result_label.config(text="Draw a digit and see prediction")  # Reset wyniku
    
    def classify_digit(self):
        vector = np.array([self.grid[i][j] for i in range(HEIGHT) for j in range(WIDTH)])
        prediction = classify(perceptrons, vector)
        self.result_label.config(text=f"Predicted digit: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("700x450")  # Większe okno dla lepszego widoku
    root.resizable(False, False)  # Zablokuj zmianę rozmiaru
    root.minsize(700, 450)  # Minimalny rozmiar
    root.maxsize(700, 450)  # Maksymalny rozmiar
    app = DigitDrawer(root)
    root.mainloop()