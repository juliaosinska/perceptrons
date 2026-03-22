import tkinter as tk
import pickle
import numpy as np
from classifier import classify, classify_detailed
from config import WIDTH, HEIGHT, CELL_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, MODEL_FILE

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Digit Classifier")
        self.root.configure(bg='#f0f0f0')
        
        # title
        title_label = tk.Label(root, text="Digit Classifier", font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # main content frame
        content_frame = tk.Frame(root, bg='#f0f0f0')
        content_frame.pack(pady=10, fill=tk.NONE, expand=False)
        
        # left side: canvas for drawing
        self.canvas = tk.Canvas(content_frame, width=WIDTH*CELL_SIZE, height=HEIGHT*CELL_SIZE, bg='white', relief='sunken', bd=2)
        self.canvas.pack(side=tk.LEFT, padx=20, pady=10)
        
        # right side: controls and result
        right_frame = tk.Frame(content_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.reset_button = tk.Button(right_frame, text="Reset Grid", font=("Arial", 12), bg="#7D2BCA", fg='white', relief='raised', command=self.reset_grid)
        self.reset_button.pack(pady=15)
        
        self.result_label = tk.Label(right_frame, text="Draw a digit and see prediction", font=("Arial", 14), bg='#f0f0f0', fg='#333', width=25, anchor='w')
        self.result_label.pack(pady=15)
        
        # initialize the grid and rectangles
        self.grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.rects = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.toggle_pixel)
    
    # draw the grid based on the current state of self.grid
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
    
    # toggle the pixel state and update the grid and classification
    def toggle_pixel(self, event):
        j = event.x // CELL_SIZE
        i = event.y // CELL_SIZE

        if 0 <= i < HEIGHT and 0 <= j < WIDTH:
            self.grid[i][j] = 1 - self.grid[i][j]
            self.draw_grid()
            self.classify_digit()
    
    # reset the grid to all zeros and update the canvas and result label
    def reset_grid(self):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.grid[i][j] = 0
                self.canvas.itemconfig(self.rects[i][j], fill='white')

        self.result_label.config(text="Draw a digit and see prediction")
    
    # classify the current grid state and update the result label with the predicted digit
    def classify_digit(self):
        vector = np.array([self.grid[i][j] for i in range(HEIGHT) for j in range(WIDTH)])
        predictions, scores, best = classify_detailed(perceptrons, vector)

        if predictions:
            votes_text = f"Votes: {', '.join(map(str, predictions))}"
        else:
            votes_text = "No votes"
            
        self.result_label.config(text=f"Predicted: {best} | {votes_text}")

# load trained model
try:
    with open(MODEL_FILE, 'rb') as f:
        perceptrons = pickle.load(f)
except FileNotFoundError:
    print(f"Error: {MODEL_FILE} not found. Run train.py first.")
    perceptrons = None

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    root.resizable(False, False)
    root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)
    root.maxsize(WINDOW_WIDTH, WINDOW_HEIGHT)
    
    if perceptrons is not None:
        app = DigitDrawer(root)
        root.mainloop()
    else:
        print("Cannot start GUI: model not loaded")