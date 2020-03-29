from tkinter import *

from PIL import ImageGrab
import torch

from reco import Net, load_dataset

count_snaps = 0


class Paint(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.canvas = Canvas(self, bg="white")  # Создаем поле для рисования, устанавливаем белый фон
        self.parent = parent
        self.setUI()
        self.brush_size = 6
        self.brush_color = "black"
        self.color = 'black'
        self.canvas.bind("<B1-Motion>", self.draw)
        self.reco_model = load_model()

    def setUI(self):
        self.parent.title("")
        self.pack(fill=BOTH, expand=1)

        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)

        self.canvas.grid(row=2, column=0, columnspan=7,
                         padx=5, pady=5,
                         sticky=E + W + S + N)

        clear_btn = Button(self, text="Clear all", width=10, command=lambda: self.canvas.delete("all"))
        clear_btn.grid(row=0, column=0, sticky=W)

        snap_btn = Button(self, text="Snap", width=10, command=lambda: self.save_canvas())
        snap_btn.grid(row=0, column=1, sticky=W)

        reco_btn = Button(self, text="Reco", width=10, command=lambda: self.recognize())
        reco_btn.grid(row=0, column=2, sticky=W)

    def draw(self, event):
        self.canvas.create_oval(event.x - self.brush_size,
                                event.y - self.brush_size,
                                event.x + self.brush_size,
                                event.y + self.brush_size,
                                fill=self.color, outline=self.color)

    def save_canvas(self, clr=True):
        global count_snaps
        count_snaps += 1
        canvas = self.canvas_coords()
        grabbed_canvas = ImageGrab.grab(bbox=canvas).convert('L').resize((28, 28))
        grabbed_canvas.save(f'to_reco/images/unknown/reco.png')
        if clr:
            self.canvas.delete('all')

    def canvas_coords(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        box = (x, y, x1, y1)
        return box

    def recognize(self):
        answers = {0: 'Круг', 1: 'Квадрат', 2: 'Треугольник'}
        self.save_canvas(clr=False)
        for batch_idx, (data, target) in enumerate(load_dataset(data_path='to_reco/images/', shuffle=False)):
            answer = self.reco_model(data)
            print(answer)
            print(answers[answer.detach().numpy().argmax()])
            canvas_id = self.canvas.create_text(10, 10, anchor="nw")
            self.canvas.itemconfig(canvas_id, text=f'{answers[answer.detach().numpy().argmax()]}\n{answer.detach().numpy()}')


def load_model():
    model = Net()
    model.load_state_dict(torch.load('models/my_model'))
    model.eval()
    return model


def main():
    root = Tk()
    root.geometry("310x336+300+300")
    root.resizable(width=False, height=False)
    app = Paint(root)
    root.mainloop()


if __name__ == "__main__":
    main()
