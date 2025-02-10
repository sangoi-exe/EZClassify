import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import ctypes


class YoloLabelEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLO Label Editor")
        self.image_list = []
        self.current_index = -1
        self.folder = ""
        self.original_image = None  # Original PIL image
        self.photo = None  # Tkinter formatted image
        self.resize_after_id = None  # Debounce ID for resizing
        self.class_font_size = 20  # Font size for class display
        self.default_button_bg = None  # To store default button background

        # Button mapping: (class name, class number as string)
        self.button_mappings = [
            ("CNH aberta", "0"),
            ("CNH frente", "1"),
            ("CNH verso", "2"),
            ("RG aberto", "3"),
            ("RG frente", "4"),
            ("RG verso", "5"),
            ("Titulo Aberto", "11"),
            ("Titulo Frente", "12"),
            ("Titulo Verso", "13"),
            ("CPF frente", "6"),
            ("CPF verso", "7"),
            ("Cert Nasc", "10"),
            ("CIC frente", "8"),
            ("CIC verso", "9"),
        ]
        # Button style configuration
        self.btn_width = 20  # in text units
        self.btn_height = 2
        self.btn_font = ("Arial", 14)
        self.button_widgets = []  # Dynamic button widgets

        self.setup_ui()

    def center_window(self, width, height):
        try:
            user32 = ctypes.windll.user32
            total_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            total_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        except Exception:
            total_width = self.master.winfo_screenwidth()
            total_height = self.master.winfo_screenheight()
        x = int((total_width - width) / 2)
        y = int((total_height - height) / 2)
        self.master.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        self.center_window(800, 600)
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Top frame: folder selection and filename display
        top_frame = tk.Frame(self.master)
        top_frame.grid(row=0, column=0, sticky="ew")
        btn_select = tk.Button(top_frame, text="Selecionar Pasta", command=self.select_folder)
        btn_select.pack(side=tk.LEFT, padx=5, pady=5)
        self.text_label = tk.Label(top_frame, text="", font=("Arial", 12))
        self.text_label.pack(side=tk.LEFT, padx=5)

        # Main frame: image and image list
        main_frame = tk.Frame(self.master)
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Left: image display frame
        self.image_frame = tk.Frame(main_frame, bg="black")
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_frame.bind("<Configure>", self.on_image_frame_configure)
        self.image_label = tk.Label(self.image_frame, text="", bg="black", fg="white", font=("Arial", 32))
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right: listbox with scrollbar for images
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="ns")
        scrollbar = tk.Scrollbar(right_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(right_frame, yscrollcommand=scrollbar.set)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # Updated class display frame: centered class name (no font control widget)
        self.class_frame = tk.Frame(self.master)
        self.class_frame.grid(row=2, column=0, sticky="ew")
        self.class_label = tk.Label(self.class_frame, text="Class: Not set", font=("Arial", self.class_font_size), anchor="center")
        self.class_label.pack(fill=tk.X, padx=5, pady=5)

        # Bottom frame: dynamic buttons for label mapping
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.grid(row=3, column=0, sticky="ew")
        self.bottom_frame.bind("<Configure>", self.arrange_buttons_layout)
        self.create_label_buttons()

        # Key binding for navigation
        self.master.bind("<Key>", self.on_key_press)

    def create_label_buttons(self):
        # Clear existing buttons
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()
        self.button_widgets = []
        for label, value in self.button_mappings:
            btn = tk.Button(
                self.bottom_frame,
                text=label,
                command=lambda v=value: self.update_label_class(v),
                width=self.btn_width,
                height=self.btn_height,
                font=self.btn_font,
            )
            # Save default background color from first button created
            if self.default_button_bg is None:
                self.default_button_bg = btn.cget("bg")
            self.button_widgets.append(btn)
        self.arrange_buttons_layout()

    def arrange_buttons_layout(self, event=None):
        # Remove current grid placement
        for btn in self.button_widgets:
            btn.grid_forget()
        self.bottom_frame.update_idletasks()
        frame_width = self.bottom_frame.winfo_width()
        button_req_width = self.button_widgets[0].winfo_reqwidth() if self.button_widgets else 100
        pad_x = 10
        max_columns = max(1, frame_width // (button_req_width + pad_x))
        # Position buttons in grid layout
        for index, btn in enumerate(self.button_widgets):
            row = index // max_columns
            col = index % max_columns
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        for col in range(max_columns):
            self.bottom_frame.columnconfigure(col, weight=1)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder = folder
            self.load_images_list()

    def load_images_list(self):
        self.image_list = []
        self.listbox.delete(0, tk.END)
        valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        for filename in os.listdir(self.folder):
            if os.path.splitext(filename)[1].lower() in valid_ext:
                fullpath = os.path.join(self.folder, filename)
                self.image_list.append(fullpath)
                self.listbox.insert(tk.END, filename)
        if self.image_list:
            self.current_index = 0
            self.display_current_image()
        else:
            self.current_index = -1

    def display_current_image(self):
        if 0 <= self.current_index < len(self.image_list):
            filepath = self.image_list[self.current_index]
            self.load_and_show_image(filepath)
            self.listbox.select_clear(0, tk.END)
            self.listbox.select_set(self.current_index)
            self.listbox.see(self.current_index)
            filename = os.path.basename(filepath)
            self.text_label.config(text=filename)
            self.update_class_display()  # Refresh the class display

    def load_and_show_image(self, filepath):
        try:
            self.original_image = Image.open(filepath)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir imagem:\n{e}")
            self.original_image = None
            return
        self.update_image_display()

    def update_image_display(self):
        if self.original_image:
            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height()
            if frame_width > 1 and frame_height > 1:
                img_copy = self.original_image.copy()
                # Resize while maintaining aspect ratio
                img_copy.thumbnail((frame_width, frame_height), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(img_copy)
                self.image_label.config(image=self.photo)

    def on_image_frame_configure(self, event):
        if self.resize_after_id:
            self.image_frame.after_cancel(self.resize_after_id)
        self.resize_after_id = self.image_frame.after(100, self.update_image_display)

    def on_listbox_select(self, event):
        selection = event.widget.curselection()
        if selection:
            self.current_index = selection[0]
            self.display_current_image()

    def on_key_press(self, event):
        # Left/up (or w/a) go back; right/down (or s/d) advance
        if event.keysym in ("Left", "Up") or event.char.lower() in ("w", "a"):
            self.prev_image()
        elif event.keysym in ("Right", "Down") or event.char.lower() in ("s", "d"):
            self.next_image()

    def prev_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.display_current_image()

    def update_label_class(self, new_class):
        if self.current_index < 0 or not self.image_list:
            return
        image_path = self.image_list[self.current_index]
        base, _ = os.path.splitext(image_path)
        label_path = base + ".txt"
        if not os.path.exists(label_path):
            messagebox.showerror("Erro", "Arquivo TXT não encontrado.")
            return
        try:
            with open(label_path, "r") as file:
                content = file.read().strip()
            if content:
                parts = content.split(maxsplit=1)
                remainder = parts[1] if len(parts) > 1 else ""
                new_content = new_class + (" " + remainder if remainder else "")
            else:
                new_content = new_class
            os.makedirs("train/images", exist_ok=True)
            os.makedirs("train/labels", exist_ok=True)
            dest_image = os.path.join("train/images", os.path.basename(image_path))
            shutil.copy(image_path, dest_image)
            dest_label = os.path.join("train/labels", os.path.basename(label_path))
            with open(dest_label, "w") as file:
                file.write(new_content)
            self.show_tooltip("Arquivo salvo com sucesso!")
            self.update_class_display()  # Refresh class display after update
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar:\n{e}")

    def update_class_display(self):
        """
        Update the class label based on the current image's TXT file.
        Also highlights the corresponding button with a pinkish red background.
        """
        if self.current_index < 0 or not self.image_list:
            self.class_label.config(text="Class: Not set")
            for btn in self.button_widgets:
                btn.config(bg=self.default_button_bg)
            return
        image_path = self.image_list[self.current_index]
        base, _ = os.path.splitext(image_path)
        label_path = base + ".txt"
        if not os.path.exists(label_path):
            self.class_label.config(text="Class: Not set")
            for btn in self.button_widgets:
                btn.config(bg=self.default_button_bg)
            return
        try:
            with open(label_path, "r") as file:
                content = file.read().strip()
            if content:
                parts = content.split(maxsplit=1)
                class_number = parts[0]
                class_name = "Unknown"
                for name, num in self.button_mappings:
                    if num == class_number:
                        class_name = name
                        break
                self.class_label.config(text=f"Class: {class_name}")
            else:
                self.class_label.config(text="Class: Not set")
                class_number = None
        except Exception:
            self.class_label.config(text="Class: Error")
            class_number = None

        # Highlight the button corresponding to the current class
        highlight_color = "#FF9999"  # Pinkish red
        for i, (name, num) in enumerate(self.button_mappings):
            if class_number is not None and num == class_number:
                self.button_widgets[i].config(bg=highlight_color)
            else:
                self.button_widgets[i].config(bg=self.default_button_bg)

    def show_tooltip(self, message, duration=2000):
        tip = tk.Toplevel(self.master)
        tip.wm_overrideredirect(True)
        x = self.master.winfo_x() + self.master.winfo_width() - 200
        y = self.master.winfo_y() + self.master.winfo_height() - 100
        tip.geometry(f"180x30+{x}+{y}")
        lbl = tk.Label(tip, text=message, bg="yellow", relief="solid", borderwidth=1)
        lbl.pack(fill=tk.BOTH, expand=True)
        tip.after(duration, tip.destroy)


def main():
    root = tk.Tk()
    app = YoloLabelEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
