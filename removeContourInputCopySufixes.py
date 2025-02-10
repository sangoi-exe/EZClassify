import os
import tkinter as tk
from tkinter import filedialog


def main():
    # Initialize Tkinter root and hide the main window
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Ensure the file dialog is on top

    # Prompt the user to select a directory
    folder = filedialog.askdirectory(title="Select folder with .txt files")
    if not folder:
        print("No folder selected. Exiting.")
        return

    # Iterate over each file in the selected folder
    for filename in os.listdir(folder):
        # Process only .txt files
        if filename.lower().endswith((".txt", ".jpg")):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                # Separate the filename from its extension
                base_name, extension = os.path.splitext(filename)
                new_base_name = base_name

                # Remove the specific suffix if it exists
                if base_name.endswith("_contours"):
                    new_base_name = base_name[: -len("_contours")]
                elif base_name.endswith("_input_copy"):
                    new_base_name = base_name[: -len("_input_copy")]

                # Rename the file only if a change in the name is needed
                if new_base_name != base_name:
                    new_filename = new_base_name + extension
                    new_file_path = os.path.join(folder, new_filename)
                    try:
                        os.rename(file_path, new_file_path)
                        print(f"Renamed '{filename}' to '{new_filename}'")
                    except Exception as e:
                        print(f"Error renaming '{filename}': {e}")


if __name__ == "__main__":
    main()
