import os
import sys
from tkinter import Tk, filedialog

def main():
    # Initialize Tkinter root window and set it topmost
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()  # Hide the main window

    # Open a folder selection dialog
    folder_path = filedialog.askdirectory(title="Select folder with JPG files")
    if not folder_path:
        print("No folder selected. Exiting.")
        sys.exit()

    # Iterate over all files in the selected folder (non-recursive)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it is a file and if its extension is .jpg (case-insensitive)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() == '.jpg':
            base_name = os.path.splitext(filename)[0]
            new_filename = base_name + ".jpeg"
            new_file_path = os.path.join(folder_path, new_filename)
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as error:
                print(f"Error renaming {filename}: {error}")

if __name__ == '__main__':
    main()
