import os
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter root and hide it
root = tk.Tk()
root.attributes("-topmost", True)  # Ensure dialog is on top
root.withdraw()

# Prompt user for the current token and the new token
current_token = input("Enter the current token (1 or 2 characters): ").strip()
new_token = input("Enter the new token (1 or 2 characters): ").strip()

# Validate that current token is either 1 or 2 characters
if len(current_token) not in [1, 2]:
    print("Current token must be 1 or 2 characters. Exiting...")
    exit()

# Prompt user to select a folder containing TXT files
folder_path = filedialog.askdirectory(title="Select Folder with TXT Files")
if not folder_path:
    print("No folder selected. Exiting...")
    exit()

# Iterate over all files in the selected folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            # Read the entire content of the file
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            if not content:
                print(f"File '{filename}' is empty. Skipping...")
                continue

            # Split content to extract the first token and the following separator
            first_token, sep, remainder = content.partition(" ")
            # Verify that a space follows the token and token length is 1 or 2
            if sep == "" or len(first_token) not in [1, 2]:
                print(
                    f"File '{filename}' does not match expected format (1 or 2 characters followed by a space). Skipping..."
                )
                continue

            # If the first token matches the provided current token, perform the replacement
            if first_token == current_token:
                new_content = new_token + sep + remainder
                # Write the modified content back to the file
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(new_content)
                print(f"Modified token in file: {filename}")
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")
