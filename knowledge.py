import os

def load_documents():
    folder_path = "company_data"
    combined_text = ""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            combined_text += f"\n\n--- {filename.upper()} ---\n"
            combined_text += file.read()

    return combined_text
