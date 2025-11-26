from pdfminer.high_level import extract_text
import os

input_path = "data/raw/"
output_path = "data/text/"

os.makedirs(output_path, exist_ok=True)

for file in os.listdir(input_path):
    if file.endswith(".pdf"):
        pdf_file = os.path.join(input_path, file)
        txt_file = os.path.join(output_path, file.replace(".pdf", ".txt"))

        text = extract_text(pdf_file)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)

        print("Extracted:", file)
