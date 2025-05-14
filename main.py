import google.generativeai as genai
import nbformat
from pathlib import Path
import re
import os
from nbconvert.preprocessors import ExecutePreprocessor


def markdown_to_notebook(md_text: str, output_path: str):
    """
    Convert markdown with ```python``` code fences into a Jupyter Notebook.

    Args:
        md_text: The full markdown text.
        output_path: Path where the .ipynb will be written.
    """
    # Split text into segments: code fences vs. other markdown
    parts = re.split(r'(```python.*?```)', md_text, flags=re.S)
    cells = []

    for part in parts:
        if part.startswith('```python'):
            # Strip the fence markers, leave only the code
            code = re.sub(r'```python\n?|```', '', part).strip()
            cells.append(nbformat.v4.new_code_cell(code))
        elif part.strip():
            # Anything else becomes a markdown cell
            cells.append(nbformat.v4.new_markdown_cell(part.strip()))

    # Assemble the notebook object
    nb = nbformat.v4.new_notebook(cells=cells)

    # Ensure output directory exists
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write the .ipynb file
    with open(out, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook written to: {output_path}")


def run_notebook(notebook_path, output_path="executed_notebook.ipynb"):
    try:
        # Load the notebook
        with open(notebook_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # Set up the execution processor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Run the notebook
        ep.preprocess(notebook)

        # Save the executed notebook
        with open(output_path, 'w') as f:
            nbformat.write(notebook, f)

        print("Notebook executed successfully.")

    except Exception as e:
        print(f"Error occurred while executing the notebook:\n{e}")


def main(data_folder):
    genai.configure(api_key="")

    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    train_data = os.path.join(data_folder, "train.csv")

    response = model.generate_content(
        [
            "Create python code to predict whether person can survive or not as Markdown.",
            train_data
        ]
    )
    print(response.text)
    markdown_to_notebook(response.text, "generated_with_gemini.ipynb")

    run_notebook("generated_with_gemini.ipynb")


if __name__ == "__main__":
    data_folder = "data/titanic"
    main(data_folder)
