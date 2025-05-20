import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def markdown_to_notebook(md_text: str, output_path: str):
    """
    Convert markdown with ```python``` code fences into a Jupyter Notebook.

    Args:
        md_text: The full markdown text.
        output_path: Path where the .ipynb will be written.
    """
    # Split text into segments: code fences vs. other markdown
    parts = re.split(r"(```python.*?```)", md_text, flags=re.S)
    cells = []

    for part in parts:
        if part.startswith("```python"):
            # Strip the fence markers, leave only the code
            code = re.sub(r"```python\n?|```", "", part).strip()
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
    with open(out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook written to: {output_path}")


def run_notebook(notebook_path, output_path="executed_notebook.ipynb"):
    try:
        # Load the notebook
        with open(notebook_path) as f:
            notebook = nbformat.read(f, as_version=4)

        # Set up the execution processor
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        # Run the notebook
        ep.preprocess(notebook)

        # Save the executed notebook
        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        print("Notebook executed successfully.")
        return None

    except Exception as e:
        print(f"Error occurred while executing the notebook:\n{e}")
        return e


def run_notebook_with_retry(
    model, retry_num, input_file, output_file, original_objective
):
    retry = 0
    while True:
        if retry >= retry_num:
            raise RuntimeError("The number of retries has exceeded the limit.")
        error_msg = run_notebook(input_file, output_file)
        if error_msg is None:
            break
        else:
            response = model.generate_content(
                [
                    "Fix attached file and output python code as markdown. Error message is here. {0} Original objective is here. {1}".format(
                        error_msg, original_objective
                    ),
                    output_file,
                ]
            )
            markdown_to_notebook(response.text, input_file)
            retry += 1


def load_accuracy(path):
    """Load the accuracy from a JSON file."""
    if not os.path.exists(path):
        return 0.0
    with open(path, "r") as f:
        data = json.load(f)
        return data.get("accuracy", 0.0)


def main(api_key, data_folder, detail_web_link, iter_num, retry_num):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    train_data = os.path.join(data_folder, "train.csv")

    original_objective = (
        "First read this link. {0} Also, create python code to achieve following goals as markdown. Data is located in {1}. Save the best accuracy value as a json file in accuracy.json with key as “accuracy” and value as the accuracy value.\n"
        "Problem definition: autonomously understand business problems and data science tasks\n"
        "Data exploration: understand, preprocess, and visualize data\n"
        "Feature Engineering: Creation of valid features"
        "Model Selection: Model selection and hyper-parameter tuning according to the problem\n"
        "Evaluation and improvement: Evaluate and improve model performance\n"
        "Reporting: reporting results and presenting insights".format(
            detail_web_link, data_folder
        )
    )

    response = model.generate_content([original_objective, train_data])
    markdown_to_notebook(response.text, "generated_with_gemini_0.ipynb")
    run_notebook_with_retry(
        model,
        retry_num,
        "generated_with_gemini_0.ipynb",
        "executed_notebook_0.ipynb",
        original_objective,
    )
    acc_path = "accuracy.json"
    max_acc = load_accuracy(acc_path)

    for i in range(iter_num):
        response = model.generate_content(
            [
                "Read attached jupyter notebook, and plan how to improve evaluation score. After that, implement python code to improve evaluation score as markdown.\n"
                "Original objective is here. {0}".format(original_objective),
                "executed_notebook_{0}.ipynb".format(i),
            ]
        )
        markdown_to_notebook(
            response.text, "generated_with_gemini_{0}.ipynb".format(i + 1)
        )
        run_notebook_with_retry(
            model,
            retry_num,
            "generated_with_gemini_{0}.ipynb".format(i + 1),
            "executed_notebook_{0}.ipynb".format(i + 1),
            original_objective,
        )
        acc = load_accuracy(acc_path)
        max_acc = max(max_acc, acc)
        i += 1
        print("{0}: Max accuracy {1}".format(i, max_acc))


if __name__ == "__main__":
    api_key = ""
    data_folder = "data/titanic"
    detail_web_link = "https://www.kaggle.com/competitions/titanic"
    iter_num = 5
    retry_num = 5
    main(api_key, data_folder, detail_web_link, iter_num, retry_num)
