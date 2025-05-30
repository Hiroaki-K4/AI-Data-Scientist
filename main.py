import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import nbformat
import yaml
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

    print("Finish creating", output_path)


def run_notebook(notebook_path, output_path):
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

        print(
            "Finish running {0}. Executed file is {1}.".format(
                notebook_path, output_path
            )
        )
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
                    "Fix the attached file and output the Python code as markdown."
                    "The error message is: {0}. The original objective is: {1}".format(
                        error_msg, original_objective
                    ),
                    output_file,
                ]
            )
            markdown_to_notebook(response.text, input_file)
            retry += 1


def load_eval_score(path):
    """Load the evaluation score from a JSON file."""
    if not os.path.exists(path):
        return 0.0
    with open(path, "r") as f:
        data = json.load(f)
        return data.get("eval_score", 0.0)


def main(api_key, train_data_path, detail_web_link, model_name, iter_num, retry_num):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    data_folder = os.path.dirname(train_data_path)
    eval_score_path = "eval_score.json"

    eval_info = model.generate_content(
        contents=[
            "Explain simply what the evaluation metrics are by reading this link {0}".format(
                detail_web_link
            )
        ]
    )

    original_objective = (
        "First, read this link: {0}. Also, create Python code to achieve the following goals in markdown format. "
        "The data is located at {1}. Save the best evaluation score as a JSON file in {2}, using the key 'eval_score' and the evaluation value as the value. "
        "Use a random seed of 42. For evaluation, split 80% of the data for training and 20% for validation. "
        "The file should be stored in the current directory without creating a new one.\n\n"
        "Problem Definition: Autonomously understand data science problems\n"
        "Data Exploration: Understand, preprocess, and visualize the data\n"
        "Feature Engineering: Create valid features\n"
        "Model Selection: Select a model and perform hyperparameter tuning based on the problem\n"
        "Evaluation and Improvement: Evaluate and improve model performance. Explanation of evaluation is as follows. {3}\n"
        "Reporting: Report results and present insights."
    ).format(detail_web_link, data_folder, eval_score_path, eval_info)

    print("Pass prompt to LLM api...")
    response = model.generate_content([original_objective, train_data_path])
    generated_notebook_name = "generated_with_gemini_0.ipynb"
    executed_notebook_name = "executed_notebook_0.ipynb"
    markdown_to_notebook(response.text, generated_notebook_name)
    run_notebook_with_retry(
        model,
        retry_num,
        generated_notebook_name,
        executed_notebook_name,
        original_objective,
    )
    eval_score = load_eval_score(eval_score_path)
    eval_res = {0: eval_score}
    print("0: Evaluation score {0}\n".format(eval_score))

    for i in range(iter_num):
        generated_notebook_name = "generated_with_gemini_{0}.ipynb".format(i + 1)
        executed_notebook_name = "executed_notebook_{0}.ipynb".format(i + 1)
        print("Pass prompt to LLM api...")
        response = model.generate_content(
            [
                "Read the attached Jupyter notebook and plan how to improve the evaluation score."
                "Then, implement Python code to improve the evaluation score in markdown format.\n"
                "The original objective is: {0}".format(original_objective),
                executed_notebook_name,
            ]
        )
        markdown_to_notebook(response.text, generated_notebook_name)
        run_notebook_with_retry(
            model,
            retry_num,
            generated_notebook_name,
            executed_notebook_name,
            original_objective,
        )
        eval_score = load_eval_score(eval_score_path)
        eval_res[i + 1] = eval_score
        print("{0}: Evaluation score {1}\n".format(i + 1, eval_score))

    # Output all evaluation scores
    print("~~~~~~~~~~ Evaluation scores ~~~~~~~~~~")
    for k, v in eval_res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(
        config["setting"]["api_key"],
        config["setting"]["train_data_path"],
        config["setting"]["detail_web_link"],
        config["setting"]["model_name"],
        config["setting"]["iter_num"],
        config["setting"]["retry_num"],
    )
