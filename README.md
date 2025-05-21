# AI-Data-Scientist
<img src='images/ai_data_scientist.jpg' width='500'>

In this repogitory, we can try AI data scientist that behaves as a data scientist. Recently, LLM has been improved a lot, so we thought we could make LLM solve problems that data scientist handle usually.

<br></br>

## How to work
The flow of AI data scientist is as follows.

<img src='images/flow.png' width='500'>

### 1. Create the code

### 2. Run the code

### 3. Fix the code

### 4. Improve the code

<br></br>

## How to run
You can create an environment with `requirements.txt`. We used python 3.10 to run a program.

We can configure information about AI data scientist by editting `config.yaml`. Items of configulation are as follows.

`api_key` is the api key of gemini, so specify your gemini api key. `train_data_path` is training dataset path that you want to try. `detail_web_link` is the link that includes information about task like kaggle competition link. `model_name` is the name of gemini model. `iter_num` is the number of running improvement loop. `retry_num` is the number of retry to fix errors.

```
- api_key
- train_data_path
- detail_web_link
- model_name
- iter_num
- retry_num
```

You can run AI data scientist by running the following command.

```bash
python3 main.py
```

<br></br>

## References
- [AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
- [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
