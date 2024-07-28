Project Title: Detecting AI-Generated vs. Human-Generated Text Using Deep Learning

The rise of large language models (LLMs) such as GPT-3 has led to an explosion in the creation of AI-generated text. These models can produce highly coherent and contextually relevant text that can easily be mistaken for human writing. This project focuses on developing a deep learning-based system to distinguish between AI-generated text and human-generated text. By leveraging a comprehensive dataset from Kaggle, which contains labeled examples of both types of text, we aim to build and train a model capable of identifying the origin of the text with high accuracy.

Project Structure
The project is organized into the following directories:

1. Data: Contains the datasets used for training and evaluation.
2. Notebooks: Contains Jupyter notebooks for experimentation and analysis.
3. Results: Stores the results and evaluation metrics obtained from the models.

Installation
To run the code in this project, follow these steps:
1. Clone the repository: `git clone https://github.com/polonium31/SEP-740-Project_Detecting-AI-vs-Human-generated-text.git`

2. The project contains three main folders:
- dataset: Includes the datasets used for training and evaluation.
  - `Training_Essay_Data.csv`: The main dataset.
  - `cleaned_data.csv`: The dataset after exploratory data analysis (EDA) is performed.

- notebooks: Contains Jupyter notebooks for experimentation and analysis.
  - `Data_Processing.ipynb`: Contains EDA code.
  - `ML_Code.ipynb`: Contains code for solving the problem using Machine Learning techniques.
  - `DL_LSTM_model.ipynb`: Contains code for using the LSTM model.
  - `DL_BERT_model.ipynb`: Contains code for using the BERT model.
  - `DL_RoBERTa_model.ipynb`: Contains code for using the RoBERTa model.
  - `DL_GPT2_model.ipynb`: Contains code for using the GPT2 model.

- results: Contains all important graphs.

All Deep Learning models are available here: https://drive.google.com/drive/folders/1RAcC2Rt0oylTFr5aONWQrZefPPnl6Qyj?usp=sharing

Running
- Open and run `Data_Processing.ipynb` to perform exploratory data analysis (EDA) and necessary data preprocessing.
- Open `ML_Code.ipynb` to explore and execute Machine Learning models like Random Forest, AdaBoost, and XGBoost.
- Open `DL_LSTM_model.ipynb` to work with the LSTM model for deep learning-based text classification.
- Open `DL_BERT_model.ipynb` to work with the BERT model for deep learning-based text classification.
- Open `DL_RoBERTa_model.ipynb` to work with the RoBERTa model for deep learning-based text classification.
- Open `DL_GPT2_model.ipynb` to work with the GPT2 model for deep learning-based text classification.

Dataset
The dataset used in this project can be obtained from Kaggle. It is available at the following link: [LLM Detect AI-Generated Text Dataset](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset).

To use this dataset, follow these steps:
1. Visit the provided Kaggle link.
2. Click on the "Download" button to download the dataset.
3. Extract the downloaded file to a desired location on your local machine.

Team Members and Contributions

Raj Joshi
- Contributions: EDA, Report
- GitHub Profile Link: https://github.com/rajjoshi18

Simran Chadda
- Contributions: Initial problem defining, ML
- GitHub Profile Link: https://github.com/SimranChadda

Jainish Patel
- Contributions: Initial problem defining, DL (LSTM & BERT), Report
- GitHub Profile Link: https://github.com/polonium31

Mayur Patel
- Contributions: DL (GPT), Presentation
- GitHub Profile Link: https://github.com/mayur045

References
1. Term Frequency-Inverse Document Frequency (TF-IDF): Text Vectorization. (n.d.). Towards Data Science. Retrieved from https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv: https://arxiv.org/abs/1810.04805.
3. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv: https://arxiv.org/abs/1907.11692.
4. Weber-Wulff, D., Anohina-Naumeca, A., Bjelobaba, S., Foltýnek, T., Guerrero-Dib, J., Popoola, O., Šigut, P., & Waddington, L. (2023). Testing of detection tools for AI-generated text. https://edintegrity.biomedcentral.com/articles/10.1007/s40979-023-00146-z
5. Islam, N., Sutradhar, D., Noor, H., Raya, J. T., Maisha, M. T., & Farid, D. M. (Year). Distinguishing Human Generated Text From ChatGPT Generated Text Using Machine Learning. arXiv: https://arxiv.org/abs/2306.01761.

