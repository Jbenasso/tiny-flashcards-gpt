# TinyFlashcards-GPT

TinyFlashcards-GPT is a lightweight transformer model built in PyTorch that generates study flashcards (Q&A pairs) from input text.  
It is designed as a learning and research project that combines natural language processing with educational applications.

---

## Features
- Character-level transformer model
- Converts context text into question-and-answer flashcards
- Simple training loop with PyTorch
- Preprocessed sample dataset (`data/flashcards.json`)
- Jupyter notebook demo for training and testing
- Extensible design for larger datasets and experiments

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/jbenasso/tiny-flashcards-gpt.git
cd tiny-flashcards-gpt
pip install -r requirements.txt
```

---

## Usage

### Training
Train the model on the provided sample dataset:

```bash
python main.py
```

This will create a checkpoint file `tinyflashcards.pth`.

### Generating Flashcards
After training, generate flashcards from new context text:

```bash
python generate.py
```

Example input:
```
Context: The mitochondria is an organelle that produces ATP for energy. Q:
```

Example output:
```
Q: What organelle produces ATP? A: The mitochondria.
```

---

## Dataset
The repository includes a small starter dataset (`data/flashcards.json`) with sample contexts and flashcards.  
You can expand this by adding more Q&A pairs or using larger datasets such as SQuAD or Wikipedia-based corpora.

Example JSON entry:
```json
{
  "context": "Photosynthesis is a process used by plants to convert light into chemical energy.",
  "flashcard": "Q: What process do plants use to convert light into chemical energy? A: Photosynthesis."
}
```

---

## Project Structure
```
tiny-flashcards-gpt/
│── main.py              # training script
│── generate.py          # generation script
│── requirements.txt     # dependencies
│── LICENSE              # license file
│── .gitignore           # ignore unnecessary files
│
├── data/
│   ├── flashcards.json  # starter dataset
│
├── notebooks/
│   ├── demo.ipynb       # notebook demo
│
└── experiments/
    ├── exp_config.yaml  # (optional) experiment configs
```

---

## Future Work
- Add support for word-level tokenization
- Train on larger question-answer datasets (e.g., SQuAD)
- Incorporate attention visualization to show model focus
- Deploy as a Streamlit app for interactive use

---

