Recipe Generation Project

This project is a GPT-style recipe text generator built from scratch with PytTorch. It learns the pattern of recipe titles, ingredients lists, and directions from a dataset of 62,126 recipes.

To run: 
- download required tools from requirement.txt.
- Using CLI which is in gpt.py
    - train data: python scipts/gpt.py train
    - generate and evaluation: python scripts/gpt.py eval
    - interactive recipe generation: scripts/gpt.py eval --interactive
  


Organization:
- Scripts/ (contains programs user runs)
    - gpt.py --> command line interface
    - preprocess.py 
    - train.py 
    - generate.py
- Src/ (contains reusable model, tokenization, data, and training logic)
    - model.py
    - tokenizer.py
    - training_utils.py
    - data.py
    - config.py
    - __init__.py
- data/ (raw and processed data)
- checkpoints/ (stores trained model parameters)
- notebooks/ (ipynb files documenting experimentation and development process)
- requirements.txt (required tools to download)
- .gitignore



Transformer

The transformer calculates predictions, compares them with the correct next tokens, computes cross-entropy loss, and updates its parameters through backpropagation.
After training, the program saves:
- Learned model weights
- Vocabulary size
- Context size
- Embedding size
- Number of heads
- Number of layers

These are saved to a .pt checkpoint

During evaluation:
1. loads the checkpoint from training
2. reconstructs the same transformer architecture
3. loads trained weights
4. tokenizes the prompt
5. predicts one new token
6. adds that token to the sequence
7. repeats until reaching token limit or <END_RECIPE>
8. decodes the IDs back to text





  
