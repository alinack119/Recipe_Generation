Recipe Generation Project

This project is a GPT-style recipe text generator built from scratch with PytTorch. It learns the pattern of recipe titles, ingredients lists, and directions from a dataset of 62,126 recipes.

To run: 
- download required tools from requirement.txt.
    - pip install -r requirements.txt
- Using CLI which is in gpt.py
    - train data: python scipts/gpt.py train
          - can change parameters
          python scripts/gpt.py train 
          --steps 5000 
          --batch-size 8
          --context-size 256 
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
- .gitignore (prevents large files from entering git


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


Preprocessing Path:
- raw data
- scripts/preprocess.py
      - cleans ingredient list
      - cleans directions
      - normalizes fractions
      - inserts special tokens
- saves to csv and txt file


Training path:
- python scripts/gpt.py train
- scripts/train.py train_model(args)
      - src/tokenizer.py (converts recipes.txt to token IDs)
      - src/model.py (constructs GPT transformer)
      - src/training_utils.py (creates batches, estimates loss, performs backpropagation
- saves data to checkpoints/recipe_transformer.pt



Evaluation path:
- python scripts/gpt.py eval --interactive
- scripts/generate.py (ask user for title and ingredients)
      - src/tokenizer.py (encode prompt and decode output)
      - src/model.py (reconstructs transformer)
      - src/training_utils.py (generates one token at a time)
- Recipe generated

Uses GPT-2's tokenizzation and BPE vocalbulary
Special Tokens:
- Structural boundaries
    - tell model where each section starts and ends
- Efficient tokenization
    - each boundary is one token
- Prompt control
    - can begin generation at a particular section
- Stopping condition
    - lets generation stop at meaningful structural boundary instead of token limit


Limitation:
- User Ingredient Generation
    - it generates recipe using ingredients outside of user input
- Ingredient measurement for user input
- Imperfect text generation
- Samples from full probability distribution
    - does not implement temperature, top-k filtering
  
