# sample-rag

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow using Python.  
It combines information retrieval techniques with generative models to answer queries based on a custom knowledge base.

## Features

- Loads and indexes documents for retrieval
- Uses a retriever to find relevant context for a given query
- Generates answers by combining retrieved context with a language model

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your document dataset.
2. Run the main script to start the retrieval and generation process.
3. Input your query and receive an answer augmented with retrieved information.

## Requirements

- Python 3.x
- Required libraries (see `requirements.txt`)

## Example

```python
query = "What is Retrieval-Augmented Generation?"
# Output: A generated answer using retrieved context from the knowledge base.
```

## License

MIT
