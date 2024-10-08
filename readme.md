This Repository contains the code for Alexander Berndt's individual project at the University of Mannheim. 

# Setup

To setup the code, it's best to create a virtual environment via 

```bash
python -m venv path/to/venv
```

Then, activate the virtual environment and install the required dependencies
```bash
source path/to/venv/bin/activate
pip install -r requirements.txt
```

To run `api.py`, a milvus instance is needed. The easiest way to achieve this is to host a local instance that can be installed by following the [docker standalone guide](https://milvus.io/docs/install_standalone-docker.md).

For managing the milvus instance, one might also want to install as explained in their [docs](https://github.com/zilliztech/attu).

# Content
The main files in this repository are 

1. `api.py`: contains the flask API that offers endpoints for persisting embeddings and retrieving relevant documents based on a semantic similarity search.
2. `experiments.ipynb`: contains the experiments to evaluate different dense retriever models.
3. `llm.py`: wrapper classes for the dense retriever models. 
4. `create_maven_sample.py`: script for random sampling of maven artifacts from `https://repo1.maven.org/maven2/`.
5. `analysis.ipynb`: notebook for SRM analysis. 

