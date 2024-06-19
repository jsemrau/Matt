# Project Matt

Welcome to the Matt repository! 
Matt is a cognitive agent reearch project that uses the ReAct pattern to think through a query
Matt also has access to memory and a diverse range of tools.
Despite using a local version of Mistral 0.3 instruct whose LLM will be downloaded during setup,
Matt is quite stable in its reply. 

Important: This is a research project. Don't use in production. Matt's answers might be incorrect. 


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/jsemrau/Matt.git
    cd your-repo-name
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```

4. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Download the Mistral LLM**:
    This project uses a local version of Mistral whose large language model (LLM) will be downloaded automatically during setup. Please ensure you have sufficient storage and a stable internet connection.

## Usage

After installation, you can start using the project by running the main script:

```sh
streamlit run app.py
