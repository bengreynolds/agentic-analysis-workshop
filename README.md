# Agentic Analysis Workshop

This repository contains setup instructions, dependency files, data, and supporting resources for an agentic coding workshop built around a guided data analysis example.

The goal is to help participants arrive with the required accounts, software, and Python environment already installed so the workshop can start quickly.

## Pre-Workshop Requirements

Please complete these steps before the workshop.

### 1. Install Anaconda

Install Anaconda from the official download page:

https://www.anaconda.com/download

After installation, open Anaconda Prompt, Terminal, or PowerShell and confirm Conda is available:

```bash
conda --version
```

### 2. Install Visual Studio Code

Install Visual Studio Code:

https://code.visualstudio.com/

After installing VS Code, install these extensions:

- GitHub Pull Requests
- Codex

You can install extensions from the VS Code Extensions panel.

### 3. Create Required Accounts

Participants need access to:

- A GitHub account: https://github.com/
- An OpenAI account: https://platform.openai.com/

Sign in to GitHub from VS Code before the workshop if possible.

## Python Environment

This repository includes a starter Conda environment file at `environment.yaml`.

Create the environment from the repository root:

```bash
conda env create -f environment.yaml
```

Activate the environment:

```bash
conda activate agentic-analysis-workshop
```

If the environment already exists and the dependency file has changed, update it with:

```bash
conda env update -f environment.yaml --prune
```

## VS Code Setup

After activating the environment, open this repository in VS Code.

Select the workshop Conda environment as the Python interpreter:

1. Open the Command Palette.
2. Search for `Python: Select Interpreter`.
3. Select the `agentic-analysis-workshop` Conda environment.

## Readiness Checklist

Before the workshop, confirm that you have:

- Anaconda installed
- VS Code installed
- GitHub Pull Requests extension installed in VS Code
- Codex extension installed in VS Code
- A GitHub account
- An OpenAI account
- The `agentic-analysis-workshop` Conda environment created

## Repository Contents

- `README.md`: setup instructions for workshop participants
- `environment.yaml`: starter Conda environment for the data analysis example
