# Contributing to Student Dropout Prediction System

Thank you for your interest in contributing! This project is open-source and welcomes contributions, bug reports, and suggestions.

## Code of Conduct
Please maintain a respectful, constructive, and inclusive environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/student-dropout-prediction.git
   cd student-dropout-prediction
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Verify setup**:
   ```bash
   python setup.py
   pytest tests/ -v
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Make your improvements or fixes.
3. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   python test_ml.py
   ```
4. Commit your changes with clear, semantic commit messages:
   ```bash
   git commit -m "feat: add feature description"
   ```
5. Push to your branch and submit a Pull Request.

## Reporting Issues
If you encounter a bug or have a feature request:
- Search existing GitHub Issues before opening a new one.
- Provide clear reproduction steps, environment details (Python version, OS), and error logs.
