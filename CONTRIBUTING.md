# Contributing to Image Classification with GPU Optimization

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🎯 Project Goals

This project demonstrates:
- End-to-end computer vision pipeline development
- GPU-accelerated deep learning with PyTorch
- Proper ML engineering practices (reproducibility, modular code, comprehensive evaluation)
- Production-ready deployment with interactive demos

## 🔧 Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Project 4 - CIFAR-10 Image Classification"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

## 📝 How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Include system information (OS, Python version, GPU if applicable)
- Provide clear steps to reproduce the issue
- Include error messages and logs

### Suggesting Enhancements

- Open an issue with the `enhancement` label
- Clearly describe the proposed feature
- Explain the use case and benefits
- Consider backward compatibility

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update documentation if needed

4. **Test your changes**
   - Run the notebooks to ensure they still work
   - Test the Gradio app: `python app/gradio_app.py`
   - Verify model loading and predictions

5. **Commit your changes**
   ```bash
   git commit -m "feat: add your feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `refactor:` for code refactoring
   - `test:` for adding tests

6. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🎨 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and modular
- Maximum line length: 100 characters

## 🧪 Testing Guidelines

- Ensure notebooks run from top to bottom without errors
- Test with both CPU and GPU (if available)
- Verify the Gradio app works with various image inputs
- Check that all example images load correctly

## 📚 Documentation

- Update README.md if adding new features
- Document new functions with docstrings
- Add inline comments for complex algorithms
- Update notebooks with clear explanations

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Celebrate contributions of all sizes

## 📧 Questions?

Feel free to open an issue with the `question` label if you need help or clarification.

---

Thank you for contributing! 🚀