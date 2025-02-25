# AWS Cloud Practitioner Exam Question Processing Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/aws-cert-exam-tools/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/aws-cert-exam-tools/actions)

## Table of Contents
- [Project Overview](#-project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Logging](#logging)
- [Rate Limiting](#rate-limiting)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Contributing](#contributing)
- [Limitations](#limitations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🌟 Project Overview

This project provides a toolkit for processing and enhancing AWS Cloud Practitioner exam questions, with two primary scripts:

1. **Question Accuracy Enhancer (`scan-correct.py`)**: 
   - Uses AI to validate and improve exam question answers
   - Checks the accuracy of existing answers
   - Generates more comprehensive explanations for incorrect answers
   - Supports batch processing of exam questions

2. **Anki Deck Generator (`create_apkg.py`)**: 
   - Converts processed exam questions into an Anki flashcard deck
   - Creates interactive study materials for AWS Cloud Practitioner exam preparation

## Features

### Question Processing Script
- AI-powered accuracy verification
- Batch processing of exam questions
- Detailed logging of processing steps
- Checkpoint and resume functionality
- Rate-limited API calls to Google Gemini

### Anki Deck Generator
- Converts text-based questions to Anki flashcards
- Supports multiple-choice questions
- Highlights correct answers
- Generates study-friendly card formats

## Prerequisites

- Python 3.8+
- Google Gemini API access
- Anki (optional, for viewing generated decks)

Copy down the original questions from this thread:
https://www.reddit.com/r/AWSCertifications/comments/1gxkwkp/anki_cards_with_over_900_questions_for_aws_cloud/

Import it into Anki and then import it to a text file:
1. Export Format -> "Cards in Plain Text (.txt)"
2. Click Export

Then feed it into the script to check all the answers using Google Gemini LLM.
The script:
- Confirms any answer asking for N answers has N answers where N is 1 or 2
- Adds additional "optionalExplanation" as needed
- Sometimes the model gives an additional explanation in regards to the question/answer, we capture that and include in the answer display
- Confirms the answer to the question is correct, and fixes it if it is not

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aws-cert-exam-tools.git
   cd aws-cert-exam-tools
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt

   ```

3. Or create virtual environment with uv:
   ```bash
   uv venv -p python3.12 venv; source venv/bin/activate; uv pip install -r requirements.txt
   ```

## Configuration

### API Key Setup
1. Create a `.env` file in the project root
2. Add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

## Usage

### Question Processing Script

```bash
python scan-correct.py -i input_questions.txt -o corrected_questions.txt \
    --tier paid_tier_1 \
    --batch-size 10  # Optional: default is 20
    --limit 100  # Optional: process only first 100 questions
```

#### Command Line Arguments
- `-i/--input`: Input question file path (required)
- `-o/--output`: Output file path (required)
- `--tier`: API tier (free or paid_tier_1, default: free)
- `--batch-size`: Questions per batch (default: 20)
- `--limit`: Limit total questions processed (optional)

### Anki Deck Generator

```bash
python create_apkg.py -i corrected_questions.txt -o aws_exam_deck.apkg
```

#### Command Line Arguments
- `-i/--input`: Input corrected questions file (required)
- `-o/--output`: Output Anki package file path (required)
- `-v/--verbose`: Enable verbose logging
- `-q/--questions`: Process specific questions (e.g., "1,2,3")

## Example Output

### Question Processing
```

```

### Generated Anki Cards
![Example Anki Card](docs/images/example_card.png)

## Logging

- Detailed processing logs in `scan_correct.log`
- Checkpoints saved in `checkpoints/` directory
- Supports resuming interrupted processing

## Rate Limiting

- Implements intelligent rate limiting for API calls
- Supports both free and paid Google Gemini API tiers (free will work but it's slow)
- Automatic backoff and retry mechanisms

## Development Setup

### Setting up Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_apkg.py
```

## Troubleshooting

### Common Issues
1. **API Rate Limiting**
   ```
   Error: 429 Too Many Requests
   ```
   Solution: Reduce batch size or upgrade to paid tier

2. **Input File Format**
   ```
   Error: Invalid file format
   ```
   Solution: Ensure exported Anki file follows required format


## Security

### API Key Safety
- Never commit your API keys to version control
- Use environment variables or .env files
- Rotate keys regularly

### Rate Limiting
The tool implements rate limiting to prevent API abuse and excessive costs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Limitations

- API usage is subject to Google Gemini rate limits
- Quality of answer generation depends on AI model performance
- Large batches may require significant processing time
- Memory usage scales with batch size

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini AI
- Anki Flashcard Platform
- Reddit AWS Cloud Exam Community
- All contributors and testers 
