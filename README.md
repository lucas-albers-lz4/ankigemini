# Regenerate anki decks using gemini llm to revise them for accuracy 

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

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  
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
python3 scan_correct.py -i "AWS Cloud Practitioner Practice Exam Questions.txt" -o "AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt" --tier paid_tier_1

```
### Anki Deck Generator

```bash
python3 create_apkg.py -i "AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt" -o aws.apkg
```

## Example Output updating accuracy of questions
```bash
python3 ./scan_correct.py -i "AWS Cloud Practitioner Practice Exam Questions.txt" -o "AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt" --tier paid_tier_1 -q "4,5,9" 2>&1 |grep -v WARNING
2025-02-25 12:57:03,816 - INFO - Initializing Gemini client...
2025-02-25 12:57:03,816 - INFO - Gemini client configured successfully
2025-02-25 12:57:03,817 - INFO - Initialized Gemini model: gemini-2.0-flash
2025-02-25 12:57:03,817 - INFO - Rate limiter initialized for paid_tier_1 tier with: 2000 RPM, 4000000 TPM, No RPD limit
2025-02-25 12:57:03,817 - INFO - Loading data from AWS Cloud Practitioner Practice Exam Questions.txt
2025-02-25 12:57:04,095 - INFO - Loaded 973 questions
2025-02-25 12:57:04,095 - INFO - Processing questions: [4, 5, 9]
2025-02-25 12:57:04,097 - INFO - Loaded checkpoint, resuming from index 972
2025-02-25 12:57:04,097 - INFO - Resuming from checkpoint at index 972 with 0 items remaining
2025-02-25 12:57:04,097 - INFO - 
Processing Complete:
    Parse Error Summary:
    - Total Strict Parse Failures: 0
    - Successfully Recovered (Lenient): 0
    - Complete Parse Failures: 0
    
    Detailed Parse Error Log:
2025-02-25 12:57:04,097 - INFO -     No parsing errors encountered
2025-02-25 12:57:04,097 - INFO - Writing output to AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt
2025-02-25 12:57:04,101 - INFO - Writing 972 unique questions to output file
2025-02-25 12:57:04,107 - INFO - Output written to AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt
2025-02-25 12:57:04,107 - INFO - Writing output to AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt
2025-02-25 12:57:04,110 - INFO - Writing 972 unique questions to output file
2025-02-25 12:57:04,116 - INFO - Output written to AWS Cloud Practitioner Practice Exam Questions.CORRECTED.txt
2025-02-25 12:57:04,116 - INFO - Starting comprehensive dataset validation
2025-02-25 12:57:04,121 - INFO - 
--- Processing Summary ---
2025-02-25 12:57:04,121 - INFO - Total questions processed: 973
2025-02-25 12:57:04,121 - INFO - Number of answers updated: 0
2025-02-25 12:57:04,121 - INFO - Starting deduplication process...
2025-02-25 12:57:04,121 - INFO - Deduplication complete. Removed -970 duplicates.
```


## Logging

- Detailed processing logs in `scan_correct.log`
- Checkpoints saved in `checkpoints/` directory
- Supports resuming interrupted processing

## Rate Limiting

- Implements intelligent rate limiting for API calls
- Supports both free and paid Google Gemini API tiers (free will work but it's slow, and you need to consider the request per minute and per day)
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
   Solution: Upgrade to paid tier

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
You must handle 429s...

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
