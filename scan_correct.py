import argparse
import html
import json
import logging
import math
import os
import random
import re
import threading
import time
import traceback

# Suppress GRPC shutdown warning
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import google.generativeai as genai
from dotenv import load_dotenv
from google.genai.errors import ClientError
from google.generativeai.types import GenerationConfig

warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scan_correct.log", mode="w"),  # Overwrite file each run
        logging.StreamHandler(),  # This will also print to console
    ],
)

# Set debug level for our specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure all handlers process debug messages
for handler in logger.handlers:
    handler.setLevel(logging.DEBUG)

# Configuration
logger.info("Initializing Gemini client...")
API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.error("No API key found. Please set GEMINI_API_KEY in .env file.")
    raise ValueError("Missing GEMINI_API_KEY environment variable")

# Configure the API
genai.configure(api_key=API_KEY)
logger.info("Gemini client configured successfully")

# Model configuration
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.0-flash")

# Generation configuration
GENERATION_CONFIG: GenerationConfig = GenerationConfig(
    temperature=0.3,
    top_p=1,
    top_k=1,
    max_output_tokens=8192,
)

# Initialize the model
try:
    MODEL: genai.GenerativeModel = genai.GenerativeModel(
        model_name=MODEL_NAME, generation_config=GENERATION_CONFIG
    )
    logger.info(f"Initialized Gemini model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Model Initialization Error: {str(e)}")
    raise

# Rate limit tiers
RATE_LIMIT_TIERS: Dict[str, Dict[str, Optional[int]]] = {
    "free": {"rpm": 15, "tpm": 1_000_000, "rpd": 1_500},
    "paid_tier_1": {
        "rpm": 2_000,
        "tpm": 4_000_000,
        "rpd": None,  # No daily limit for paid tier 1
    },
}


# Rate limiting configuration
class RateLimiter:
    def __init__(self, tier: str = "free") -> None:
        limits = RATE_LIMIT_TIERS[tier]
        self.max_rpm: int = limits["rpm"]  # type: ignore
        self.max_tpm: int = limits["tpm"]  # type: ignore
        self.max_rpd: Optional[int] = limits["rpd"]
        self.tier: str = tier

        self.minute_requests: List[datetime] = []
        self.day_requests: List[datetime] = []
        self.minute_tokens: List[Tuple[datetime, int]] = []
        self.last_request_time: Optional[datetime] = None
        self.lock: threading.Lock = threading.Lock()

        # Rate tracking
        self.current_rpm: int = 0
        self.current_tpm: int = 0
        self.current_rpd: int = 0
        self.last_rate_check: datetime = datetime.now()

        # Limit tracking
        self.rpm_limit_hits: int = 0
        self.tpm_limit_hits: int = 0
        self.quota_limit_hits: int = 0

        # Increased minimum time between requests
        self.base_request_interval: float = 2.5  # Default to 2.5 seconds for all tiers
        self.min_request_interval: float = self.base_request_interval

        # Preserve existing dynamic backoff factor for current run
        self.run_start_time: datetime = datetime.now()
        self.consecutive_429s: int = 0
        self.last_429_time: Optional[datetime] = None
        self.backoff_multiplier: float = 1.0

        logger.info(
            f"Rate limiter initialized for {tier} tier with: {self.max_rpm} RPM, {self.max_tpm} TPM, "
            + (f"{self.max_rpd} RPD" if self.max_rpd else "No RPD limit")
        )

    def reset_backoff(self) -> None:
        """Reset backoff state for new run"""
        self.consecutive_429s = 0
        self.last_429_time = None
        self.backoff_multiplier = 1.0
        self.min_request_interval = self.base_request_interval
        self.run_start_time = datetime.now()
        self.rpm_limit_hits = 0
        self.tpm_limit_hits = 0
        self.quota_limit_hits = 0
        logger.info("Reset rate limiter backoff state for new run")

    def _update_current_rates(self) -> None:
        """Update current rate measurements"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)

        # Clean old entries
        self.minute_requests = [req for req in self.minute_requests if req > minute_ago]
        self.minute_tokens = [
            (time, tokens) for time, tokens in self.minute_tokens if time > minute_ago
        ]
        if self.max_rpd:
            self.day_requests = [req for req in self.day_requests if req > day_ago]

        # Update current rates
        self.current_rpm = len(self.minute_requests)
        self.current_tpm = sum(tokens for _, tokens in self.minute_tokens)
        self.current_rpd = len(self.day_requests) if self.max_rpd else 0
        self.last_rate_check = now

    def _diagnose_rate_limit(self) -> List[str]:
        """Diagnose which limit is likely being hit"""
        self._update_current_rates()

        rpm_percentage = (self.current_rpm / self.max_rpm) * 100 if self.max_rpm else 0
        tpm_percentage = (self.current_tpm / self.max_tpm) * 100 if self.max_tpm else 0
        rpd_percentage = (self.current_rpd / self.max_rpd) * 100 if self.max_rpd else 0

        diagnosis: List[str] = []
        if rpm_percentage > 80:
            diagnosis.append(
                f"RPM at {rpm_percentage:.1f}% ({self.current_rpm}/{self.max_rpm})"
            )
        if tpm_percentage > 80:
            diagnosis.append(
                f"TPM at {tpm_percentage:.1f}% ({self.current_tpm}/{self.max_tpm})"
            )
        if rpd_percentage > 80:
            diagnosis.append(
                f"RPD at {rpd_percentage:.1f}% ({self.current_rpd}/{self.max_rpd})"
            )

        return diagnosis

    def _handle_429(self) -> None:
        """Update backoff strategy when encountering 429s within current run"""
        now = datetime.now()

        # If this is a 429 from a previous run, reset state
        if self.last_429_time and (now - self.run_start_time).total_seconds() > 3600:
            self.reset_backoff()

        # Diagnose the cause
        diagnosis = self._diagnose_rate_limit()
        if diagnosis:
            logger.warning(f"Rate limit diagnosis: {', '.join(diagnosis)}")

        if self.last_429_time:
            # If we got another 429 within 30 seconds, increase backoff
            if (now - self.last_429_time).total_seconds() < 30:
                self.consecutive_429s += 1
                self.backoff_multiplier = min(5.0, self.backoff_multiplier * 1.5)
            else:
                # Reset if it's been more than 30 seconds
                self.consecutive_429s = 1
                self.backoff_multiplier = 1.0

        self.last_429_time = now

        # Adjust minimum interval based on 429 frequency and diagnosis
        self.min_request_interval = self.base_request_interval * self.backoff_multiplier
        logger.warning(
            f"Adjusting minimum request interval to {self.min_request_interval:.2f}s after {self.consecutive_429s} consecutive 429s"
        )

    def reset_consecutive_429s(self) -> None:
        """Reset consecutive 429s tracking on successful requests"""
        self.consecutive_429s = 0
        self.backoff_multiplier = 1.0
        self.min_request_interval = self.base_request_interval
        logger.info(
            f"Reset consecutive 429s tracking. Restoring base request interval to {self.min_request_interval:.2f}s"
        )

    def _wait_for_interval(self) -> None:
        """Ensure minimum time between requests with dynamic backoff"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                logger.debug(
                    f"Waiting {sleep_time:.2f}s between requests (backoff multiplier: {self.backoff_multiplier:.2f})"
                )
                time.sleep(sleep_time)

    def wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """
        Check all rate limits and wait if necessary.
        estimated_tokens: Estimated number of tokens for this request
        """
        now = datetime.now()

        with self.lock:
            self._handle_429()
            self._wait_for_interval()

            while True:
                # Check RPM
                rpm_current = len(self.minute_requests)
                if rpm_current >= self.max_rpm:
                    sleep_time = (
                        self.minute_requests[0] + timedelta(minutes=1) - now
                    ).total_seconds()
                    if sleep_time > 0:
                        logger.warning(
                            f"RPM limit reached ({rpm_current}/{self.max_rpm}), waiting {sleep_time:.2f} seconds"
                        )
                        time.sleep(sleep_time)
                elif rpm_current >= self.max_rpm * 0.8:  # Warning at 80% of limit
                    logger.warning(
                        f"RPM approaching limit ({rpm_current}/{self.max_rpm})"
                    )

                # Check RPD only if there's a daily limit
                if self.max_rpd:
                    rpd_current = len(self.day_requests)
                    if rpd_current >= self.max_rpd:
                        sleep_time = (
                            self.day_requests[0] + timedelta(days=1) - now
                        ).total_seconds()
                        if sleep_time > 0:
                            logger.warning(
                                f"RPD limit reached ({rpd_current}/{self.max_rpd}), waiting {sleep_time:.2f} seconds"
                            )
                            time.sleep(sleep_time)
                    elif rpd_current >= self.max_rpd * 0.8:  # Warning at 80% of limit
                        logger.warning(
                            f"RPD approaching limit ({rpd_current}/{self.max_rpd})"
                        )

                # Check TPM
                current_tpm = sum(tokens for _, tokens in self.minute_tokens)
                if current_tpm + estimated_tokens > self.max_tpm:
                    sleep_time = (
                        self.minute_tokens[0][0] + timedelta(minutes=1) - now
                    ).total_seconds()
                    if sleep_time > 0:
                        logger.warning(
                            f"TPM limit reached ({current_tpm}/{self.max_tpm}), waiting {sleep_time:.2f} seconds"
                        )
                        time.sleep(sleep_time)
                elif current_tpm >= self.max_tpm * 0.8:  # Warning at 80% of limit
                    logger.warning(
                        f"TPM approaching limit ({current_tpm}/{self.max_tpm})"
                    )

                # Refresh time and clean old requests
                now = datetime.now()

                # If all limits are good, we can proceed
                rpm_ok = len(self.minute_requests) < self.max_rpm
                rpd_ok = not self.max_rpd or len(self.day_requests) < self.max_rpd
                tpm_ok = current_tpm + estimated_tokens <= self.max_tpm

                if rpm_ok and rpd_ok and tpm_ok:
                    break

            # Add the new request
            self.minute_requests.append(now)
            if self.max_rpd:  # Only track daily if there's a limit
                self.day_requests.append(now)
            self.minute_tokens.append((now, estimated_tokens))
            self.last_request_time = now


# Create global rate limiter - will be updated based on command line args
rate_limiter: Optional[RateLimiter] = None


# Function to load the data
def load_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse the exam questions file into a structured JSON-like format.
    """
    logger.info(f"Loading data from {filepath}")

    questions: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex pattern to extract questions and answers
    pattern = r"<div>\s*<b>Question:</b><br>\s*(.*?)\s*<ul>(.*?)</ul>.*?<br><b>Correct Answer\(s\):</b>\s*([A-Z,\s]+)\s*</div>"

    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

    for match in matches:
        question_text = match[0].strip()
        options_html = match[1]
        correct_answers = match[2].strip().split(",")

        # Extract options
        option_pattern = r"<li(?:\s+class=\'.*?\')?>([^<]+)</li>"
        options = re.findall(option_pattern, options_html)

        # Prepare options with correct flag
        formatted_options: List[Dict[str, Union[str, bool]]] = []
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, etc.
            is_correct = letter in correct_answers
            formatted_options.append({"text": option.strip(), "is_correct": is_correct})

        question = {
            "question": question_text,
            "options": formatted_options,
            "correct_answer": next(
                (opt["text"] for opt in formatted_options if opt["is_correct"]), None
            ),
        }

        questions.append(question)

    logger.info(f"Loaded {len(questions)} questions")
    return questions


T = TypeVar("T")


def exponential_backoff(attempt: int, max_delay: float = 120.0) -> float:
    """Calculate exponential backoff time with jitter"""
    base_delay = min(max_delay, (2**attempt) * 1.25)  # More aggressive base delay
    jitter = random.uniform(0, min(max_delay - base_delay, base_delay))
    delay = base_delay + jitter
    return cast(float, delay)


def retry_on_429(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to retry on 429 errors with exponential backoff"""

    def wrapper(*args: Any, **kwargs: Any) -> T:
        if rate_limiter is None:
            raise RuntimeError("Rate limiter not initialized")

        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            try:
                # Wait if we're approaching rate limits
                rate_limiter.wait_if_needed()
                return func(*args, **kwargs)
            except ClientError as e:
                if hasattr(e, "code") and e.code == 429:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) reached. Giving up."
                        )
                        raise

                    # Update rate limiter's 429 tracking
                    rate_limiter._handle_429()

                    delay = exponential_backoff(attempt)
                    logger.warning(
                        f"Rate limit hit. Waiting {delay:.2f} seconds before retry {attempt}/{max_attempts}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Unexpected ClientError: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise

        # If we've exhausted all attempts, raise the last error
        raise RuntimeError(f"Failed after {max_attempts} attempts due to rate limiting")

    return wrapper


def save_checkpoint(
    data: List[Dict[str, Any]], output_filepath: str, current_index: int
) -> None:
    """Save current progress to a checkpoint file."""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_file = checkpoint_dir / f"{Path(output_filepath).name}.checkpoint"
    checkpoint_data: Dict[str, Any] = {
        "current_index": current_index,
        "data": data[: current_index + 1],
    }

    try:
        checkpoint_file.write_text(
            json.dumps(checkpoint_data, indent=2), encoding="utf-8"
        )
        logger.info(f"Checkpoint saved at index {current_index}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(output_filepath: str) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
    """Load progress from checkpoint if it exists."""
    checkpoint_file = Path("checkpoints") / f"{Path(output_filepath).name}.checkpoint"

    try:
        if checkpoint_file.exists():
            data: Dict[str, Any] = json.loads(
                checkpoint_file.read_text(encoding="utf-8")
            )
            logger.info(
                f"Loaded checkpoint, resuming from index {data['current_index']}"
            )
            return data["current_index"], data["data"]
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")

    return 0, None


@retry_on_429
def batch_check_accuracy_gemini(
    questions_data: List[Dict[str, Any]],
    client: Optional[genai.GenerativeModel] = None,
    max_output_tokens: int = 8192,
    max_batch_size: int = 10,
) -> List[Tuple[bool, str]]:
    if rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")

    try:
        questions_data = questions_data[:max_batch_size]

        if not questions_data:
            logger.error("No questions to process in batch")
            return []

        total_estimated_tokens = sum(
            len(str(item.get("question", "")).split())
            + len(str(item.get("correct_answer", "")).split())
            + 1000
            for item in questions_data
        )

        logger.info(
            f"Batch checking accuracy for {len(questions_data)} questions (Max Batch Size: {max_batch_size})"
        )
        logger.info(f"Estimated total tokens: {total_estimated_tokens}")

        # Add debug logging for the actual prompt
        prompt = """You are an AWS certification expert. Review each question and answer for accuracy.

CRITICAL FORMATTING REQUIREMENTS:
Your response MUST follow this EXACT format for each question:
{number}. [Accurate/Inaccurate]: {explanation}

Format Rules:
1. Start each line with the question number and a period
2. Add a single space after the period
3. Use square brackets around either "Accurate" or "Inaccurate"
4. Add a colon and a space after the closing bracket
5. Provide a clear explanation
6. Each response must be on its own line
7. No extra text or formatting

Example Valid Responses:
1. [Accurate]: The answer correctly explains AWS S3 as an object storage service.
2. [Inaccurate]: The response confuses EC2 instances with RDS databases.

Example Invalid Responses (DO NOT USE):
❌ 1) [Accurate] - Missing colon
❌ 1. accurate: Missing brackets
❌ [Accurate]: Missing question number
❌ 1. [Accurate] The answer is correct - Missing colon

Questions to evaluate:
{
            chr(10).join(
                f"{i + 1}. Q: {item.get('question', 'N/A')}\n   A: {item.get('correct_answer', 'N/A')}"
                for i, item in enumerate(questions_data)
            )
        }

IMPORTANT: 
- Each response MUST match the exact format shown in the valid examples
- Do not include any additional text or explanations
- Responses must be numbered sequentially starting from 1
- Verify your response format before submitting
"""

        rate_limiter.wait_if_needed(total_estimated_tokens)
        chat_session = MODEL.start_chat(history=[])
        response = chat_session.send_message(prompt)

        if not response or not response.text:
            logger.error("Received empty response from Gemini API")
            return [(False, "Empty API response") for _ in range(len(questions_data))]

        content = response.text.strip()
        logger.info(f"Batch accuracy check response received: {content[:500]}...")

        # Add detailed parsing debug logs
        logger.debug("Starting response parsing")
        logger.debug("Raw response content:\n{content}")

        # First try strict regex matching
        results: List[Tuple[bool, str]] = []
        lines = content.split("\n")

        # Log each line for debugging
        for i, line in enumerate(lines):
            logger.debug("Parsing line {i + 1}: {line}")
            match = re.match(
                r"^(\d+)\.\s*\[(Accurate|Inaccurate)\]:\s*(.+)", line, re.IGNORECASE
            )
            if match:
                index = int(match.group(1)) - 1
                is_accurate = match.group(2).lower() == "accurate"
                explanation = match.group(3).strip()
                logger.debug(
                    f"Matched line {i + 1}: index={index}, accurate={is_accurate}, explanation={explanation[:50]}..."
                )

                if 0 <= index < len(questions_data):
                    results.append((is_accurate, explanation))
            else:
                logger.debug("Line {i + 1} did not match expected format")

        # Log parsing results
        logger.debug(
            "Strict parsing found {len(results)} results out of {len(questions_data)} expected"
        )

        # If strict parsing fails, try lenient parsing
        if len(results) < len(questions_data):
            logger.warning("Strict parsing failed. Attempting lenient parsing.")
            logger.debug("Starting lenient parsing")
            results = []
            for line in lines:
                line = line.strip()
                if re.search(r"(Accurate|Inaccurate)", line, re.IGNORECASE):
                    is_accurate = "accurate" in line.lower()
                    logger.debug("Lenient parsing matched line: {line[:50]}...")
                    results.append((is_accurate, line))
                else:
                    logger.debug("Lenient parsing failed to match line: {line[:50]}...")

        # Pad results if necessary
        while len(results) < len(questions_data):
            logger.warning(
                "Missing results - padding with default values. Expected {len(questions_data)}, got {len(results)}"
            )
            results.append((False, "Failed to parse response"))

        rate_limiter.reset_consecutive_429s()
        return results

    except Exception:
        logger.error("Error during batch accuracy check: {str(e)}", exc_info=True)
        return [(False, "Error: {str(e)}") for _ in range(len(questions_data))]


@retry_on_429
def check_accuracy_gemini(
    question: str,
    correct_answer: str,
    client: Optional[genai.GenerativeModel] = None,
    max_output_tokens: int = 8192,
) -> Tuple[bool, str]:
    if rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")

    """Checks if the correct answer is indeed correct using the Gemini API."""
    # Estimate tokens: question + answer + prompt (~500) + max response (500)
    estimated_tokens = len(question.split()) + len(correct_answer.split()) + 1000
    logger.info("Checking accuracy for question: {question[:100]}...")

    prompt = """You are an AWS Certified Cloud Practitioner exam expert. Your task is to verify the accuracy of the given answer.

QUESTION: {question}
GIVEN ANSWER: {correct_answer}

INSTRUCTIONS:
1. Evaluate if the given answer is the BEST and most accurate response according to AWS Cloud Practitioner certification standards
2. Consider ONLY official AWS documentation and best practices
3. Ignore any minor formatting or grammatical issues
4. Focus solely on technical accuracy and completeness
5. Do not suggest alternative answers or improvements

REQUIRED RESPONSE FORMAT:
1. Brief explanation (1-2 sentences max)
2. Single word verdict: MUST be either "Accurate" or "Inaccurate"

DO NOT include any other text or explanations."""

    try:
        # Wait for rate limit with token estimation
        rate_limiter.wait_if_needed(estimated_tokens)

        # Start a new chat session
        chat_session = MODEL.start_chat(history=[])
        response = chat_session.send_message(prompt)

        content = response.text
        logger.debug("Received response: {content[:200]}...")

        # Reset consecutive 429s on successful request
        rate_limiter.reset_consecutive_429s()

        if "Inaccurate" in content:
            logger.info("Answer marked as inaccurate")
            return False, content
        else:
            logger.info("Answer marked as accurate")
            return True, content
    except Exception:
        logger.error("Error during accuracy check: {str(e)}", exc_info=True)
        raise


@retry_on_429
def generate_better_answer_gemini(
    question: str,
    correct_answer: str,
    client: Optional[genai.GenerativeModel] = None,
    max_output_tokens: int = 1024,
) -> str:
    if rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")

    """
    Generates a more detailed and comprehensive answer using the Gemini API.
    """
    # Estimate tokens: question + answer + prompt (~500) + max response (1024)
    estimated_tokens = len(question.split()) + len(correct_answer.split()) + 1524

    prompt = """You are an AWS Certified Cloud Practitioner exam expert. Your task is to provide a comprehensive explanation for the following question.

QUESTION: {question}
CURRENT ANSWER: {correct_answer}

REQUIRED RESPONSE STRUCTURE:
1. Core Explanation (2-3 sentences)
   - Why this is the correct answer
   - Key AWS concepts involved

2. Technical Context (2-3 bullet points)
   - Relevant AWS services
   - Service relationships/dependencies
   - Technical limitations or requirements

3. Real-world Application (1-2 sentences)
   - Business use case
   - Common implementation scenario

4. Exam Tips (2-3 bullet points)
   - Key points to remember
   - Common misconceptions to avoid
   - Related topics that might appear

STRICT REQUIREMENTS:
- Use ONLY official AWS terminology
- Focus on Cloud Practitioner level knowledge
- Keep explanations concise and clear
- Include specific service names where relevant
- Maintain professional, technical language
- Format response in HTML with proper paragraph and list tags
- Do not include personal opinions or non-AWS content
- Do not suggest alternative answers
- Do not include exam-taking strategies

RESPONSE FORMAT:
<div class="detailed-explanation">
    <p><strong>Core Explanation:</strong></p>
    [Your explanation here]
    
    <p><strong>Technical Context:</strong></p>
    <ul>
        [Your bullet points here]
    </ul>
    
    <p><strong>Real-world Application:</strong></p>
    [Your application example here]
    
    <p><strong>Exam Tips:</strong></p>
    <ul>
        [Your tips here]
    </ul>
</div>"""

    try:
        # Wait for rate limit with token estimation
        rate_limiter.wait_if_needed(estimated_tokens)

        # Start a new chat session
        chat_session = MODEL.start_chat(history=[])
        response = chat_session.send_message(prompt)

        # Reset consecutive 429s on successful request
        rate_limiter.reset_consecutive_429s()

        return str(response.text)
    except Exception:
        logger.error("Error during API call: {str(e)}", exc_info=True)
        raise


def find_duplicate_answers(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Find and log duplicate answers across the dataset
    """
    answer_counts: Dict[str, Dict[str, Any]] = {}
    duplicates: List[Dict[str, str]] = []

    for item in data:
        answer = item.get("correct_answer")
        if not answer:  # Skip None or empty answers
            continue

        if answer in answer_counts:
            duplicates.append(
                {
                    "question": item.get("question", ""),
                    "duplicate_of": str(answer_counts[answer]["question"]),
                }
            )
        else:
            answer_counts[answer] = {"question": item.get("question", ""), "count": 1}

    return duplicates


def normalize_question(question: Optional[str]) -> Tuple[str, str]:
    """
    Normalize question text to help identify duplicates while preserving original text
    Returns tuple of (normalized_text, original_text)
    """
    if not question:
        return "", ""

    # Store original text with only whitespace normalization
    original_text = re.sub(r"\s+", " ", question).strip()

    # Create normalized version for comparison (lowercase, no HTML)
    normalized = re.sub(r"<[^>]+>", "", original_text)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.lower()

    return normalized, original_text


def normalize_answer(answer: Optional[str]) -> Tuple[str, str]:
    """
    Normalize answer text to help identify duplicates while preserving original text
    Returns tuple of (normalized_text, original_text)
    """
    if not answer:
        return "", ""

    # Store original text with only whitespace normalization
    original_text = re.sub(r"\s+", " ", answer).strip()

    # Create normalized version for comparison (lowercase, no HTML)
    normalized = re.sub(r"<[^>]+>", "", original_text)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.lower()

    return normalized, original_text


def deduplicate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate questions and answers while preserving the best version
    and maintaining original capitalization
    """
    logger.info("Starting deduplication process...")

    # Create a dictionary to store unique questions
    unique_questions: Dict[str, Dict[str, Any]] = {}
    duplicates_removed = 0

    for item in data:
        # Normalize question and answer for comparison while keeping original text
        norm_question, orig_question = normalize_question(item.get("question", ""))
        norm_answer, orig_answer = normalize_answer(item.get("correct_answer", ""))

        # Create a unique key combining normalized question and answer
        unique_key = "{norm_question}|{norm_answer}"

        if unique_key in unique_questions:
            # If we already have this question, check which version is better
            existing_item = unique_questions[unique_key]

            # Prefer items with more complete data
            if len(item.get("options", [])) > len(
                existing_item.get("options", [])
            ) or len(item.get("correct_answer", "")) > len(
                existing_item.get("correct_answer", "")
            ):
                # Store the item with original capitalization
                unique_questions[unique_key] = item

            duplicates_removed += 1
        else:
            # Store the item with original capitalization
            unique_questions[unique_key] = item

    logger.info("Deduplication complete. Removed {duplicates_removed} duplicates.")
    return list(unique_questions.values())


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Comprehensive dataset validation with improved error handling and duplicate detection
    while preserving original capitalization
    """
    logger.info("Starting comprehensive dataset validation")

    if not data:
        logger.warning("Empty dataset provided for validation")
        return {
            "duplicate_answers": [],
            "missing_answers": 0,
            "total_items": 0,
            "duplicate_questions": 0,
        }

    # Track various issues
    missing_answers = 0
    duplicate_questions = 0
    seen_questions: Set[str] = set()

    # Check for duplicate answers
    duplicate_answers = find_duplicate_answers(data)

    # Check for missing answers and duplicates
    for item in data:
        # Check for missing answers
        answer = item.get("correct_answer")
        if answer is None:
            missing_answers += 1
            continue

        # Check for duplicate questions using normalized version for comparison
        norm_question, _ = normalize_question(item.get("question", ""))
        if norm_question in seen_questions:
            duplicate_questions += 1
        else:
            seen_questions.add(norm_question)

    # Log findings
    if duplicate_answers:
        logger.warning("Found {len(duplicate_answers)} duplicate answers")
    if missing_answers > 0:
        logger.warning("Found {missing_answers} missing or invalid answers")
    if duplicate_questions > 0:
        logger.warning("Found {duplicate_questions} duplicate questions")

    return {
        "duplicate_answers": duplicate_answers,
        "missing_answers": missing_answers,
        "duplicate_questions": duplicate_questions,
        "total_items": len(data),
    }


def write_output_file(data: List[Dict[str, Any]], output_filepath: str) -> int:
    """
    Write the processed data back to a file in a clean, consistent HTML format
    that is easy to parse and process.
    """
    logger.info("Writing output to {output_filepath}")

    # First, deduplicate the data
    deduplicated_data = deduplicate_data(data)
    logger.info("Writing {len(deduplicated_data)} unique questions to output file")

    def replace_url(match: re.Match) -> str:
        """Replace matched URLs with escaped versions"""
        url = match.group(0)
        return str(html.escape(url))

    def escape_urls(text: str) -> str:
        """Escape URLs in text to prevent HTML rendering issues"""
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        return str(re.sub(url_pattern, replace_url, text))

    with open(output_filepath, "w", encoding="utf-8") as f:
        # Write headers
        f.write("#separator:tab\n")
        f.write("#html:true\n")

        for item in deduplicated_data:
            try:
                # Clean and normalize the question text
                question_html = item.get(
                    "original_question", item.get("question", "")
                ).strip()
                question_html = escape_urls(question_html)

                # Ensure options are properly formatted
                options = item.get("options", [])
                if not options:
                    logger.warning(
                        "Missing options for question: {question_html[:100]}..."
                    )
                    continue

                # Build options HTML with consistent formatting
                options_html: List[str] = []
                correct_letters: List[str] = []
                for i, opt in enumerate(options):
                    letter = chr(65 + i)  # A, B, C, D, etc.
                    opt_text = opt.get("text", "").strip()
                    is_correct = opt.get("is_correct", False)

                    if not opt_text:
                        continue

                    if is_correct:
                        correct_letters.append(letter)

                    # Use consistent single quotes and no extra whitespace
                    # Remove the letter from opt_text if it starts with it
                    if opt_text.startswith("{letter}. "):
                        opt_text = opt_text[3:].strip()
                    elif opt_text.startswith("{letter}."):
                        opt_text = opt_text[2:].strip()

                    # Escape URLs in the option text
                    opt_text = escape_urls(opt_text)

                    # Include the letter prefix in the output
                    options_html.append(
                        "<li class='{('correct' if is_correct else '')}'>{letter}. {opt_text}</li>"
                    )

                if not options_html:
                    logger.warning("No valid options found for question, skipping...")
                    continue

                if not correct_letters:
                    logger.warning("No correct answers found for question, skipping...")
                    continue

                # Get the explanation if available
                explanation = ""
                if item.get("correct_answer") and isinstance(
                    item["correct_answer"], str
                ):
                    # Extract the explanation div content
                    explanation_match = re.search(
                        r'<div class="detailed-explanation">(.*?)</div>',
                        item["correct_answer"],
                        re.DOTALL,
                    )
                    if explanation_match:
                        # Clean up the explanation by removing markdown code block markers and html tag
                        explanation = explanation_match.group(0)
                        explanation = re.sub(
                            r"```html\s*", "", explanation
                        )  # Remove ```html
                        explanation = re.sub(r"```\s*", "", explanation)  # Remove ```
                        # Escape URLs in the explanation
                        explanation = escape_urls(explanation)

                # Create a single entry with both question and correct answer
                entry = f"""<div>
    <b>Question:</b><br>
    {question_html}
    <ul>
        {chr(10).join("        " + line for line in options_html)}
    </ul>
    <br><b>Correct Answer(s):</b> {", ".join(sorted(correct_letters))}
    {f"<br><b>Explanation(s):</b>\n{explanation}" if explanation else ""}
</div>"""

                # Write entry with tab separation and newline
                f.write(f"{entry}\n")

            except Exception:
                logger.error("Error writing question: {str(e)}")
                continue

    logger.info("Output written to {output_filepath}")
    return len(deduplicated_data)


def estimate_token_usage(
    data: List[Dict[str, Any]], batch_size: int = 10
) -> Dict[str, int]:
    """
    Estimate total token usage for the dataset based on Gemini 2.0 Flash model limits
    """
    # Model limits for Gemini 2.0 Flash
    MAX_COMBINED_TOKENS = 1_048_576  # Combined input+output token limit
    MAX_OUTPUT_TOKENS = 8_192  # Output token limit

    def estimate_item_tokens(item: Dict[str, Any]) -> Tuple[int, int]:
        """Estimate input and output tokens separately"""
        # Safely handle potentially missing or None values
        question = str(item.get("question", "") or "")
        correct_answer = str(item.get("correct_answer", "") or "")
        options = item.get("options", [])

        # Input tokens (question + options + prompt overhead)
        input_tokens = (
            len(question.split()) * 4  # Question text
            + sum(
                len(str(opt.get("text", "") or "").split()) * 4 for opt in options
            )  # Options
            + 500  # Prompt overhead
        )

        # Output tokens (expected response size + buffer)
        output_tokens = (
            len(correct_answer.split()) * 4  # Current answer length as baseline
            + 1000  # Buffer for generated content
        )

        return input_tokens, output_tokens

    # Calculate totals and per-batch estimates
    total_input_tokens = 0
    total_output_tokens = 0
    estimated_batches = math.ceil(len(data) / batch_size)

    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start : batch_start + batch_size]
        batch_input = 0
        batch_output = 0

        for item in batch:
            input_tokens, output_tokens = estimate_item_tokens(item)
            batch_input += input_tokens
            batch_output += output_tokens

        total_input_tokens += batch_input
        total_output_tokens += batch_output

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_batches": estimated_batches,
        "avg_input_tokens_per_batch": total_input_tokens / estimated_batches,
        "avg_output_tokens_per_batch": total_output_tokens / estimated_batches,
        "avg_combined_tokens_per_batch": (total_input_tokens + total_output_tokens)
        / estimated_batches,
        "max_combined_tokens": MAX_COMBINED_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def calculate_batch_statistics(
    items_processed: int,
    total_items: int,
    total_requests: int,
    total_tokens: int,
    rate_limit_hits: int,
    elapsed_time: float,
    current_backoff: float,
    rpm_limit: int = 2000,
    tpm_limit: int = 4_000_000,
) -> Dict[str, Any]:
    """Calculate comprehensive batch statistics"""

    # Basic calculations
    remaining_items = total_items - items_processed
    success_rate = (
        100.0 if total_requests == 0 else (1 - rate_limit_hits / total_requests) * 100
    )
    requests_per_second = total_requests / elapsed_time if elapsed_time > 0 else 0

    # Rate calculations
    current_rpm = min(int(requests_per_second * 60), rpm_limit)
    current_tpm = min(
        int(total_tokens / (elapsed_time / 60)) if elapsed_time > 0 else 0, tpm_limit
    )

    # Time estimation
    avg_time_per_item = elapsed_time / items_processed if items_processed > 0 else 0
    estimated_remaining_time = avg_time_per_item * remaining_items
    hours = int(estimated_remaining_time // 3600)
    minutes = int((estimated_remaining_time % 3600) // 60)
    seconds = int(estimated_remaining_time % 60)
    estimated_completion_time = f"{hours}h {minutes:02d}m {seconds:02d}s"

    return {
        "items_processed": items_processed,
        "total_items": total_items,
        "remaining_items": remaining_items,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "rate_limit_hits": rate_limit_hits,
        "success_rate": success_rate,
        "requests_per_second": requests_per_second,
        "elapsed_time": int(elapsed_time),
        "current_rpm": current_rpm,
        "rpm_limit": rpm_limit,
        "rpm_percentage": (current_rpm / rpm_limit) * 100,
        "current_tpm": current_tpm,
        "tpm_limit": tpm_limit,
        "tpm_percentage": (current_tpm / tpm_limit) * 100,
        "current_backoff": current_backoff,
        "estimated_completion_time": estimated_completion_time,
    }


class BatchProcessingStats:
    def __init__(self, batch_size: int = 10):
        self.parsing_errors = []  # List to store details about parsing errors
        self.total_strict_parse_failures = 0
        self.total_lenient_parse_successes = 0
        self.total_parse_failures = 0
        self.batch_size = batch_size  # Store batch size as instance variable

    def record_parsing_error(self, batch_index: int, error_type: str, details: str):
        """Record details about parsing errors"""
        self.parsing_errors.append(
            {
                "batch_index": batch_index,
                "error_type": error_type,
                "details": details,
                "items": "Items {batch_index * self.batch_size + 1} to {min((batch_index + 1) * self.batch_size, self.total_items)}",
            }
        )

        if error_type == "strict_parse_failure":
            self.total_strict_parse_failures += 1
        elif error_type == "complete_parse_failure":
            self.total_parse_failures += 1
        elif error_type == "lenient_parse_success":
            self.total_lenient_parse_successes += 1


def log_final_statistics(stats: BatchProcessingStats):
    """Log comprehensive final statistics including parsing errors"""
    logger.info("""
Processing Complete:
    Parse Error Summary:
    - Total Strict Parse Failures: {stats.total_strict_parse_failures}
    - Successfully Recovered (Lenient): {stats.total_lenient_parse_successes}
    - Complete Parse Failures: {stats.total_parse_failures}
    
    Detailed Parse Error Log:""")

    if stats.parsing_errors:
        for error in stats.parsing_errors:
            logger.info("""    
    Batch {error["batch_index"]} ({error["items"]}):
    - Error Type: {error["error_type"]}
    - Details: {error["details"]}""")
    else:
        logger.info("    No parsing errors encountered")


def log_batch_progress(stats: Dict[str, Any]) -> None:
    """Log batch processing progress statistics"""
    logger.info(
        """
Batch Progress:
- Items Processed: {stats["items_processed"]}/{stats["total_items"]} ({stats["items_processed"] / stats["total_items"] * 100:.1f}%))
- Requests: {stats["total_requests"]} (Rate: {stats["requests_per_second"]:.2f}/s)
- Rate Limits Hit: {stats["rate_limit_hits"]}
- Current RPM: {stats["current_rpm"]}/{stats["rpm_limit"]} ({stats["rpm_percentage"]:.1f}%))
- Current TPM: {stats["current_tpm"]}/{stats["tpm_limit"]} ({stats["tpm_percentage"]:.1f}%))
- Elapsed Time: {stats["elapsed_time"]}s
- Est. Completion: {stats["estimated_completion_time"]}
""".strip()
    )


def process_batch_data(
    data: List[Dict[str, Any]],
    client: Optional[genai.GenerativeModel] = None,
    batch_size: int = 5,
    output_filepath: str = "corrected_data.json",
    selected_questions: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Process data in batches with optional question selection"""
    stats = BatchProcessingStats(batch_size=batch_size)

    # If processing selected questions, adjust total items
    if selected_questions:
        stats.total_items = len(selected_questions)
        data_to_process = [
            data[i - 1] for i in selected_questions
        ]  # Convert to 0-based index
    else:
        stats.total_items = len(data)
        data_to_process = data

    try:
        start_time = time.time()
        items_processed = 0
        total_requests = 0
        total_tokens = 0
        rate_limit_hits = 0
        processed_data = []
        estimated_tokens = 0  # Initialize here to avoid undefined variable error

        for batch_start in range(0, len(data_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(data_to_process))
            batch = data_to_process[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // batch_size + 1} (Items {batch_start + 1} to {batch_end})"
            )

            try:
                for idx, item in enumerate(batch):
                    current_item = item.copy()
                    estimated_tokens = len(str(item)) // 4  # Rough estimate

                    # Check if this is a multiple-answer question
                    if is_multiple_answer_question(item["question"]):
                        logger.info(
                            f"Detected multiple-answer question at index {batch_start + idx}"
                        )

                        # Process multiple answer question
                        correct_letters, explanation = process_multiple_answer_question(
                            item, client
                        )

                        if correct_letters:
                            current_correct = [
                                chr(65 + i)
                                for i, opt in enumerate(item["options"])
                                if opt.get("is_correct", False)
                            ]

                            if set(correct_letters) != set(current_correct):
                                logger.info(
                                    f"Updating multiple answers for question {batch_start + idx}: Original: {current_correct} -> New: {correct_letters}"
                                )

                                current_item = update_question_with_multiple_answers(
                                    current_item, correct_letters, explanation
                                )

                    processed_data.append(current_item)
                    items_processed += 1
                    total_requests += 1
                    total_tokens += estimated_tokens

                # Calculate and log progress only if we have processed items
                if items_processed > 0:
                    elapsed_time = time.time() - start_time
                    batch_stats = calculate_batch_statistics(
                        items_processed=items_processed,
                        total_items=stats.total_items,
                        total_requests=total_requests,
                        total_tokens=total_tokens,
                        rate_limit_hits=rate_limit_hits,
                        elapsed_time=elapsed_time,
                        current_backoff=rate_limiter.min_request_interval
                        if rate_limiter
                        else 0,
                    )

                    # Log progress
                    log_batch_progress(batch_stats)

                # Save checkpoint
                save_checkpoint(processed_data, output_filepath, batch_end - 1)

            except Exception as batch_error:
                logger.error(
                    f"Error processing batch starting at index {batch_start}: {str(batch_error)}"
                )
                rate_limit_hits += 1
                # Add the batch to processed data without changes on error
                processed_data.extend(batch)
                continue

        # Log final statistics
        log_final_statistics(stats)

        return processed_data

    except Exception:
        logger.error("Error processing batch data: {str(e)}")
        logger.error(traceback.format_exc())
        return processed_data


def prepare_accuracy_check_prompt(batch: List[Dict[str, Any]]) -> str:
    """
    Prepare a prompt for checking the accuracy of a batch of questions.

    Args:
        batch: List of question dictionaries to check

    Returns:
        str: Formatted prompt for the model
    """
    prompt_parts = [
        "Review each AWS certification practice question and answer for accuracy.",
        "For each question, indicate if the answer is [Accurate] or [Inaccurate], and explain why.",
        "If inaccurate, explain what needs to be corrected.\n\n",
    ]

    for i, item in enumerate(batch, 1):
        prompt_parts.append("{i}. Question: {item['question']}\n")
        prompt_parts.append("   Answer: {item['correct_answer']}\n")
        if "explanation" in item and item["explanation"]:
            prompt_parts.append("   Explanation: {item['explanation']}\n")
        prompt_parts.append("\n")

    prompt_parts.append("\nPlease review each question and answer pair above.")
    prompt_parts.append(
        "Format your response as a numbered list matching the questions."
    )
    prompt_parts.append(
        "For each item, start with either [Accurate] or [Inaccurate] followed by your explanation."
    )

    return "\n".join(prompt_parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Using a simple approximation of 4 characters per token.

    Args:
        text: String to estimate tokens for

    Returns:
        int: Estimated number of tokens
    """
    return len(text) // 4


def await_response(response_future: Any) -> Any:
    """
    Helper function to handle response from Gemini API.
    Wraps the response handling to provide consistent error handling.
    """
    try:
        response = response_future
        if not response or not response.text:
            raise ValueError("Empty response received from API")
        return response
    except Exception:
        logger.error("Error getting response from API: {str(e)}")
        raise


def parse_question_list(question_str: str) -> List[int]:
    """Parse a comma-delimited string of question numbers into a list of integers."""
    try:
        # Split by comma and convert to integers
        return [int(q.strip()) for q in question_str.split(",")]
    except ValueError:
        logger.error("Invalid question number format: {str(e)}")
        raise ValueError(
            "Question numbers must be comma-separated integers (e.g., '1,2,3')"
        )


def is_multiple_answer_question(question: str) -> bool:
    """
    Detect if a question requires multiple answers.
    Currently checks for explicit "Choose TWO" text.

    # TODO: Future improvement - could analyze all answers and ask LLM
    # if any indicate multiple required answers
    """
    return "Choose TWO" in question


@retry_on_429
def process_multiple_answer_question(
    question: Dict[str, Any], client: Optional[genai.GenerativeModel] = None
) -> Tuple[List[str], Optional[str]]:
    """
    Process a question that requires multiple answers.
    Returns tuple of (correct_answer_letters, optional_explanation)
    """
    prompt = """
Please analyze this AWS certification multiple-answer question and determine the correct answers.

CRITICAL FORMATTING REQUIREMENTS:
Your response MUST follow this EXACT format for each question:
{number}. [Accurate/Inaccurate]: {explanation}

Format Rules:
1. Start each line with the question number and a period
2. Add a single space after the period
3. Use square brackets around either "Accurate" or "Inaccurate"
4. Add a colon and a space after the closing bracket
5. Provide a clear explanation
6. Each response must be on its own line
7. No extra text or formatting

Example Valid Responses:
1. [Accurate]: The answer correctly explains AWS S3 as an object storage service.
2. [Inaccurate]: The response confuses EC2 instances with RDS databases.

Example Invalid Responses (DO NOT USE):
❌ 1) [Accurate] - Missing colon
❌ 1. accurate: Missing brackets
❌ [Accurate]: Missing question number
❌ 1. [Accurate] The answer is correct - Missing colon

Questions to evaluate:
{
            chr(10).join(
                f"{i + 1}. Q: {item.get('question', 'N/A')}\n   A: {item.get('correct_answer', 'N/A')}"
                for i, item in enumerate(question["options"])
            )
        }

IMPORTANT: 
- Each response MUST match the exact format shown in the valid examples
- Do not include any additional text or explanations
- Responses must be numbered sequentially starting from 1
- Verify your response format before submitting
"""

    try:
        chat = MODEL.start_chat(history=[])
        response = chat.send_message(prompt)

        # Parse the response for the two letters
        answer_match = re.search(
            r"Correct Answers:\s*\[([A-E]),\s*([A-E])\]", response.text
        )
        if not answer_match:
            logger.warning(
                "Could not parse multiple answers from response: {response.text}"
            )
            return [], None

        letters = sorted([answer_match.group(1), answer_match.group(2)])

        # Extract explanation if present
        explanation_match = re.search(
            r"Explanation:\s*(.+?)(?=\n|$)", response.text, re.DOTALL
        )
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return letters, explanation

    except Exception:
        logger.error("Error processing multiple answer question: {str(e)}")
        return [], None


def update_question_with_multiple_answers(
    question: Dict[str, Any], correct_letters: List[str], explanation: Optional[str]
) -> Dict[str, Any]:
    """
    Update a question dict with new multiple correct answers
    """
    updated_question = question.copy()

    # Update the options
    for i, opt in enumerate(updated_question["options"]):
        letter = chr(65 + i)
        opt["is_correct"] = letter in correct_letters

    # If we have an explanation, update or add it
    if explanation:
        if "correct_answer" in updated_question:
            # Try to preserve existing explanation div structure
            explanation_div = """<div class="detailed-explanation">
    <p><strong>Core Explanation:</strong></p>
    {explanation}
</div>"""
            updated_question["correct_answer"] = explanation_div

    return updated_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process and correct AWS exam questions using AI-powered analysis.",
        epilog="Example: python scan-correct.py -i input.txt -o output.txt --tier paid_tier_1 --batch-size 10",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input text file containing exam questions (HTML-like format)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save the corrected output file"
    )
    parser.add_argument(
        "--tier",
        choices=["free", "paid_tier_1"],
        default="free",
        help="Google AI API tier (default: free). Paid tier allows more requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to process in each batch (recommended: 5-20, depending on API limits)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of questions to process (useful for testing)",
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Process specific question numbers (comma-delimited, 1-based index, e.g., '1,2,3')",
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize rate limiter with specified tier
    global rate_limiter
    rate_limiter = RateLimiter(tier=args.tier)

    # Load and process data
    data = load_data(args.input)

    # Handle specific questions if specified
    if args.question is not None:
        try:
            question_numbers = parse_question_list(args.question)
            # Validate question numbers
            invalid_numbers = [q for q in question_numbers if q < 1 or q > len(data)]
            if invalid_numbers:
                logger.error(
                    f"Invalid question numbers: {invalid_numbers}. Valid range: 1-{len(data)}"
                )
                return

            logger.info("Processing questions: {question_numbers}")
            # Convert 1-based indices to 0-based and get the questions
            data = [data[q - 1] for q in question_numbers]
        except ValueError as e:
            logger.error(str(e))
            return
    elif args.limit:
        data = data[: args.limit]

    # Process the data using the MODEL instead of client
    processed_data = process_batch_data(
        data=data,
        client=MODEL,  # Pass the model instance
        batch_size=args.batch_size,
        output_filepath=args.output,
    )

    # Write output
    write_output_file(processed_data, args.output)

    # Validate the processed dataset
    validation_results = validate_dataset(processed_data)

    # Log summary of updates and validation
    logger.info("\n--- Processing Summary ---")
    logger.info("Total questions processed: {len(processed_data)}")

    # Count updated answers
    updated_answers = [
        item for item in processed_data if item.get("original_correct_answer")
    ]
    logger.info("Number of answers updated: {len(updated_answers)}")

    # Log validation results
    if validation_results["duplicate_answers"]:
        logger.warning(
            "Duplicate answers found: {len(validation_results['duplicate_answers'])}"
        )

    if validation_results["missing_answers"] > 0:
        logger.warning("Missing answers found: {validation_results['missing_answers']}")

    # Optional: Log details of updated answers
    if updated_answers:
        logger.info("\nUpdated Answers Details:")
        for item in updated_answers[:10]:  # Log first 10 updated answers
            logger.info("Question: {item['question'][:100]}...")
            logger.info("Original Answer: {item.get('original_correct_answer')}")
            logger.info("Updated Answer: {item['correct_answer']}\n")


if __name__ == "__main__":
    main()
