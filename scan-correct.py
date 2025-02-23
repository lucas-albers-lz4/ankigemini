import argparse
import re
import logging
import json
import time
import google.generativeai as genai
from google.genai.errors import ClientError
import random
import threading
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path

# Suppress GRPC shutdown warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scan_correct.log'),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

# Configuration
logger.info("Initializing Gemini client...")
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    logger.error("No API key found. Please set GEMINI_API_KEY in .env file.")
    raise ValueError("Missing GEMINI_API_KEY environment variable")

# Configure the API
genai.configure(api_key=API_KEY)
logger.info("Gemini client configured successfully")

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.0-flash')

# Generation configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
try:
    MODEL = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=GENERATION_CONFIG
    )
    logger.info(f"Initialized Gemini model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Model Initialization Error: {str(e)}")
    raise

# Rate limit tiers
RATE_LIMIT_TIERS = {
    'free': {
        'rpm': 15,
        'tpm': 1_000_000,
        'rpd': 1_500
    },
    'paid_tier_1': {
        'rpm': 2_000,
        'tpm': 4_000_000,
        'rpd': None  # No daily limit for paid tier 1
    }
}

# Rate limiting configuration
class RateLimiter:
    def __init__(self, tier='free'):
        limits = RATE_LIMIT_TIERS[tier]
        self.max_rpm = limits['rpm']
        self.max_tpm = limits['tpm']
        self.max_rpd = limits['rpd']
        self.tier = tier
        
        self.minute_requests = []
        self.day_requests = []
        self.minute_tokens = []
        self.last_request_time = None
        self.lock = threading.Lock()
        
        # Rate tracking
        self.current_rpm = 0
        self.current_tpm = 0
        self.current_rpd = 0
        self.last_rate_check = datetime.now()
        
        # Limit tracking
        self.rpm_limit_hits = 0
        self.tpm_limit_hits = 0
        self.quota_limit_hits = 0
        
        # Increased minimum time between requests
        self.base_request_interval = 2.5  # Default to 2.5 seconds for all tiers
        self.min_request_interval = self.base_request_interval
        
        # Preserve existing dynamic backoff factor for current run
        self.run_start_time = datetime.now()
        self.consecutive_429s = 0
        self.last_429_time = None
        self.backoff_multiplier = 1.0
        
        logger.info(f"Rate limiter initialized for {tier} tier with: {self.max_rpm} RPM, {self.max_tpm} TPM, " + 
                   (f"{self.max_rpd} RPD" if self.max_rpd else "No RPD limit"))

    def reset_backoff(self):
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

    def _update_current_rates(self):
        """Update current rate measurements"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        # Clean old entries
        self.minute_requests = [req for req in self.minute_requests if req > minute_ago]
        self.minute_tokens = [(time, tokens) for time, tokens in self.minute_tokens if time > minute_ago]
        if self.max_rpd:
            self.day_requests = [req for req in self.day_requests if req > day_ago]
        
        # Update current rates
        self.current_rpm = len(self.minute_requests)
        self.current_tpm = sum(tokens for _, tokens in self.minute_tokens)
        self.current_rpd = len(self.day_requests) if self.max_rpd else 0
        self.last_rate_check = now

    def _diagnose_rate_limit(self):
        """Diagnose which limit is likely being hit"""
        self._update_current_rates()
        
        rpm_percentage = (self.current_rpm / self.max_rpm) * 100 if self.max_rpm else 0
        tpm_percentage = (self.current_tpm / self.max_tpm) * 100 if self.max_tpm else 0
        rpd_percentage = (self.current_rpd / self.max_rpd) * 100 if self.max_rpd else 0
        
        diagnosis = []
        if rpm_percentage > 80:
            diagnosis.append(f"RPM at {rpm_percentage:.1f}% ({self.current_rpm}/{self.max_rpm})")
        if tpm_percentage > 80:
            diagnosis.append(f"TPM at {tpm_percentage:.1f}% ({self.current_tpm}/{self.max_tpm})")
        if rpd_percentage > 80:
            diagnosis.append(f"RPD at {rpd_percentage:.1f}% ({self.current_rpd}/{self.max_rpd})")
        
        return diagnosis

    def _handle_429(self):
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
        logger.warning(f"Adjusting minimum request interval to {self.min_request_interval:.2f}s after {self.consecutive_429s} consecutive 429s")

    def reset_consecutive_429s(self):
        """Reset consecutive 429s tracking on successful requests"""
        self.consecutive_429s = 0
        self.backoff_multiplier = 1.0
        self.min_request_interval = self.base_request_interval
        logger.info(f"Reset consecutive 429s tracking. Restoring base request interval to {self.min_request_interval:.2f}s")

    def _wait_for_interval(self):
        """Ensure minimum time between requests with dynamic backoff"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                logger.debug(f"Waiting {sleep_time:.2f}s between requests (backoff multiplier: {self.backoff_multiplier:.2f})")
                time.sleep(sleep_time)

    def wait_if_needed(self, estimated_tokens=1000):
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
                    sleep_time = (self.minute_requests[0] + timedelta(minutes=1) - now).total_seconds()
                    if sleep_time > 0:
                        logger.warning(f"RPM limit reached ({rpm_current}/{self.max_rpm}), waiting {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                elif rpm_current >= self.max_rpm * 0.8:  # Warning at 80% of limit
                    logger.warning(f"RPM approaching limit ({rpm_current}/{self.max_rpm})")
                
                # Check RPD only if there's a daily limit
                if self.max_rpd:
                    rpd_current = len(self.day_requests)
                    if rpd_current >= self.max_rpd:
                        sleep_time = (self.day_requests[0] + timedelta(days=1) - now).total_seconds()
                        if sleep_time > 0:
                            logger.warning(f"RPD limit reached ({rpd_current}/{self.max_rpd}), waiting {sleep_time:.2f} seconds")
                            time.sleep(sleep_time)
                    elif rpd_current >= self.max_rpd * 0.8:  # Warning at 80% of limit
                        logger.warning(f"RPD approaching limit ({rpd_current}/{self.max_rpd})")
                
                # Check TPM
                current_tpm = sum(tokens for _, tokens in self.minute_tokens)
                if current_tpm + estimated_tokens > self.max_tpm:
                    sleep_time = (self.minute_tokens[0][0] + timedelta(minutes=1) - now).total_seconds()
                    if sleep_time > 0:
                        logger.warning(f"TPM limit reached ({current_tpm}/{self.max_tpm}), waiting {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                elif current_tpm >= self.max_tpm * 0.8:  # Warning at 80% of limit
                    logger.warning(f"TPM approaching limit ({current_tpm}/{self.max_tpm})")
                
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
rate_limiter = None

# Function to load the data
def load_data(filepath):
    """
    Parse the exam questions file into a structured JSON-like format.
    """
    logger.info(f"Loading data from {filepath}")
    
    questions = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex pattern to extract questions and answers
    pattern = r'<div>\s*<b>Question:</b><br>\s*(.*?)\s*<ul>(.*?)</ul>.*?<br><b>Correct Answer\(s\):</b>\s*([A-Z,\s]+)\s*</div>'
    
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    
    for match in matches:
        question_text = match[0].strip()
        options_html = match[1]
        correct_answers = match[2].strip().split(',')
        
        # Extract options
        option_pattern = r'<li(?:\s+class=\'.*?\')?>([^<]+)</li>'
        options = re.findall(option_pattern, options_html)
        
        # Prepare options with correct flag
        formatted_options = []
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, etc.
            is_correct = letter in correct_answers
            formatted_options.append({
                'text': option.strip(),
                'is_correct': is_correct
            })
        
        question = {
            'question': question_text,
            'options': formatted_options,
            'correct_answer': next((opt['text'] for opt in formatted_options if opt['is_correct']), None)
        }
        
        questions.append(question)
    
    logger.info(f"Loaded {len(questions)} questions")
    return questions

def exponential_backoff(attempt, max_delay=120):  # Increased max delay
    """Calculate exponential backoff time with jitter"""
    base_delay = min(max_delay, (2 ** attempt) * 1.25)  # More aggressive base delay
    jitter = random.uniform(0, min(max_delay - base_delay, base_delay))
    delay = base_delay + jitter
    return delay

def retry_on_429(func):
    """Decorator to retry on 429 errors with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Wait if we're approaching rate limits
                rate_limiter.wait_if_needed()
                return func(*args, **kwargs)
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(f"Max retry attempts ({max_attempts}) reached. Giving up.")
                        raise
                    
                    # Update rate limiter's 429 tracking
                    rate_limiter._handle_429()
                    
                    delay = exponential_backoff(attempt)
                    logger.warning(f"Rate limit hit. Waiting {delay:.2f} seconds before retry {attempt}/{max_attempts}")
                    time.sleep(delay)
                else:
                    logger.error(f"Unexpected ClientError: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise
    return wrapper

def save_checkpoint(data, output_filepath, current_index):
    """Save current progress to a checkpoint file."""
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f"{Path(output_filepath).name}.checkpoint"
    checkpoint_data = {
        'current_index': current_index,
        'data': data[:current_index + 1]
    }
    
    try:
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2), encoding='utf-8')
        logger.info(f"Checkpoint saved at index {current_index}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(output_filepath):
    """Load progress from checkpoint if it exists."""
    checkpoint_file = Path('checkpoints') / f"{Path(output_filepath).name}.checkpoint"
    
    try:
        if checkpoint_file.exists():
            data = json.loads(checkpoint_file.read_text(encoding='utf-8'))
            logger.info(f"Loaded checkpoint, resuming from index {data['current_index']}")
            return data['current_index'], data['data']
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    
    return 0, None

@retry_on_429
def batch_check_accuracy_gemini(questions_data, client=None, max_output_tokens=8192, max_batch_size=10):
    try:
        questions_data = questions_data[:max_batch_size]
        
        if not questions_data:
            logger.error("No questions to process in batch")
            return []

        total_estimated_tokens = sum(
            len(str(item.get('question', '')).split()) + len(str(item.get('correct_answer', '')).split()) + 1000 
            for item in questions_data
        )
        
        logger.info(f"Batch checking accuracy for {len(questions_data)} questions (Max Batch Size: {max_batch_size})")
        logger.info(f"Estimated total tokens: {total_estimated_tokens}")
        
        # Add debug logging for the actual prompt
        prompt = f"""
        Batch Accuracy Verification for AWS Cloud Practitioner Questions

        Carefully evaluate each question and its current answer.
        Respond EXACTLY in this format for EACH question:
        1. [Accurate/Inaccurate]: Concise explanation
        2. [Accurate/Inaccurate]: Concise explanation
        ...and so on.

        Questions to evaluate:
        {chr(10).join(
            f"{i+1}. Q: {item.get('question', 'N/A')}\n   A: {item.get('correct_answer', 'N/A')}" 
            for i, item in enumerate(questions_data)
        )}
        
        IMPORTANT: Provide numbered responses matching the question numbers above.
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
        logger.debug(f"Raw response content:\n{content}")

        # First try strict regex matching
        results = []
        lines = content.split('\n')
        
        # Log each line for debugging
        for i, line in enumerate(lines):
            logger.debug(f"Parsing line {i+1}: {line}")
            match = re.match(r'^(\d+)\.\s*\[(Accurate|Inaccurate)\]:\s*(.+)', line, re.IGNORECASE)
            if match:
                index = int(match.group(1)) - 1
                is_accurate = match.group(2).lower() == 'accurate'
                explanation = match.group(3).strip()
                logger.debug(f"Matched line {i+1}: index={index}, accurate={is_accurate}, explanation={explanation[:50]}...")
                
                if 0 <= index < len(questions_data):
                    results.append((is_accurate, line))
            else:
                logger.debug(f"Line {i+1} did not match expected format")
        
        # Log parsing results
        logger.debug(f"Strict parsing found {len(results)} results out of {len(questions_data)} expected")
        
        # If strict parsing fails, try lenient parsing
        if len(results) < len(questions_data):
            logger.warning("Strict parsing failed. Attempting lenient parsing.")
            logger.debug("Starting lenient parsing")
            results = []
            for line in lines:
                line = line.strip()
                if re.search(r'(Accurate|Inaccurate)', line, re.IGNORECASE):
                    is_accurate = 'accurate' in line.lower()
                    logger.debug(f"Lenient parsing matched line: {line[:50]}...")
                    results.append((is_accurate, line))
                else:
                    logger.debug(f"Lenient parsing failed to match line: {line[:50]}...")
        
        # Pad results if necessary
        while len(results) < len(questions_data):
            logger.warning(f"Missing results - padding with default values. Expected {len(questions_data)}, got {len(results)}")
            results.append((False, "Failed to parse response"))
        
        rate_limiter.reset_consecutive_429s()
        return results

    except Exception as e:
        logger.error(f"Error during batch accuracy check: {str(e)}", exc_info=True)
        return [(False, f"Error: {str(e)}") for _ in range(len(questions_data))]

@retry_on_429
def check_accuracy_gemini(question, correct_answer, client=None, max_output_tokens=8192):
    """Checks if the correct answer is indeed correct using the Gemini API."""
    # Estimate tokens: question + answer + prompt (~500) + max response (500)
    estimated_tokens = len(question.split()) + len(correct_answer.split()) + 1000
    logger.info(f"Checking accuracy for question: {question[:100]}...")
    
    prompt = f"""You are an AWS Certified Cloud Practitioner exam expert. Your task is to verify the accuracy of the given answer.

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
        logger.debug(f"Received response: {content[:200]}...")

        # Reset consecutive 429s on successful request
        rate_limiter.reset_consecutive_429s()

        if "Inaccurate" in content:
            logger.info("Answer marked as inaccurate")
            return False, content
        else:
            logger.info("Answer marked as accurate")
            return True, content
    except Exception as e:
        logger.error(f"Error during accuracy check: {str(e)}", exc_info=True)
        raise

@retry_on_429
def generate_better_answer_gemini(question, correct_answer, client=None, max_output_tokens=1024):
    """
    Generates a more detailed and comprehensive answer using the Gemini API.
    """
    # Estimate tokens: question + answer + prompt (~500) + max response (1024)
    estimated_tokens = len(question.split()) + len(correct_answer.split()) + 1524
    
    prompt = f"""You are an AWS Certified Cloud Practitioner exam expert. Your task is to provide a comprehensive explanation for the following question.

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
        
        return response.text
    except Exception as e:
        logger.error(f"Error during API call: {str(e)}", exc_info=True)
        raise

def find_duplicate_answers(data):
    """
    Find and log duplicate answers across the dataset
    """
    answer_counts = {}
    duplicates = []
    
    for item in data:
        answer = item.get('correct_answer')
        if not answer:  # Skip None or empty answers
            continue
            
        if answer in answer_counts:
            duplicates.append({
                'question': item.get('question', ''),
                'duplicate_of': answer_counts[answer]['question']
            })
        else:
            answer_counts[answer] = {
                'question': item.get('question', ''),
                'count': 1
            }
    
    return duplicates

def normalize_question(question):
    """
    Normalize question text to help identify duplicates while preserving original text
    Returns tuple of (normalized_text, original_text)
    """
    if not question:
        return "", ""
    
    # Store original text with only whitespace normalization
    original_text = re.sub(r'\s+', ' ', question).strip()
    
    # Create normalized version for comparison (lowercase, no HTML)
    normalized = re.sub(r'<[^>]+>', '', original_text)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.lower()
    
    return normalized, original_text

def normalize_answer(answer):
    """
    Normalize answer text to help identify duplicates while preserving original text
    Returns tuple of (normalized_text, original_text)
    """
    if not answer:
        return "", ""
    
    # Store original text with only whitespace normalization
    original_text = re.sub(r'\s+', ' ', answer).strip()
    
    # Create normalized version for comparison (lowercase, no HTML)
    normalized = re.sub(r'<[^>]+>', '', original_text)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.lower()
    
    return normalized, original_text

def deduplicate_data(data):
    """
    Remove duplicate questions and answers while preserving the best version
    and maintaining original capitalization
    """
    logger.info("Starting deduplication process...")
    
    # Create a dictionary to store unique questions
    unique_questions = {}
    duplicates_removed = 0
    
    for item in data:
        # Normalize question and answer for comparison while keeping original text
        norm_question, orig_question = normalize_question(item.get('question', ''))
        norm_answer, orig_answer = normalize_answer(item.get('correct_answer', ''))
        
        # Create a unique key combining normalized question and answer
        unique_key = f"{norm_question}|{norm_answer}"
        
        if unique_key in unique_questions:
            # If we already have this question, check which version is better
            existing_item = unique_questions[unique_key]
            
            # Prefer items with more complete data
            if (len(item.get('options', [])) > len(existing_item.get('options', [])) or
                    len(item.get('correct_answer', '')) > len(existing_item.get('correct_answer', ''))):
                # Store the item with original capitalization
                unique_questions[unique_key] = item
            
            duplicates_removed += 1
        else:
            # Store the item with original capitalization
            unique_questions[unique_key] = item
    
    logger.info(f"Deduplication complete. Removed {duplicates_removed} duplicates.")
    return list(unique_questions.values())

def validate_dataset(data):
    """
    Comprehensive dataset validation with improved error handling and duplicate detection
    while preserving original capitalization
    """
    logger.info("Starting comprehensive dataset validation")
    
    if not data:
        logger.warning("Empty dataset provided for validation")
        return {
            'duplicate_answers': [],
            'missing_answers': 0,
            'total_items': 0,
            'duplicate_questions': 0
        }
    
    # Track various issues
    missing_answers = 0
    duplicate_questions = 0
    seen_questions = set()
    
    # Check for duplicate answers
    duplicate_answers = find_duplicate_answers(data)
    
    # Check for missing answers and duplicates
    for item in data:
        # Check for missing answers
        answer = item.get('correct_answer')
        if answer is None:
            missing_answers += 1
            continue
        
        # Check for duplicate questions using normalized version for comparison
        norm_question, _ = normalize_question(item.get('question', ''))
        if norm_question in seen_questions:
            duplicate_questions += 1
        else:
            seen_questions.add(norm_question)
    
    # Log findings
    if duplicate_answers:
        logger.warning(f"Found {len(duplicate_answers)} duplicate answers")
    if missing_answers > 0:
        logger.warning(f"Found {missing_answers} missing or invalid answers")
    if duplicate_questions > 0:
        logger.warning(f"Found {duplicate_questions} duplicate questions")
    
    return {
        'duplicate_answers': duplicate_answers,
        'missing_answers': missing_answers,
        'duplicate_questions': duplicate_questions,
        'total_items': len(data)
    }

def write_output_file(data, output_filepath):
    """
    Write the processed data back to a file in a clean, consistent HTML format
    that is easy to parse and process.
    """
    logger.info(f"Writing output to {output_filepath}")
    
    # First, deduplicate the data
    deduplicated_data = deduplicate_data(data)
    logger.info(f"Writing {len(deduplicated_data)} unique questions to output file")
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # Write headers
        f.write("#separator:tab\n")
        f.write("#html:true\n")
        
        for item in deduplicated_data:
            try:
                # Clean and normalize the question text
                question_html = item.get('original_question', item.get('question', '')).strip()
                
                # Ensure options are properly formatted
                options = item.get('options', [])
                if not options:
                    logger.warning(f"Missing options for question: {question_html[:100]}...")
                    continue
                
                # Build options HTML with consistent formatting
                options_html = []
                correct_letters = []
                for i, opt in enumerate(options):
                    letter = chr(65 + i)  # A, B, C, D, etc.
                    opt_text = opt.get('text', '').strip()
                    is_correct = opt.get('is_correct', False)
                    
                    if not opt_text:
                        continue
                        
                    if is_correct:
                        correct_letters.append(letter)
                        
                    # Use consistent single quotes and no extra whitespace
                    # Remove the letter from opt_text if it starts with it
                    if opt_text.startswith(f"{letter}. "):
                        opt_text = opt_text[3:].strip()
                    elif opt_text.startswith(f"{letter}."):
                        opt_text = opt_text[2:].strip()
                        
                    options_html.append(f"<li class='{('correct' if is_correct else '')}'>{letter}. {opt_text}</li>")
                
                if not options_html:
                    logger.warning("No valid options found for question, skipping...")
                    continue
                
                if not correct_letters:
                    logger.warning("No correct answers found for question, skipping...")
                    continue
                
                # Create the question entry with consistent formatting
                question_entry = f"""<div>
    <b>Question:</b><br>
    {question_html}
    <ul>
        {chr(10).join('        ' + line for line in options_html)}
    </ul>
</div>"""
                
                # Create the answer entry with consistent formatting
                answer_entry = f"""<div>
    <b>Question:</b><br>
    {question_html}
    <ul>
        {chr(10).join('        ' + line for line in options_html)}
    </ul>
    <br><b>Correct Answer(s):</b> {', '.join(sorted(correct_letters))}
</div>"""
                
                # Write entries with tab separation and newline
                f.write(f"{question_entry}\t{answer_entry}\n")
                
            except Exception as e:
                logger.error(f"Error writing question: {str(e)}")
                continue
    
    logger.info(f"Output written to {output_filepath}")
    return len(deduplicated_data)

def process_data(data, client=None, output_filepath="corrected_data.json", batch_size=10):
    """Processes the data, checks accuracy, and potentially generates better answers."""
    if batch_size > 1:
        return process_batch_data(data, client, batch_size, output_filepath)
    
    logger.info(f"Starting to process {len(data)} items")
    rate_limiter.reset_backoff()
    updated_answers_count = 0
    
    # Load from checkpoint
    start_index, checkpoint_data = load_checkpoint(output_filepath)
    corrected_data = checkpoint_data if checkpoint_data else []
    
    # Calculate remaining items
    remaining_data = data[start_index:] if start_index > 0 else data
    
    if not remaining_data:
        logger.info("No remaining items to process")
        return corrected_data
    
    logger.info(f"Processing remaining {len(remaining_data)} items starting from index {start_index}")
    
    total_requests = 0
    total_tokens = 0
    start_time = datetime.now()
    last_progress_time = start_time
    rate_limit_hits = 0
    successful_requests = 0
    
    def log_progress_stats():
        elapsed = (datetime.now() - start_time).total_seconds()
        items_per_second = (total_requests / elapsed) if elapsed > 0 else 0
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Get current rate diagnostics
        rate_limiter._update_current_rates()
        rpm_percentage = (rate_limiter.current_rpm / rate_limiter.max_rpm) * 100
        tpm_percentage = (rate_limiter.current_tpm / rate_limiter.max_tpm) * 100
        
        # Estimate time to completion
        total_questions = len(data)
        processed_questions = len(corrected_data)
        remaining_questions = total_questions - processed_questions
        
        # Calculate estimated processing rate considering batch size and rate limits
        effective_batch_size = 1  # Single item processing
        requests_per_batch = 2  # Accuracy check and potential answer generation
        
        # Estimate requests needed for remaining questions
        remaining_requests = (remaining_questions / effective_batch_size) * requests_per_batch
        
        # Estimate time considering rate limits
        # Add buffer for rate limit waits and API constraints
        rate_limit_buffer = 1.5  # 50% additional time for rate limit handling
        
        # Calculate estimated time to completion
        estimated_time_to_completion = (remaining_requests / max(items_per_second, 0.1)) * rate_limit_buffer if items_per_second > 0 else float('inf')
        
        # Convert to human-readable format
        def format_time(seconds):
            if seconds == float('inf'):
                return "Indeterminate"
            hours, remainder = divmod(int(seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {seconds}s"
        
        logger.info(f"""Progress Statistics:
        - Items Processed: {len(corrected_data)} / {total_questions}
        - Total Requests: {total_requests}
        - Total Tokens: {total_tokens}
        - Rate Limit Hits: {rate_limit_hits}
        - Success Rate: {success_rate:.1f}%
        - Requests/Second: {items_per_second:.2f}
        - Elapsed Time: {elapsed:.0f}s
        
        Current Rate Status:
        - RPM: {rate_limiter.current_rpm}/{rate_limiter.max_rpm} ({rpm_percentage:.1f}%)
        - TPM: {rate_limiter.current_tpm}/{rate_limiter.max_tpm} ({tpm_percentage:.1f}%)
        - Current Backoff: {rate_limiter.min_request_interval:.2f}s
        
        Completion Estimate:
        - Remaining Questions: {remaining_questions}
        - Estimated Time to Completion: {format_time(estimated_time_to_completion)}
        
        Rate Limit Hit Distribution:
        - RPM Limits: {rate_limiter.rpm_limit_hits}
        - TPM Limits: {rate_limiter.tpm_limit_hits}
        - Quota Limits: {rate_limiter.quota_limit_hits}""")
    
    try:
        for i, item in enumerate(remaining_data, start=start_index):
            current_time = datetime.now()
            if (current_time - last_progress_time).total_seconds() >= 60:
                log_progress_stats()
                last_progress_time = current_time
            
            logger.info(f"Processing item {i+1}/{len(data)} (Total requests: {total_requests}, Total tokens: {total_tokens})")
            
            try:
                # First API call - Check accuracy
                accurate, accuracy_response = check_accuracy_gemini(item['question'], item['correct_answer'], MODEL)
                total_requests += 1
                successful_requests += 1
                total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1000
                
                if not accurate:
                    logger.info(f"Generating new answer for item {i+1}")
                    time.sleep(0.5)  # Half second delay between calls
                    
                    # Second API call - Generate better answer
                    new_answer = generate_better_answer_gemini(item['question'], item['correct_answer'], MODEL)
                    total_requests += 1
                    successful_requests += 1
                    total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1524
                    
                    if new_answer:
                        logger.info("Successfully generated new answer")
                        updated_answers_count += 1
                        item["original_correct_answer"] = item["correct_answer"]
                        item["correct_answer"] = new_answer
                    else:
                        logger.warning("Failed to generate new answer, keeping original")
                
                # Store API responses in item for future reference
                item["accuracy_check_response"] = accuracy_response
                
                corrected_data.append(item)
                
                # Save checkpoint every 10 items
                if (i + 1) % 10 == 0:
                    save_checkpoint(corrected_data, output_filepath, i)
                    log_progress_stats()
                
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    rate_limit_hits += 1
                    logger.error(f"Rate limit hit after {total_requests} requests and {total_tokens} tokens")
                    save_checkpoint(corrected_data, output_filepath, i-1)
                    log_progress_stats()
                    raise
                else:
                    logger.error(f"Error processing item {i+1}: {str(e)}")
                    corrected_data.append(item)  # Keep original for non-429 errors
            except Exception as e:
                logger.error(f"Error processing item {i+1}: {str(e)}")
                corrected_data.append(item)
    finally:
        # Log final stats
        log_progress_stats()
        
        # Save final checkpoint
        save_checkpoint(corrected_data, output_filepath, len(data) - 1)
        logger.info(f"Processing complete - Total requests: {total_requests}, Total tokens: {total_tokens}")
        logger.info(f"Number of answers updated: {updated_answers_count}")
    
    return corrected_data

def process_batch_data(data, client=None, batch_size=5, output_filepath="corrected_data.json"):
    """Process data in batches to potentially improve throughput."""
    # Warn about unusually large batch sizes
    if batch_size > 50:
        logger.warning(f"Large batch size of {batch_size} specified. This may impact performance or hit API limits.")
    
    logger.info(f"Starting batch processing with requested batch size {batch_size}")
    
    # Reset rate limiter state for new run
    rate_limiter.reset_backoff()
    
    # Track updated answers
    updated_answers_count = 0
    
    # Load from checkpoint
    start_index, checkpoint_data = load_checkpoint(output_filepath)
    corrected_data = checkpoint_data if checkpoint_data else []
    
    # Calculate remaining items to process
    remaining_data = data[start_index:] if start_index > 0 else data
    
    if not remaining_data:
        logger.info("No remaining items to process")
        return corrected_data
    
    logger.info(f"Processing remaining {len(remaining_data)} items starting from index {start_index}")
    
    total_requests = 0
    total_tokens = 0
    start_time = datetime.now()
    last_progress_time = start_time
    rate_limit_hits = 0
    successful_requests = 0
    
    def log_progress_stats():
        elapsed = (datetime.now() - start_time).total_seconds()
        items_per_second = (total_requests / elapsed) if elapsed > 0 else 0
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Get current rate diagnostics
        rate_limiter._update_current_rates()
        rpm_percentage = (rate_limiter.current_rpm / rate_limiter.max_rpm) * 100
        tpm_percentage = (rate_limiter.current_tpm / rate_limiter.max_tpm) * 100
        
        # Estimate time to completion
        total_questions = len(data)
        processed_questions = len(corrected_data)
        remaining_questions = total_questions - processed_questions
        
        # Calculate estimated processing rate considering batch size and rate limits
        # Adjust for actual batch processing and API constraints
        effective_batch_size = min(batch_size, len(data))
        requests_per_batch = 2  # Accuracy check and potential answer generation
        
        # Estimate requests needed for remaining questions
        remaining_requests = (remaining_questions / effective_batch_size) * requests_per_batch
        
        # Estimate time considering rate limits
        # Add buffer for rate limit waits and API constraints
        rate_limit_buffer = 1.5  # 50% additional time for rate limit handling
        
        # Calculate estimated time to completion
        estimated_time_to_completion = (remaining_requests / max(items_per_second, 0.1)) * rate_limit_buffer if items_per_second > 0 else float('inf')
        
        # Convert to human-readable format
        def format_time(seconds):
            if seconds == float('inf'):
                return "Indeterminate"
            hours, remainder = divmod(int(seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {seconds}s"
        
        logger.info(f"""Batch Progress Statistics:
        - Items Processed: {len(corrected_data)} / {total_questions}
        - Total Requests: {total_requests}
        - Total Tokens: {total_tokens}
        - Rate Limit Hits: {rate_limit_hits}
        - Success Rate: {success_rate:.1f}%
        - Requests/Second: {items_per_second:.2f}
        - Elapsed Time: {elapsed:.0f}s
        
        Current Rate Status:
        - RPM: {rate_limiter.current_rpm}/{rate_limiter.max_rpm} ({rpm_percentage:.1f}%)
        - TPM: {rate_limiter.current_tpm}/{rate_limiter.max_tpm} ({tpm_percentage:.1f}%)
        - Current Backoff: {rate_limiter.min_request_interval:.2f}s
        
        Completion Estimate:
        - Remaining Questions: {remaining_questions}
        - Estimated Time to Completion: {format_time(estimated_time_to_completion)}
        
        Rate Limit Hit Distribution:
        - RPM Limits: {rate_limiter.rpm_limit_hits}
        - TPM Limits: {rate_limiter.tpm_limit_hits}
        - Quota Limits: {rate_limiter.quota_limit_hits}""")
    
    try:
        for i in range(0, len(remaining_data), batch_size):
            batch = remaining_data[i:i+batch_size]
            current_time = datetime.now()
            
            if (current_time - last_progress_time).total_seconds() >= 60:
                log_progress_stats()
                last_progress_time = current_time
            
            logger.info(f"Processing batch {i//batch_size + 1} (Items {i+1} to {i+len(batch)})")
            
            try:
                # Batch accuracy check
                batch_results = batch_check_accuracy_gemini(batch, MODEL, max_batch_size=batch_size)
                total_requests += 1
                successful_requests += 1
                
                # Process each item in the batch
                for j, (item, (is_accurate, accuracy_response)) in enumerate(zip(batch, batch_results)):
                    total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1000
                    
                    if not is_accurate:
                        logger.info(f"Generating new answer for item {start_index+i+j+1}")
                        time.sleep(0.5)  # Half second delay between calls
                        
                        # Generate better answer
                        new_answer = generate_better_answer_gemini(item['question'], item['correct_answer'], MODEL)
                        total_requests += 1
                        successful_requests += 1
                        total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1524
                        
                        if new_answer:
                            logger.info(f"Successfully generated new answer for item {start_index+i+j+1}")
                            updated_answers_count += 1
                            item["original_correct_answer"] = item["correct_answer"]
                            item["correct_answer"] = new_answer
                        else:
                            logger.warning(f"Failed to generate new answer for item {start_index+i+j+1}, keeping original")
                    
                    # Store API responses
                    item["accuracy_check_response"] = accuracy_response
                    corrected_data.append(item)
                
                # Save checkpoint every batch
                current_index = start_index + i + len(batch) - 1
                save_checkpoint(corrected_data, output_filepath, current_index)
                log_progress_stats()
                
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    rate_limit_hits += 1
                    logger.error(f"Rate limit hit after {total_requests} requests and {total_tokens} tokens")
                    save_checkpoint(corrected_data, output_filepath, start_index+i-1)
                    log_progress_stats()
                    raise
                else:
                    logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                    corrected_data.extend(batch)  # Keep original items
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                corrected_data.extend(batch)
    
    finally:
        # Log final stats
        log_progress_stats()
        
        # Save final checkpoint
        save_checkpoint(corrected_data, output_filepath, len(data) - 1)
        logger.info(f"Batch processing complete - Total requests: {total_requests}, Total tokens: {total_tokens}")
        logger.info(f"Number of answers updated: {updated_answers_count}")
    
    return corrected_data

def main():
    parser = argparse.ArgumentParser(
        description='Process and correct AWS exam questions using AI-powered analysis.',
        epilog='Example: python scan-correct.py -i input.txt -o output.txt --tier paid_tier_1 --batch-size 10'
    )
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help='Path to the input text file containing exam questions (HTML-like format)'
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help='Path to save the corrected output file'
    )
    parser.add_argument(
        '--tier', 
        choices=['free', 'paid_tier_1'], 
        default='free', 
        help='Google AI API tier (default: free). Paid tier allows more requests.'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=10, 
        help='Number of questions to process in each batch (recommended: 5-20, depending on API limits)'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        help='Limit the number of questions to process (useful for testing)'
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Initialize rate limiter with specified tier
    global rate_limiter
    rate_limiter = RateLimiter(tier=args.tier)

    # Load and process data
    data = load_data(args.input)
    if args.limit:
        data = data[:args.limit]

    # Process the data using the MODEL instead of client
    processed_data = process_data(data, output_filepath=args.output, batch_size=args.batch_size)
    
    # Write output
    write_output_file(processed_data, args.output)

    # Validate the processed dataset
    validation_results = validate_dataset(processed_data)

    # Log summary of updates and validation
    logger.info("\n--- Processing Summary ---")
    logger.info(f"Total questions processed: {len(processed_data)}")
    
    # Count updated answers
    updated_answers = [item for item in processed_data if item.get('original_correct_answer')]
    logger.info(f"Number of answers updated: {len(updated_answers)}")
    
    # Log validation results
    if validation_results['duplicate_answers']:
        logger.warning(f"Duplicate answers found: {len(validation_results['duplicate_answers'])}")
    
    if validation_results['missing_answers'] > 0:
        logger.warning(f"Missing answers found: {validation_results['missing_answers']}")

    # Optional: Log details of updated answers
    if updated_answers:
        logger.info("\nUpdated Answers Details:")
        for item in updated_answers[:10]:  # Log first 10 updated answers
            logger.info(f"Question: {item['question'][:100]}...")
            logger.info(f"Original Answer: {item.get('original_correct_answer')}")
            logger.info(f"Updated Answer: {item['correct_answer']}\n")

if __name__ == '__main__':
    main()


