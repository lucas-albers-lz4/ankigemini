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

def save_progress(data, output_filepath, current_index):
    """Save current progress to a temporary file"""
    temp_filepath = f"{output_filepath}.temp"
    logger.info(f"Saving progress to {temp_filepath} at index {current_index}")
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'current_index': current_index,
            'data': data[:current_index + 1]
        }, f)

def load_progress(output_filepath):
    """Load progress from temporary file if it exists"""
    temp_filepath = f"{output_filepath}.temp"
    try:
        if os.path.exists(temp_filepath):
            with open(temp_filepath, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                logger.info(f"Loaded progress from {temp_filepath}, resuming from index {progress['current_index']}")
                return progress['current_index'], progress['data']
    except Exception as e:
        logger.warning(f"Failed to load progress: {str(e)}")
    return 0, None

def save_safe_checkpoint(data, checkpoint_filepath, current_index):
    """
    Save a checkpoint with additional error handling and consistency checks.
    
    :param data: Data to save
    :param checkpoint_filepath: Path to save checkpoint
    :param current_index: Current processing index
    """
    try:
        # Ensure checkpoints directory exists
        os.makedirs('checkpoints', exist_ok=True)
        
        # Create a temporary checkpoint file first
        temp_checkpoint_filepath = f"{checkpoint_filepath}.tmp"
        
        # Checkpoint data with additional metadata for recovery
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'current_index': current_index,
            'total_questions': len(data),
            'data': data[:current_index + 1],
            'version': '1.0'  # Version for future compatibility
        }
        
        # Write to temporary file first
        with open(temp_checkpoint_filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Atomically replace the actual checkpoint file
        os.replace(temp_checkpoint_filepath, checkpoint_filepath)
        
        logger.info(f"Safe checkpoint saved to {checkpoint_filepath} at index {current_index}")
    
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}", exc_info=True)
        # Attempt to log error details without raising
        try:
            with open('checkpoint_errors.log', 'a') as error_log:
                error_log.write(f"{datetime.now().isoformat()} - Checkpoint Error: {str(e)}\n")
        except:
            pass

def load_safe_checkpoint(checkpoint_filepath):
    """
    Load a checkpoint with robust error handling.
    
    :param checkpoint_filepath: Path to checkpoint file
    :return: Tuple of (start_index, loaded_data)
    """
    try:
        if os.path.exists(checkpoint_filepath):
            with open(checkpoint_filepath, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # Validate checkpoint data
            if not all(key in checkpoint for key in ['current_index', 'data', 'timestamp']):
                logger.warning("Invalid checkpoint format")
                return 0, None
            
            # Check checkpoint age
            checkpoint_time = datetime.fromisoformat(checkpoint['timestamp'])
            if datetime.now() - checkpoint_time > timedelta(days=7):
                logger.warning("Checkpoint is too old, starting fresh")
                return 0, None
            
            logger.info(f"Loaded safe checkpoint from {checkpoint_filepath}, resuming from index {checkpoint['current_index']}")
            return checkpoint['current_index'], checkpoint['data']
    
    except json.JSONDecodeError:
        logger.error("Checkpoint file is corrupted")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
    
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
    
    prompt = f"""
    Question: {question}
    Correct Answer: {correct_answer}

    Is the "Correct Answer" truly the best and most accurate answer to the "Question", 
    specifically in the context of AWS Cloud Practitioner knowledge?  Provide a brief explanation
    and then state either "Accurate" or "Inaccurate". Return only this information with no preamble.
    """

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
    
    Aims to provide:
    1. Explanation of why this is the correct answer
    2. Context and background information
    3. Practical implications or real-world relevance
    4. Potential exam-related insights
    """
    # Estimate tokens: question + answer + prompt (~500) + max response (1024)
    estimated_tokens = len(question.split()) + len(correct_answer.split()) + 1524
    
    prompt = f"""
    Provide a comprehensive explanation for the following AWS Cloud Practitioner exam question:

    Question: {question}
    Current Correct Answer: {correct_answer}

    Generate a detailed response that includes:
    1. A clear, concise explanation of why this is the correct answer
    2. Contextual background information related to the topic
    3. Practical real-world application or significance in cloud computing
    4. Key points an exam taker should understand
    5. Any related AWS services or concepts that provide additional insight
    6. Common misconceptions or tricky aspects of this topic

    Format your response to be clear, informative, and exam-preparation focused. 
    Aim for a response that not only confirms the answer but provides deep understanding.
    """

    try:
        # Wait for rate limit with token estimation
        rate_limiter.wait_if_needed(estimated_tokens)
        
        # Start a new chat session
        chat_session = MODEL.start_chat(history=[])
        response = chat_session.send_message(prompt)
        
        # Reset consecutive 429s on successful request
        rate_limiter.reset_consecutive_429s()
        
        # Enhance the response with structured formatting
        detailed_answer = f"""
        <div class="detailed-explanation">
            <p><strong>Comprehensive Explanation:</strong></p>
            {response.text}
        </div>
        """
        
        return detailed_answer
    except Exception as e:
        logger.error(f"Error during API call: {str(e)}", exc_info=True)
        raise

def is_low_quality_answer(text, min_length=50, max_length=1000):
    """
    Check if an answer is low quality based on improved criteria.
    """
    # Length constraints
    if len(text) < min_length or len(text) > max_length:
        return True
    
    # Generic or unhelpful responses
    generic_patterns = [
        r'^(just|only|simply)\s+',  # Oversimplified starts
        r'^(the\s+)?(correct\s+)?answer\s+is\s+[A-D]$',  # Just stating the letter
        r'^\s*$',  # Empty or whitespace only
        r'^[A-D]\.?\s*$'  # Single letter answers
    ]
    
    for pattern in generic_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    
    # Check for AWS-specific content
    aws_terms = ['aws', 'amazon', 'cloud', 'service']
    if not any(term in text.lower() for term in aws_terms):
        return True
    
    # More nuanced repetition check
    words = text.lower().split()
    unique_words = len(set(words))
    # Exclude common AWS terms from repetition calculation
    if unique_words < len(words) * 0.25 and len(words) > 20:  # Only check longer answers
        return True
    
    return False

def find_duplicate_answers(data):
    """
    Find and log duplicate answers across the dataset
    """
    answer_counts = {}
    duplicates = []
    
    for item in data:
        answer = item['correct_answer']
        if answer in answer_counts:
            duplicates.append({
                'question': item['question'],
                'duplicate_of': answer_counts[answer]['question']
            })
        else:
            answer_counts[answer] = {
                'question': item['question'],
                'count': 1
            }
    
    return duplicates

def validate_dataset(data):
    """
    Comprehensive dataset validation
    """
    logger.info("Starting comprehensive dataset validation")
    
    # Check for duplicate answers
    duplicate_answers = find_duplicate_answers(data)
    if duplicate_answers:
        logger.warning(f"Found {len(duplicate_answers)} duplicate answers")
        for dup in duplicate_answers[:10]:  # Log first 10 duplicates
            logger.warning(f"Duplicate Answer: \n"
                           f"Question: {dup['question']}\n"
                           f"Duplicate of Question: {dup['duplicate_of']}")
    
    # Check for low-quality answers
    low_quality_answers = []
    for item in data:
        if is_low_quality_answer(item['correct_answer']):
            low_quality_answers.append(item)
    
    if low_quality_answers:
        logger.warning(f"Found {len(low_quality_answers)} low-quality answers")
        for answer in low_quality_answers[:10]:  # Log first 10 low-quality answers
            logger.warning(f"Low-Quality Answer: \n"
                           f"Question: {answer['question']}\n"
                           f"Answer: {answer['correct_answer']}")
    
    return {
        'duplicate_answers': duplicate_answers,
        'low_quality_answers': low_quality_answers
    }

def write_output_file(data, output_filepath):
    """
    Write the processed data back to a file in the original HTML-like format
    """
    logger.info(f"Writing output to {output_filepath}")
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for item in data:
            # Use original question HTML if available, otherwise use current question
            question_html = item.get('original_question', item['question'])
            
            # Reconstruct options HTML
            options_html = ''.join([
                f'<li class=\'{"correct" if opt["is_correct"] else ""}\'>{opt["text"]}</li>'
                for opt in item.get('options', [])
            ])
            
            # Construct correct answers string
            correct_letters = [chr(65 + i) for i, opt in enumerate(item.get('options', [])) if opt['is_correct']]
            correct_answers_str = ', '.join(correct_letters)
            
            # Create the full HTML-like entry
            entry = f"""
            <div>
                <b>Question:</b><br>
                {question_html}
                <ul>
                    {options_html}
                </ul>
                <br><b>Correct Answer(s):</b> {correct_answers_str}
            </div>
            """
            
            # Write the entry
            f.write(entry + "\t")  # Maintain the tab separator
    
    logger.info(f"Output written to {output_filepath}")

def process_data(data, client=None, output_filepath="corrected_data.json", batch_size=10):
    """Processes the data, checks accuracy, and potentially generates better answers."""
    # If batch size is greater than 1, use batch processing
    if batch_size > 1:
        return process_batch_data(data, client, batch_size, output_filepath)
    
    logger.info(f"Starting to process {len(data)} items")
    
    # Reset rate limiter state for new run
    rate_limiter.reset_backoff()
    
    # Track updated answers
    updated_answers_count = 0
    
    # Try to load from checkpoint first
    checkpoint_filepath = os.path.join('checkpoints', f"{os.path.basename(output_filepath)}.checkpoint")
    start_index, checkpoint_data = load_safe_checkpoint(checkpoint_filepath)
    
    # If no checkpoint, try temp file
    if start_index == 0 and checkpoint_data is None:
        start_index, checkpoint_data = load_progress(output_filepath)
    
    corrected_data = checkpoint_data if checkpoint_data else []
    
    if start_index > 0:
        data = data[start_index:]  # Skip already processed items
    
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
        for i, item in enumerate(data, start=start_index):
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
                    # Add a small delay between consecutive API calls for the same item
                    time.sleep(0.5)  # Half second delay between calls
                    
                    # Second API call - Generate better answer
                    new_answer = generate_better_answer_gemini(item['question'], item['correct_answer'], MODEL)
                    total_requests += 1
                    successful_requests += 1
                    total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1524
                    
                    if new_answer:
                        logger.info("Successfully generated new answer")
                        # Increment updated answers count
                        updated_answers_count += 1
                        
                        # Preserve the full original HTML structure
                        # Extract the list of choices
                        import re
                        choices_match = re.search(r'<ul>(.*?)</ul>', item['question'], re.DOTALL)
                        
                        if choices_match:
                            choices_html = choices_match.group(1)
                            
                            # Modify the correct choice to highlight the new answer
                            modified_choices = re.sub(
                                r'<li class=[\'"]correct[\'"]>(.*?)</li>', 
                                f'<li class="correct">{new_answer}</li>', 
                                choices_html
                            )
                            
                            # Reconstruct the full HTML with the new answer
                            modified_question_html = re.sub(
                                r'<ul>.*?</ul>', 
                                f'<ul>{modified_choices}</ul>', 
                                item['question'], 
                                flags=re.DOTALL
                            )
                            
                            item["original_question"] = item["question"]
                            item["question"] = modified_question_html
                            item["original_correct_answer"] = item["correct_answer"]
                            item["correct_answer"] = new_answer
                        else:
                            logger.warning("Could not extract choices from question HTML")
                    else:
                        logger.warning("Failed to generate new answer, keeping original")
                
                # Store API responses in item for future reference
                item["accuracy_check_response"] = accuracy_response
                
                corrected_data.append(item)
                
                # Save progress every 10 items
                if (i + 1) % 10 == 0:
                    save_progress(corrected_data, output_filepath, i)
                    save_safe_checkpoint(corrected_data, checkpoint_filepath, i)
                    log_progress_stats()
                
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    rate_limit_hits += 1
                    logger.error(f"Rate limit hit after {total_requests} requests and {total_tokens} tokens")
                    # Save progress before stopping
                    save_progress(corrected_data, output_filepath, i-1)
                    save_safe_checkpoint(corrected_data, checkpoint_filepath, i-1)
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
        save_safe_checkpoint(corrected_data, checkpoint_filepath, len(data) - 1)
        logger.info(f"Processing complete - Total requests: {total_requests}, Total tokens: {total_tokens}")
        
        # Log updated answers count
        logger.info(f"Number of answers updated: {updated_answers_count}")
    
    return corrected_data

def process_batch_data(data, client=None, batch_size=5, output_filepath="corrected_data.json"):
    """
    Process data in batches to potentially improve throughput.
    
    :param data: List of questions
    :param client: Ignored in new implementation (using global MODEL)
    :param batch_size: Number of questions to process in each batch
    :param output_filepath: Path to save processed data
    """
    # Warn about unusually large batch sizes
    if batch_size > 50:
        logger.warning(f"Large batch size of {batch_size} specified. This may impact performance or hit API limits.")
    
    logger.info(f"Starting batch processing with requested batch size {batch_size}")
    
    # Reset rate limiter state for new run
    rate_limiter.reset_backoff()
    
    # Track updated answers
    updated_answers_count = 0
    
    # Try to load from checkpoint first
    checkpoint_filepath = os.path.join('checkpoints', f"{os.path.basename(output_filepath)}.checkpoint")
    start_index, checkpoint_data = load_safe_checkpoint(checkpoint_filepath)
    
    # If no checkpoint, try temp file
    if start_index == 0 and checkpoint_data is None:
        start_index, checkpoint_data = load_progress(output_filepath)
    
    corrected_data = checkpoint_data if checkpoint_data else []
    
    if start_index > 0:
        data = data[start_index:]  # Skip already processed items
    
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
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
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
                
                # Track updated answers in the batch
                batch_updated_answers = [item for item, (is_accurate, _) in zip(batch, batch_results) if not is_accurate]
                updated_answers_count += len(batch_updated_answers)
                
                # Process each item in the batch
                for j, (item, (is_accurate, accuracy_response)) in enumerate(zip(batch, batch_results)):
                    total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1000
                    
                    if not is_accurate:
                        logger.info(f"Generating new answer for item {i+j+1}")
                        time.sleep(0.5)  # Half second delay between calls
                        
                        # Generate better answer
                        new_answer = generate_better_answer_gemini(item['question'], item['correct_answer'], MODEL)
                        total_requests += 1
                        successful_requests += 1
                        total_tokens += len(item['question'].split()) + len(item['correct_answer'].split()) + 1524
                        
                        if new_answer:
                            logger.info(f"Successfully generated new answer for item {i+j+1}")
                            item["original_correct_answer"] = item["correct_answer"]
                            item["correct_answer"] = new_answer
                        else:
                            logger.warning(f"Failed to generate new answer for item {i+j+1}, keeping original")
                    
                    # Store API responses
                    item["accuracy_check_response"] = accuracy_response
                    corrected_data.append(item)
                
                # Save progress every batch
                save_progress(corrected_data, output_filepath, i + len(batch) - 1)
                save_safe_checkpoint(corrected_data, checkpoint_filepath, i + len(batch) - 1)
                log_progress_stats()
                
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    rate_limit_hits += 1
                    logger.error(f"Rate limit hit after {total_requests} requests and {total_tokens} tokens")
                    save_progress(corrected_data, output_filepath, i-1)
                    save_safe_checkpoint(corrected_data, checkpoint_filepath, i-1)
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
        save_safe_checkpoint(corrected_data, checkpoint_filepath, len(data) - 1)
        logger.info(f"Batch processing complete - Total requests: {total_requests}, Total tokens: {total_tokens}")
        
        # Log updated answers count
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
    
    if validation_results['low_quality_answers']:
        logger.warning(f"Low-quality answers found: {len(validation_results['low_quality_answers'])}")

    # Optional: Log details of updated answers
    if updated_answers:
        logger.info("\nUpdated Answers Details:")
        for item in updated_answers[:10]:  # Log first 10 updated answers
            logger.info(f"Question: {item['question'][:100]}...")
            logger.info(f"Original Answer: {item.get('original_correct_answer')}")
            logger.info(f"Updated Answer: {item['correct_answer']}\n")

if __name__ == '__main__':
    main()


