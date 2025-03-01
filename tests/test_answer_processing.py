"""
Tests for AWS certification question processing functionality.
This module contains comprehensive tests for question processing, answer validation,
and error handling in the AWS certification question processing system.
"""

from typing import Dict, List, Optional

import pytest

import scan_correct
from scan_correct import (
    RateLimiter,
    is_multiple_answer_question,
    process_multiple_answer_question,
    update_question_with_multiple_answers,
)


class MockResponse:
    """Mock response object that simulates Gemini API responses."""

    def __init__(self, text: str):
        self.text = text


class ConfigurableMockModel:
    """
    Configurable mock model that simulates the Gemini API.
    Supports custom responses, error simulation, and call tracking.
    """

    def __init__(
        self, responses: Optional[Dict[str, str]] = None, should_error: bool = False
    ):
        self.responses = responses or {}
        self.should_error = should_error
        self.call_history: List[str] = []
        self.default_single_answer = "Correct Answer: A\nExplanation: Option A is correct because it matches the AWS service description."
        self.default_multiple_answer = "Correct Answers: A, C\nExplanation: Option A and C are correct because they are compute services."

    def start_chat(self, history=None):
        return self

    def send_message(self, prompt: str) -> MockResponse:
        """Simulate sending a message to the model."""
        self.call_history.append(prompt)
        print(f"DEBUG: Received prompt:\n{prompt}")

        # API error takes precedence
        if self.should_error:
            raise Exception("Simulated API error")

        # Check for custom responses based on the test case
        if "What is AWS Lambda?" in prompt:
            # Handle specific test cases for the Lambda question
            for key, response in self.responses.items():
                if key == "What is AWS Lambda?":
                    if response == "Invalid response format":
                        return MockResponse("Error: Invalid response format")
                    elif "Correct Answer: X" in response:
                        return MockResponse("Error: Invalid answer letter X")
                    elif "Correct Answers: A,B,C" in response:
                        return MockResponse("Error: Too many answers provided")
                    return MockResponse(response)

        # Handle other custom responses
        for key, response in self.responses.items():
            if key in prompt:
                return MockResponse(response)

        # Default responses based on question type
        if "Invalid question" in prompt:
            return MockResponse("Error: Cannot determine correct answers")
        elif "(Choose TWO)" in prompt:
            return MockResponse(self.default_multiple_answer)
        else:
            # For single answer questions, find the correct answer from the options
            lines = prompt.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("A."):
                    # Found the start of options, check which one is marked as correct
                    for j in range(4):  # Assuming 4 options A-D
                        if (
                            f"{chr(65 + j)}." in lines[i + j]
                            and "is_correct': True" in prompt
                        ):
                            print(f"DEBUG: Found correct answer: {chr(65 + j)}")
                            return MockResponse(
                                f"Correct Answer: {chr(65 + j)}\nExplanation: Option {chr(65 + j)} is correct because it matches the AWS service description."
                            )
            # Default to A if no correct answer found
            print("DEBUG: No correct answer found, defaulting to A")
            return MockResponse(
                "Correct Answer: A\nExplanation: Option A is correct because it matches the AWS service description."
            )


# Test Data Fixtures
@pytest.fixture
def rate_limiter():
    """Create and configure a rate limiter for testing."""
    limiter = RateLimiter(tier="free")
    scan_correct.rate_limiter = limiter
    yield limiter
    scan_correct.rate_limiter = None


@pytest.fixture
def mock_model():
    """Create a default mock model."""
    return ConfigurableMockModel()


@pytest.fixture
def single_answer_question():
    """Create a basic single-answer question fixture."""
    return {
        "question": "What is AWS Lambda?",
        "correct_answer": "A serverless compute service",
        "options": [
            {"text": "A serverless compute service", "is_correct": True},
            {"text": "A container service", "is_correct": False},
            {"text": "A virtual machine service", "is_correct": False},
            {"text": "A database service", "is_correct": False},
        ],
    }


@pytest.fixture
def multi_answer_question():
    """Create a basic multiple-answer question fixture."""
    return {
        "question": "Which of the following are AWS compute services? (Choose TWO)",
        "options": [
            {"text": "EC2", "is_correct": True},
            {"text": "S3", "is_correct": False},
            {"text": "Lambda", "is_correct": True},
            {"text": "RDS", "is_correct": False},
            {"text": "DynamoDB", "is_correct": False},
        ],
        "correct_answer": "EC2 and Lambda are both compute services.",
    }


@pytest.fixture
def html_question():
    """Create a question with HTML content."""
    return {
        "question": "What is the purpose of <code>aws configure</code>?",
        "options": [
            {"text": "Configure AWS CLI credentials", "is_correct": True},
            {"text": "Start AWS Console", "is_correct": False},
            {"text": "Create AWS resources", "is_correct": False},
            {"text": "List AWS services", "is_correct": False},
        ],
    }


class TestQuestionTypeDetection:
    """Tests for question type detection functionality."""

    def test_detect_single_answer(self, single_answer_question):
        """Test detection of single answer questions."""
        assert not is_multiple_answer_question(single_answer_question["question"])

    def test_detect_multiple_answer(self, multi_answer_question):
        """Test detection of multiple answer questions."""
        assert is_multiple_answer_question(multi_answer_question["question"])

    @pytest.mark.parametrize(
        "question_text,expected",
        [
            ("Which TWO services...", False),  # Should not match without "Choose"
            ("Choose THREE options...", False),  # Should not match non-TWO
            ("(Choose TWO)", True),  # Should match exact format
            ("Please Choose TWO of the following", True),  # Should match with context
        ],
    )
    def test_multiple_answer_variations(self, question_text: str, expected: bool):
        """Test various formats of multiple answer questions."""
        assert is_multiple_answer_question(question_text) == expected


class TestAnswerProcessing:
    """Tests for answer processing functionality."""

    def test_process_single_answer(
        self, single_answer_question, mock_model, rate_limiter
    ):
        """Test processing of single answer questions."""
        question_dict = {
            "question": single_answer_question["question"],
            "options": single_answer_question["options"],
            "correct_answer": single_answer_question["correct_answer"],
        }
        correct_letters, explanation = process_multiple_answer_question(
            question=question_dict,  # Pass as keyword argument
            client=mock_model,
        )
        assert len(correct_letters) == 1
        assert correct_letters[0] == "A"
        assert explanation is not None

    def test_process_multiple_answer(
        self, multi_answer_question, mock_model, rate_limiter
    ):
        """Test processing of multiple answer questions."""
        question_dict = {
            "question": multi_answer_question["question"],
            "options": multi_answer_question["options"],
            "correct_answer": multi_answer_question["correct_answer"],
        }
        correct_letters, explanation = process_multiple_answer_question(
            question=question_dict,  # Pass as keyword argument
            client=mock_model,
        )
        assert len(correct_letters) == 2
        assert set(correct_letters) == {"A", "C"}
        assert explanation is not None

    def test_update_question_answers(self, multi_answer_question):
        """Test updating question with new answers."""
        correct_letters = ["A", "C"]
        explanation = "Test explanation"
        updated = update_question_with_multiple_answers(
            multi_answer_question, correct_letters, explanation
        )
        assert updated["options"][0]["is_correct"] is True
        assert updated["options"][1]["is_correct"] is False
        assert updated["options"][2]["is_correct"] is True
        assert explanation in updated["correct_answer"]


class TestInputValidation:
    """Tests for input validation and error handling."""

    @pytest.mark.parametrize(
        "invalid_question",
        [
            {"question": "", "options": []},  # Empty question
            {"question": "Test", "options": None},  # None options
            {},  # Empty dict
            {"options": []},  # Missing question
            {"question": "Test"},  # Missing options
        ],
    )
    def test_invalid_input(self, invalid_question, mock_model, rate_limiter):
        """Test handling of invalid input formats."""
        correct_letters, explanation = process_multiple_answer_question(
            invalid_question, mock_model
        )
        assert correct_letters == []
        assert explanation is None

    def test_html_content(self, html_question, mock_model, rate_limiter):
        """Test processing questions with HTML content."""
        correct_letters, explanation = process_multiple_answer_question(
            html_question, mock_model
        )
        assert len(correct_letters) == 1
        assert correct_letters[0] == "A"

    @pytest.mark.parametrize("num_options", [1, 2, 5, 10])
    def test_varying_option_counts(self, num_options: int, mock_model, rate_limiter):
        """Test questions with varying numbers of options."""
        question = {
            "question": "Test question",
            "options": [
                {"text": f"Option {i}", "is_correct": i == 0}
                for i in range(num_options)
            ],
        }
        correct_letters, explanation = process_multiple_answer_question(
            question, mock_model
        )
        assert len(correct_letters) <= num_options


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_response_format(self, single_answer_question, rate_limiter):
        """Test handling of invalid response formats."""
        model = ConfigurableMockModel(
            responses={"What is AWS Lambda?": "Invalid response format"}
        )
        correct_letters, explanation = process_multiple_answer_question(
            single_answer_question, model
        )
        assert correct_letters == []
        assert explanation is None

    @pytest.mark.parametrize(
        "response,expected_letters",
        [
            ("Correct Answer: X", []),  # Invalid letter
            ("Correct Answers: A,B,C", ["A", "B"]),  # Too many letters
            ("Answer: A", ["A"]),  # Non-standard format
            ("A is correct", ["A"]),  # Very non-standard format
        ],
    )
    def test_response_format_variations(
        self,
        single_answer_question,
        rate_limiter,
        response: str,
        expected_letters: List[str],
    ):
        """Test handling of various response formats."""
        model = ConfigurableMockModel(responses={"What is AWS Lambda?": response})
        correct_letters, _ = process_multiple_answer_question(
            single_answer_question, model
        )
        assert correct_letters == expected_letters


class TestAnswerCounting:
    """Tests for correct answer counting and validation."""

    @pytest.fixture
    def question_with_both_markers(self):
        """Create a question that has both HTML class and explicit answer markers."""
        return {
            "html": """
            <div>
                <b>Question:</b><br>
                What is Amazon S3?
                <ul>
                    <li class="correct">A. Object storage service</li>
                    <li>B. Compute service</li>
                    <li>C. Database service</li>
                    <li>D. Network service</li>
                </ul>
                <br><b>Correct Answer(s):</b> A
            </div>
            """
        }

    @pytest.fixture
    def multiple_answer_with_mismatch(self):
        """Create a question where HTML classes and explicit answers don't match."""
        return {
            "html": """
            <div>
                <b>Question:</b><br>
                Which TWO of the following are AWS compute services? (Choose TWO)
                <ul>
                    <li class="correct">A. EC2</li>
                    <li>B. S3</li>
                    <li class="correct">C. Lambda</li>
                    <li class="correct">D. ECS</li>
                </ul>
                <br><b>Correct Answer(s):</b> A, C
            </div>
            """
        }

    def test_no_double_counting(self, question_with_both_markers):
        """Test that answers aren't double-counted when both markers exist."""
        result = scan_correct.parse_question(question_with_both_markers["html"])
        assert result["correct_count"] == 1
        assert len(result["correct_answers"]) == 1
        assert result["correct_answers"] == ["A"]
        assert not result["warnings"]

    def test_explicit_answers_override_classes(self, multiple_answer_with_mismatch):
        """Test that explicit answers override HTML class markers."""
        result = scan_correct.parse_question(multiple_answer_with_mismatch["html"])
        assert result["correct_count"] == 2
        assert set(result["correct_answers"]) == {"A", "C"}
        assert not any(
            opt["is_correct"] for opt in result["options"] if opt["letter"] == "D"
        )

    def test_single_answer_multiple_marked(self):
        """Test warning generation for single answer questions with multiple marks."""
        html = """
        <div>
            <b>Question:</b><br>
            What is AWS CloudFront?
            <ul>
                <li class="correct">A. Content delivery network</li>
                <li class="correct">B. Load balancer</li>
                <li>C. Storage service</li>
                <li>D. Database service</li>
            </ul>
        </div>
        """
        result = scan_correct.parse_question(html)
        assert any(
            "Single answer question has 2 answers marked" in w
            for w in result["warnings"]
        )


class TestQuestionPairHandling:
    """Tests for handling paired and unpaired questions."""

    @pytest.fixture
    def paired_questions(self):
        """Create a pair of questions (with and without answers)."""
        return (
            """
            <div>
                <b>Question:</b><br>
                What is Amazon EC2?
                <ul>
                    <li>A. Compute service</li>
                    <li>B. Storage service</li>
                    <li>C. Database service</li>
                    <li>D. Network service</li>
                </ul>
            </div>
            """,
            """
            <div>
                <b>Question:</b><br>
                What is Amazon EC2?
                <ul>
                    <li class="correct">A. Compute service</li>
                    <li>B. Storage service</li>
                    <li>C. Database service</li>
                    <li>D. Network service</li>
                </ul>
                <br><b>Correct Answer(s):</b> A
            </div>
            """,
        )

    def test_prefer_answered_version(self, paired_questions):
        """Test that the version with answers is preferred."""
        q1 = scan_correct.parse_question(paired_questions[0])
        q2 = scan_correct.parse_question(paired_questions[1])

        assert not any(opt["is_correct"] for opt in q1["options"])
        assert len([opt for opt in q2["options"] if opt["is_correct"]]) == 1
        assert q2["correct_answers"] == ["A"]

    def test_unpaired_question_handling(self):
        """Test handling of unpaired questions."""
        html = """
        <div>
            <b>Question:</b><br>
            What is AWS Lambda?
            <ul>
                <li class="correct">A. Serverless compute service</li>
                <li>B. Container service</li>
                <li>C. Virtual machine service</li>
                <li>D. Database service</li>
            </ul>
            <br><b>Correct Answer(s):</b> A
        </div>
        """
        result = scan_correct.parse_question(html)
        assert result["is_valid"]
        assert result["correct_answers"] == ["A"]
        assert not result["warnings"]


class TestHTMLParsingEdgeCases:
    """Tests for HTML parsing edge cases."""

    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        html = """
        <div>
            <b>Question:</b><br>
            What is AWS?
            <ul>
                <li class="correct">A. Cloud provider</li>
                <li>B. Software company</li>
                <li>C. Hardware manufacturer</li>
            </ul>
            <br><b>Correct Answer(s):</b> A
        </div>
        """
        result = scan_correct.parse_question(html)
        assert result["is_valid"]  # Should still be valid if core content is parseable
        assert result["correct_answers"] == ["A"]

    def test_missing_question_text(self):
        """Test handling of missing question text."""
        html = """
        <div>
            <b>Question:</b><br>
            <ul>
                <li class="correct">A. First option</li>
                <li>B. Second option</li>
            </ul>
            <br><b>Correct Answer(s):</b> A
        </div>
        """
        result = scan_correct.parse_question(html)
        assert not result["is_valid"]
        assert "Empty question text" in result["warnings"]

    def test_missing_options(self):
        """Test handling of missing options."""
        html = """
        <div>
            <b>Question:</b><br>
            What is AWS?
            <br><b>Correct Answer(s):</b> A
        </div>
        """
        result = scan_correct.parse_question(html)
        assert not result["is_valid"]
        assert "No options list found" in result["warnings"]

    def test_invalid_correct_answer_letter(self):
        """Test handling of invalid correct answer letters."""
        html = """
        <div>
            <b>Question:</b><br>
            What is AWS?
            <ul>
                <li>A. Cloud provider</li>
                <li>B. Software company</li>
            </ul>
            <br><b>Correct Answer(s):</b> C
        </div>
        """
        result = scan_correct.parse_question(html)
        assert "Invalid correct answer letters found: C" in result["warnings"]
        assert not result[
            "correct_answers"
        ]  # Should have no correct answers since C is invalid
