"""
Unit tests for create_apkg.py functionality.
Tests question processing, HTML escaping, and Anki card generation features.
"""

from typing import Any

import pytest
from bs4 import BeautifulSoup

from create_apkg import (
    extract_explanation,
    extract_options_and_answers,
    extract_question_text,
    find_correct_answers_from_text,
    parse_question_numbers,
    process_file,
)


@pytest.fixture
def basic_question_html() -> str:
    """Create a basic question HTML fixture."""
    return """<div>
        <b>Question:</b> what is amazon s3?
        <ul>
            <li>A. A storage service</li>
            <li class="correct">B. An object storage service</li>
            <li>C. A file system</li>
            <li>D. A block storage service</li>
        </ul>
        <b>Explanation(s):</b> Amazon S3 is an object storage service.
    </div>"""


@pytest.fixture
def html_content_question() -> str:
    """Create a question with HTML content that needs escaping."""
    return """<div>
        <b>Question:</b> What does the <code>aws s3 sync</code> command do?
        <ul>
            <li class="correct">A. Syncs directories &amp; S3 buckets</li>
            <li>B. Creates S3 buckets</li>
            <li>C. Lists S3 buckets</li>
            <li>D. Deletes S3 buckets</li>
        </ul>
        <b>Explanation(s):</b> The <code>aws s3 sync</code> command synchronizes directories.
    </div>"""


@pytest.fixture
def detailed_explanation_question():
    """Create a question with a detailed explanation div."""
    return """<div>
        <b>Question:</b> Which AWS service is serverless?
        <ul>
            <li class="correct">A. Lambda</li>
            <li>B. EC2</li>
            <li>C. ECS</li>
            <li>D. EKS</li>
        </ul>
        <div class="detailed-explanation">
            <p><strong>Detailed Explanation:</strong></p>
            <p>AWS Lambda is a serverless compute service that:</p>
            <ul>
                <li>Runs code without provisioning servers</li>
                <li>Scales automatically</li>
                <li>Only charges for actual usage</li>
            </ul>
            <p>Other options require server management:</p>
            <ul>
                <li>EC2: Virtual servers</li>
                <li>ECS: Container orchestration</li>
                <li>EKS: Managed Kubernetes</li>
            </ul>
        </div>
    </div>"""


class TestQuestionExtraction:
    """Tests for question text extraction and formatting."""

    def test_capitalize_first_letter(self, basic_question_html: str) -> None:
        """Test that the first letter of the question is capitalized."""
        soup = BeautifulSoup(basic_question_html, "html.parser")
        question_text = extract_question_text(soup)
        assert question_text is not None
        assert question_text[0].isupper()
        assert "What is amazon s3?" in question_text

    def test_html_escaping_in_question(self, html_content_question: str) -> None:
        """Test that HTML in questions is properly escaped."""
        soup = BeautifulSoup(html_content_question, "html.parser")
        question_text = extract_question_text(soup)
        assert question_text is not None
        assert "<code>" not in question_text
        assert "aws s3 sync" in question_text
        assert "&amp;" not in question_text
        assert "&" not in question_text  # Should be properly unescaped

    def test_missing_question(self) -> None:
        """Test handling of HTML without a question."""
        soup = BeautifulSoup("<div><ul><li>A. Option</li></ul></div>", "html.parser")
        question_text = extract_question_text(soup)
        assert question_text is None


class TestOptionsAndAnswers:
    """Tests for option extraction and answer identification."""

    def test_extract_options(self, basic_question_html: str) -> None:
        """Test extraction of options and identification of correct answers."""
        soup = BeautifulSoup(basic_question_html, "html.parser")
        options, correct_answers = extract_options_and_answers(soup)

        assert len(options) == 4
        assert len(correct_answers) == 1
        assert correct_answers[0] == "B"
        assert options[1][0] == "B"  # Second option should be B
        assert "object storage service" in options[1][1]  # Verify option text

    def test_html_escaping_in_options(self, html_content_question: str) -> None:
        """Test that HTML in options is properly escaped."""
        soup = BeautifulSoup(html_content_question, "html.parser")
        options, correct_answers = extract_options_and_answers(soup)

        assert len(options) == 4
        assert len(correct_answers) == 1
        assert correct_answers[0] == "A"
        assert "&amp;" not in options[0][1]  # HTML entities should be unescaped
        assert "&" in options[0][1]  # But actual ampersands should be preserved

    def test_no_options(self) -> None:
        """Test handling of HTML without options."""
        soup = BeautifulSoup("<div><b>Question:</b> Test</div>", "html.parser")
        options, correct_answers = extract_options_and_answers(soup)
        assert len(options) == 0
        assert len(correct_answers) == 0


class TestExplanationExtraction:
    """Tests for explanation extraction and formatting."""

    def test_simple_explanation(self, basic_question_html: str) -> None:
        """Test extraction of simple text explanations."""
        soup = BeautifulSoup(basic_question_html, "html.parser")
        explanation = extract_explanation(soup)
        assert "detailed-explanation" in explanation
        assert "Amazon S3 is an object storage service" in explanation

    def test_detailed_explanation(self, detailed_explanation_question: str) -> None:
        """Test extraction of detailed HTML explanations."""
        soup = BeautifulSoup(detailed_explanation_question, "html.parser")
        explanation = extract_explanation(soup)

        assert "detailed-explanation" in explanation
        assert "AWS Lambda is a serverless compute service" in explanation
        assert "<ul>" in explanation  # HTML structure should be preserved
        assert "<li>" in explanation
        assert "Runs code without provisioning servers" in explanation

    def test_html_escaping_in_explanation(self, html_content_question: str) -> None:
        """Test that HTML in explanations is properly handled."""
        soup = BeautifulSoup(html_content_question, "html.parser")
        explanation = extract_explanation(soup)

        assert "detailed-explanation" in explanation
        assert (
            "<code>aws s3 sync</code>" in explanation
        )  # Code tags should be preserved
        assert "synchronizes directories" in explanation

    def test_missing_explanation(self) -> None:
        """Test handling of questions without explanations."""
        html = """<div>
            <b>Question:</b> Test
            <ul><li>A. Option</li></ul>
        </div>"""
        soup = BeautifulSoup(html, "html.parser")
        explanation = extract_explanation(soup)
        assert explanation == ""


class TestQuestionNumberParsing:
    """Tests for question number parsing and validation."""

    def test_valid_question_numbers(self) -> None:
        """Test parsing of valid question number strings."""
        assert parse_question_numbers("1,2,3", 5) == [1, 2, 3]
        assert parse_question_numbers("1", 5) == [1]
        assert parse_question_numbers("1, 3, 5", 5) == [1, 3, 5]

    def test_invalid_question_numbers(self) -> None:
        """Test handling of invalid question number strings."""
        with pytest.raises(ValueError):
            parse_question_numbers("1,2,6", 5)  # Number too high
        with pytest.raises(ValueError):
            parse_question_numbers("0,1,2", 5)  # Number too low
        with pytest.raises(ValueError):
            parse_question_numbers("a,b,c", 5)  # Invalid format
        with pytest.raises(ValueError):
            parse_question_numbers("1.5,2,3", 5)  # Decimal numbers

    def test_duplicate_numbers(self) -> None:
        """Test handling of duplicate question numbers."""
        assert parse_question_numbers("1,1,1", 5) == [1]  # Should deduplicate
        assert parse_question_numbers("1,2,2,3", 5) == [1, 2, 3]

    def test_whitespace_handling(self) -> None:
        """Test handling of whitespace in question number strings."""
        assert parse_question_numbers(" 1, 2,3 ", 5) == [1, 2, 3]
        assert parse_question_numbers("1,  2,   3", 5) == [1, 2, 3]


def test_extract_question_text() -> None:
    """Test extracting question text from BeautifulSoup object."""
    html = """
    <div>
        <b>Question:</b><br>
        What is the capital of France?
        <ul>
            <li>A. London</li>
            <li class="correct">B. Paris</li>
            <li>C. Berlin</li>
        </ul>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    result = extract_question_text(soup)
    assert result == "What is the capital of France?"


def test_extract_options_and_answers() -> None:
    """Test extracting options and answers from BeautifulSoup object."""
    html = """
    <div>
        <ul>
            <li>A. Option 1</li>
            <li class="correct">B. Option 2</li>
            <li>C. Option 3</li>
        </ul>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    options, answers = extract_options_and_answers(soup)
    assert len(options) == 3
    assert len(answers) == 1
    assert answers[0] == "B"


def test_extract_explanation() -> None:
    """Test extracting explanation from BeautifulSoup object."""
    html = """
    <div>
        <b>Explanation(s):</b>
        This is the explanation.
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    result = extract_explanation(soup)
    assert "This is the explanation" in result


def test_find_correct_answers_from_text() -> None:
    """Test finding correct answers from text."""
    html = """
    <div>
        <b>Correct Answer(s):</b> A, B
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    result = find_correct_answers_from_text(soup)
    assert result == ["A", "B"]


def test_parse_question_numbers() -> None:
    """Test parsing question numbers."""
    result = parse_question_numbers("1,2,3", 5)
    assert result == [1, 2, 3]

    with pytest.raises(ValueError):
        parse_question_numbers("1,2,6", 5)  # 6 is out of range


def test_process_file(tmp_path: Any) -> None:
    """Test processing a file."""
    test_file = tmp_path / "test.html"
    test_file.write_text("""
    <div>
        <b>Question:</b><br>
        Test question
        <ul>
            <li>A. Option 1</li>
            <li class="correct">B. Option 2</li>
            <li>C. Option 3</li>
        </ul>
        <b>Explanation(s):</b>
        Test explanation
    </div>
    """)
    process_file(str(test_file))
    # Add assertions based on expected output
