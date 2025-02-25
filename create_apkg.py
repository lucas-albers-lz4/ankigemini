import argparse  # Add argparse import
import html  # Import the html module for escaping HTML
import logging
import os
import re
from typing import List, Optional, Tuple

import genanki
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to INFO level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load environment variables
load_dotenv()

# Read configuration from .env
DECK_ID = int(os.getenv("ANKI_DECK_ID", "2059400110"))
MODEL_ID = int(os.getenv("ANKI_MODEL_ID", "1607392319"))
DECK_NAME = os.getenv("ANKI_DECK_NAME", "AWS Cloud Practitioner Practice Exam")

# Define the Anki deck
my_deck = genanki.Deck(
    DECK_ID,  # Unique ID for the deck
    DECK_NAME,
)

# Define the model for the cards
my_model = genanki.Model(
    MODEL_ID,
    "AWS Exam Question Model",
    fields=[
        {"name": "Question"},
        {"name": "Options"},
        {"name": "OptionsWithCorrect"},
        {"name": "CorrectOptions"},
        {"name": "OptionalExplanation"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """<div>
                <b>Question:</b><br>{{Question}}<ul>{{Options}}</ul></div>""",
            "afmt": """<div>
                <b>Question:</b><br>{{Question}}<ul>{{OptionsWithCorrect}}</ul>
                <br><b>Correct Answer(s):</b> {{CorrectOptions}}
                {{#OptionalExplanation}}<br>{{OptionalExplanation}}{{/OptionalExplanation}}
            </div>""",
        }
    ],
    css=""".card {
    font-family: Arial, sans-serif;
    font-size: 18px;
    text-align: left;
    color: black;
    background-color: white;
    line-height: 1.6;
}

ul {
    margin: 0;
    padding-left: 20px;
}

li {
    margin: 5px 0;
}

.correct {
    color: green;
    font-weight: bold;
}

.detailed-explanation {
    margin-top: 10px;
    padding: 10px;
    background-color: #f5f5f5;
    border-left: 4px solid #4CAF50;
}

.detailed-explanation p {
    margin: 5px 0;
}

.detailed-explanation ul {
    margin: 5px 0;
    padding-left: 25px;
}

.detailed-explanation strong {
    color: #2E7D32;
}""",
)


def extract_text_from_html(html_content: str) -> str:
    """
    Extract plain text from HTML content using BeautifulSoup.
    Preserves basic formatting like line breaks.

    Args:
        html_content: HTML content to extract text from

    Returns:
        Plain text with basic formatting preserved
    """
    if not html_content:
        return ""

    # Replace <br> tags with newlines before parsing
    html_content = (
        html_content.replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
    )

    # Parse HTML and extract text
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").strip()


def extract_question_text(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the question text from the BeautifulSoup object.

    Args:
        soup: BeautifulSoup object containing the question

    Returns:
        Question text or None if not found
    """
    question_element = soup.find("b", string="Question:")
    if not question_element:
        logging.warning("No question element found in HTML")
        return None

    # Get the text after the <b>Question:</b> element until the <ul> element
    question_text = ""
    current = question_element.next_sibling
    while current and not (hasattr(current, "name") and current.name == "ul"):
        if hasattr(current, "string") and current.string:
            question_text += str(current.string)
        elif isinstance(current, str):
            question_text += current
        current = current.next_sibling if hasattr(current, "next_sibling") else None

    # Clean up the question text
    question_text = question_text.replace("<br>", "").replace("<br/>", "").strip()

    # Log if question was empty after cleanup
    if not question_text:
        logging.warning("Question text was empty after cleanup")
        return None

    # Ensure the first letter of the question is capitalized
    if question_text and len(question_text) > 0:
        original = question_text
        question_text = question_text[0].upper() + question_text[1:]
        if original != question_text:
            logging.debug("Capitalized first letter of question")

    return html.unescape(question_text)


def extract_options_and_answers(
    soup: BeautifulSoup,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Extract options and correct answers from the BeautifulSoup object.

    Args:
        soup: BeautifulSoup object containing the options and answers

    Returns:
        Tuple of (options, correct_answers)
    """
    options = []
    correct_answers = []

    # Find all list items in the unordered list
    ul_element = soup.find("ul")
    if not ul_element:
        return [], []

    for li in ul_element.find_all("li"):
        # Extract the option letter and text
        li_text = li.get_text().strip()
        match = re.match(r"([A-Z])\.\s+(.*)", li_text)
        if match:
            letter, option_text = match.groups()
            # Ensure the first letter of the option text is capitalized
            if option_text and len(option_text) > 0:
                option_text = option_text[0].upper() + option_text[1:]
            options.append((letter, option_text.strip()))
            # Check if this option is marked as correct
            if "class" in li.attrs and "correct" in li["class"]:
                correct_answers.append(letter)

    return options, correct_answers


def extract_explanation(soup: BeautifulSoup) -> str:
    """
    Extract explanation from the BeautifulSoup object.
    Looks for both simple explanations and detailed explanation divs.

    Args:
        soup: BeautifulSoup object containing the explanation

    Returns:
        Explanation text or empty string if not found
    """
    # First try to find a detailed explanation div
    detailed_div = soup.find("div", class_="detailed-explanation")
    if detailed_div:
        logging.debug("Found detailed explanation div with HTML formatting")
        return str(detailed_div)  # Return the entire div with HTML formatting

    # If no detailed div, look for explanation after the "Explanation(s):" text
    explanation = ""
    explanation_element = soup.find("b", string=re.compile(r"Explanation\(s\):"))
    if explanation_element:
        logging.debug("Found simple explanation text")
        # Get all text and HTML after the explanation element until the next div
        current = explanation_element.next_sibling
        while current and not (isinstance(current, Tag) and current.name == "div"):
            if isinstance(current, NavigableString):
                explanation += str(current)
            elif isinstance(current, Tag):
                explanation += str(current)  # Preserve HTML tags
                logging.debug(f"Preserved HTML tag in explanation: <{current.name}>")
            current = current.next_sibling

        # Clean up the explanation text
        explanation = explanation.strip()
        if explanation:
            # Wrap simple explanations in the detailed-explanation div for consistent styling
            explanation = f"""<div class="detailed-explanation">
    <p><strong>Explanation:</strong></p>
    <p>{explanation}</p>
</div>"""
            logging.debug("Wrapped simple explanation in detailed-explanation div")
    else:
        logging.debug("No explanation found in question")

    return explanation


def find_correct_answers_from_text(soup: BeautifulSoup) -> List[str]:
    """
    Find correct answers from the "Correct Answer(s):" text.

    Args:
        soup: BeautifulSoup object containing the correct answers

    Returns:
        List of correct answer letters
    """
    correct_element = soup.find("b", string=re.compile(r"Correct Answer\(s\):"))
    if correct_element:
        correct_text = correct_element.next_sibling.strip()
        return [ans.strip() for ans in correct_text.split(",")]
    return []


def parse_question_numbers(question_str: str, total_questions: int) -> List[int]:
    """Parse comma-delimited question numbers and validate them."""
    try:
        # Split by comma and convert to integers
        numbers = [int(q.strip()) for q in question_str.split(",")]

        # Check for duplicates before validation
        original_count = len(numbers)
        unique_count = len(set(numbers))
        if original_count != unique_count:
            logging.info(
                f"Removed {original_count - unique_count} duplicate question numbers"
            )

        # Validate numbers are within range
        invalid = [n for n in numbers if n < 1 or n > total_questions]
        if invalid:
            msg = (
                f"Invalid question numbers: {invalid}. Valid range: 1-{total_questions}"
            )
            logging.error(msg)
            raise ValueError(msg)

        # Remove duplicates while preserving order
        seen = set()
        deduped_numbers = []
        for n in numbers:
            if n not in seen:
                seen.add(n)
                deduped_numbers.append(n)

        return deduped_numbers
    except ValueError as e:
        if "invalid literal for int()" in str(e):
            msg = "Question numbers must be comma-separated integers (e.g., '1,2,3')"
            logging.error(msg)
            raise ValueError(msg)
        raise


def process_file(
    file_path: str, selected_questions: Optional[List[int]] = None
) -> None:
    """
    Process the input file and add cards to the deck.

    Args:
        file_path: Path to the input file
        selected_questions: Optional list of question numbers to process (1-based indices)
    """
    with open(file_path, "r", encoding="utf-8") as file:
        # Read the entire content
        content = file.read()

        # Verify header
        if not content.startswith("#separator:tab\n#html:true\n"):
            logging.error(
                "Invalid file format. Expected #separator:tab and #html:true headers"
            )
            return

        # Remove headers
        content = content[content.index("#html:true\n") + len("#html:true\n") :]

        # Find all <div> blocks - each div is a complete question
        div_blocks = re.findall(r"<div>.*?</div>", content, re.DOTALL)

        # Filter questions if selected_questions is provided
        if selected_questions:
            try:
                div_blocks = [div_blocks[i - 1] for i in selected_questions]
                logging.info(
                    f"Processing {len(div_blocks)} selected questions: {selected_questions}"
                )
            except IndexError:
                logging.error(
                    f"Question number out of range. Total questions: {len(div_blocks)}"
                )
                return

        # Initialize counters
        cards_processed = 0
        failed_question_matches = 0
        failed_answer_matches = 0
        failed_option_matches = 0
        other_errors = 0
        capitalized_questions = 0
        preserved_html_count = 0

        # Process each div block
        for block_num, div_block in enumerate(div_blocks, start=1):
            # Check for explanation before logging
            has_explanation = (
                '<div class="detailed-explanation">' in div_block
                or "<b>Explanation(s):</b>" in div_block
            )
            log_message = f"Processing question {block_num}/{len(div_blocks)}"
            if has_explanation:
                log_message += " : OptionalExplanation"
            logging.info(log_message)

            try:
                # Parse the HTML block with BeautifulSoup
                soup = BeautifulSoup(div_block, "html.parser")

                # Extract question
                original_question = extract_question_text(soup)
                if not original_question:
                    failed_question_matches += 1
                    logging.warning(
                        f"Question {block_num}: Failed to match question pattern"
                    )
                    continue

                # Check if question was capitalized
                if (
                    original_question[0].isupper()
                    and div_block.find("Question:") + 10 < len(div_block)
                    and not div_block[div_block.find("Question:") + 10].isupper()
                ):
                    capitalized_questions += 1
                    logging.debug(f"Question {block_num}: Capitalized first letter")

                logging.debug(
                    f"Question {block_num} text: {original_question[:100]}..."
                )

                # Extract options and correct answers
                options, correct_answers = extract_options_and_answers(soup)

                if not options:
                    failed_option_matches += 1
                    logging.warning(f"Question {block_num}: No options found")
                    continue

                # If no correct answers found from classes, try to find from Correct Answer(s) section
                if not correct_answers:
                    correct_answers = find_correct_answers_from_text(soup)
                    if not correct_answers:
                        failed_answer_matches += 1
                        logging.warning(
                            f"Question {block_num}: No correct answers found"
                        )
                        continue

                # Extract explanation if present
                explanation = extract_explanation(soup)
                if (
                    "<code>" in explanation
                    or "<strong>" in explanation
                    or "<ul>" in explanation
                ):
                    preserved_html_count += 1
                    logging.debug(
                        f"Question {block_num}: Preserved HTML formatting in explanation"
                    )

                # Add a card to the deck
                my_deck.add_note(
                    genanki.Note(
                        model=my_model,
                        fields=[
                            original_question,
                            "".join(
                                [
                                    f"<li>{letter}. {text}</li>"
                                    for letter, text in options
                                ]
                            ),
                            "".join(
                                [
                                    f'<li class="{"correct" if letter in correct_answers else ""}">{letter}. {text}</li>'
                                    for letter, text in options
                                ]
                            ),
                            ", ".join(sorted(correct_answers)),
                            explanation,
                        ],
                    )
                )

                cards_processed += 1

            except Exception as e:
                other_errors += 1
                logging.error(
                    f"Question {block_num}: Error processing question: {str(e)}"
                )
                continue

        # Print summary
        logging.info("\n=== Processing Summary ===")
        logging.info(f"Total questions found: {len(div_blocks)}")
        logging.info(f"Successfully processed: {cards_processed}")
        logging.info(f"Questions capitalized: {capitalized_questions}")
        logging.info(f"HTML preserved in explanations: {preserved_html_count}")
        if failed_question_matches > 0:
            logging.info(f"Failed question matches: {failed_question_matches}")
        if failed_answer_matches > 0:
            logging.info(f"Failed answer matches: {failed_answer_matches}")
        if failed_option_matches > 0:
            logging.info(f"Failed option matches: {failed_option_matches}")
        if other_errors > 0:
            logging.info(f"Other errors: {other_errors}")
        logging.info("=======================")


def main():
    """Main function to parse arguments and process the file."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create Anki deck from exam questions")
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output .apkg file path")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "-q",
        "--questions",
        help="Process specific question numbers (comma-delimited, e.g., '1,2,3')",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process selected questions if specified
    selected_questions = None
    if args.questions:
        try:
            # Get total number of questions first
            with open(args.input, "r", encoding="utf-8") as f:
                content = f.read()
                total_questions = len(re.findall(r"<div>.*?</div>", content, re.DOTALL))
            selected_questions = parse_question_numbers(args.questions, total_questions)
            logging.info(f"Processing selected questions: {selected_questions}")
        except ValueError as e:
            logging.error(str(e))
            return

    # Process the input file
    process_file(args.input, selected_questions)

    # Save the deck to an .apkg file
    genanki.Package(my_deck).write_to_file(args.output)
    print(f"Anki deck created: {args.output}")


if __name__ == "__main__":
    main()
