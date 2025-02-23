import genanki
import re
import html  # Import the html module for escaping HTML
import argparse  # Add argparse import
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to INFO level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Read configuration from .env
DECK_ID = int(os.getenv('ANKI_DECK_ID', '2059400110'))
MODEL_ID = int(os.getenv('ANKI_MODEL_ID', '1607392319'))
DECK_NAME = os.getenv('ANKI_DECK_NAME', 'AWS Cloud Practitioner Practice Exam')

# Define the Anki deck
my_deck = genanki.Deck(
    DECK_ID,  # Unique ID for the deck
    DECK_NAME
)

# Define the model for the cards
my_model = genanki.Model(
    MODEL_ID,
    'AWS Exam Question Model',
    fields=[
        {'name': 'Question'},
        {'name': 'Options'},
        {'name': 'OptionsWithCorrect'},
        {'name': 'CorrectOptions'},
        {'name': 'OptionalExplanation'},
    ],
    templates=[
        {
            'name': 'Card 1',
            'qfmt': '''<div>
                <b>Question:</b><br>
                {{Question}}
                <ul>
                    {{Options}}
                </ul>
            </div>''',
            'afmt': '''
            <div>
                <b>Question:</b><br>
                {{Question}}
                <ul>
                    {{OptionsWithCorrect}}
                </ul>
                <br><b>Correct Answer(s):</b> {{CorrectOptions}}
<br><b>Explanation(s):</b> {{OptionalExplanation}}
            </div>
            ''',
        }
    ],
    css='''.card {
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
}'''
)

# Function to process the text file and add cards to the deck
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire content
        content = file.read()
        
        # Verify header
        if not content.startswith('#separator:tab\n#html:true\n'):
            logging.error("Invalid file format. Expected #separator:tab and #html:true headers")
            return
            
        # Remove headers
        content = content[content.index('#html:true\n') + len('#html:true\n'):]
        
        # Find all <div> blocks - each div is a complete question
        div_blocks = re.findall(r'<div>.*?</div>', content, re.DOTALL)
        
        # Initialize counters
        cards_processed = 0
        failed_question_matches = 0
        failed_answer_matches = 0
        failed_option_matches = 0
        other_errors = 0
        
        # Process each div block
        for block_num, div_block in enumerate(div_blocks, start=1):
            # Check for explanation before logging
            has_explanation = '<div class="detailed-explanation">' in div_block
            log_message = f"Processing question {block_num}/{len(div_blocks)}"
            if has_explanation:
                log_message += " : OptionalExplanation"
            logging.info(log_message)
            
            try:
                # Extract question
                question_match = re.search(r'<b>Question:</b><br>\s*(.*?)\s*<ul>', 
                                        div_block, re.DOTALL | re.IGNORECASE)
                if not question_match:
                    failed_question_matches += 1
                    logging.warning(f"Question {block_num}: Failed to match question pattern")
                    continue
                
                question = question_match.group(1).strip()
                logging.debug(f"Question {block_num} text: {question[:100]}...")
                
                # Extract options and correct answers
                options = []
                correct_answers = []
                
                # Match options with their letters and correctness
                for match in re.finditer(r"<li class='([^']*)'>(([A-Z])\.\s+(.*?))</li>", div_block):
                    class_value, full_text, letter, option_text = match.groups()
                    options.append((letter, option_text.strip()))
                    if 'correct' in class_value:
                        correct_answers.append(letter)
                
                if not options:
                    failed_option_matches += 1
                    logging.warning(f"Question {block_num}: No options found")
                    continue
                
                # If no correct answers found from classes, try to find from Correct Answer(s) section
                if not correct_answers:
                    answers_match = re.search(r'<b>Correct Answer\(s\):</b>\s*([A-Z](?:\s*,\s*[A-Z])*)', 
                                           div_block, re.DOTALL | re.IGNORECASE)
                    if answers_match:
                        correct_answers = [ans.strip() for ans in answers_match.group(1).split(',')]
                    else:
                        failed_answer_matches += 1
                        logging.warning(f"Question {block_num}: No correct answers found")
                        continue
                
                # Prepare options with correctness
                options_with_correct = ''.join([
                    f'<li class="{"correct" if letter in correct_answers else ""}">{letter}. {text}</li>'
                    for letter, text in options
                ])
                
                # Prepare options without correctness marking
                options_list = ''.join([
                    f'<li>{letter}. {text}</li>'
                    for letter, text in options
                ])
                
                # Prepare correct options string
                correct_options_str = ', '.join(sorted(correct_answers))
                
                # Extract explanation if present
                explanation = ''
                explanation_match = re.search(r'<br><b>Explanation\(s\):</b>\s*(.*?)\s*(?:</div>|$)', 
                                           div_block, re.DOTALL | re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                
                # Add a card to the deck
                my_deck.add_note(genanki.Note(
                    model=my_model,
                    fields=[
                        question,
                        options_list,
                        options_with_correct,
                        correct_options_str,
                        explanation
                    ]
                ))
                
                cards_processed += 1
                
            except Exception as e:
                other_errors += 1
                logging.error(f"Question {block_num}: Error processing question: {str(e)}")
                continue
        
        # Print summary
        logging.info("\n=== Processing Summary ===")
        logging.info(f"Total questions found: {len(div_blocks)}")
        logging.info(f"Successfully processed: {cards_processed}")
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create Anki deck from exam questions')
    parser.add_argument(
        '-i', '--input', 
        required=True,
        help='Input file path'
    )
    parser.add_argument(
        '-o', '--output', 
        required=True,
        help='Output .apkg file path'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the input file
    process_file(args.input)

    # Save the deck to an .apkg file
    genanki.Package(my_deck).write_to_file(args.output)
    print(f"Anki deck created: {args.output}")

if __name__ == "__main__":
    main()
