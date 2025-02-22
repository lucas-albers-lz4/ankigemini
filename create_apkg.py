import genanki
import re
import html  # Import the html module for escaping HTML
import argparse  # Add argparse import
import os
from dotenv import load_dotenv

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
    with open(file_path, 'r') as file:
        content = file.read()
        questions = re.split(r'\n\s*\n', content)  # Split by double newlines

        for question_block in questions:
            # Extract question and answer
            question_match = re.search(r'<b>Question:</b><br>(.*?)<ul>(.*?)</ul>.*?<b>Correct Answer\(s\):</b>\s*([A-Z,\s]+)', question_block, re.DOTALL)
            if question_match:
                # Extract components
                question = html.escape(question_match.group(1).strip())
                options_html = question_match.group(2)
                correct_answers = question_match.group(3).strip().split(',')
                
                # Parse options
                answers = re.findall(r"<li(?:\s+class='.*?')?>([^<]+)</li>", options_html)
                
                # Prepare options with correctness
                options_with_correct = '<ul>' + ''.join([
                    f'<li class="{("correct" if chr(65 + i) in correct_answers else "incorrect")}">{html.escape(opt)}</li>' 
                    for i, opt in enumerate(answers)
                ]) + '</ul>'
                
                # Prepare options without correctness marking
                options = '<ul>' + ''.join([
                    f'<li>{html.escape(opt)}</li>' 
                    for opt in answers
                ]) + '</ul>'
                
                # Prepare correct options
                correct_options_str = ', '.join(correct_answers)

                # Add a card to the deck
                my_deck.add_note(genanki.Note(
                    model=my_model,
                    fields=[
                        question, 
                        options, 
                        options_with_correct, 
                        correct_options_str
                    ]
                ))

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

def main(input_file=None, output_file=None):
    """
    Alternate main function that can be called directly with file paths.
    Useful for testing and programmatic use.
    
    :param input_file: Path to input file (optional, overrides command line)
    :param output_file: Path to output file (optional, overrides command line)
    """
    # If called without arguments, use command line parsing
    if input_file is None or output_file is None:
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
        input_file = args.input
        output_file = args.output
    
    # Reset the deck to ensure clean state for testing
    global my_deck
    my_deck = genanki.Deck(
        DECK_ID,  # Unique ID for the deck
        DECK_NAME
    )

    # Process the input file
    process_file(input_file)

    # Save the deck to an .apkg file
    genanki.Package(my_deck).write_to_file(output_file)
    print(f"Anki deck created: {output_file}")

    # Return the deck for testing purposes
    return my_deck

if __name__ == "__main__":
    main()
