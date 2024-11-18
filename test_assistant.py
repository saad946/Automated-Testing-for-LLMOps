from app import assistant_chain
from app import system_message
from langchain.prompts                import ChatPromptTemplate
from langchain.chat_models            import ChatOpenAI
from langchain.schema.output_parser   import StrOutputParser
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def eval_expected_words(
    system_message,
    question,
    expected_words,
    human_template="{question}",
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):

  assistant = assistant_chain(system_message)
  answer = assistant.invoke({"question": question})
  print(answer)
    
  assert any(word in answer.lower() \
             for word in expected_words), \
    f"Expected the assistant questions to include \
    '{expected_words}', but it did not"

def evaluate_refusal(
    system_message,
    question,
    decline_response,
    human_template="{question}", 
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):
    
  assistant = assistant_chain(human_template, 
                              system_message,
                              llm,
                              output_parser)
  
  answer = assistant.invoke({"question": question})
  print(answer)
  
  assert decline_response.lower() in answer.lower(), \
    f"Expected the bot to decline with \
    '{decline_response}' got {answer}"

"""
  Test cases
"""

def test_science_quiz():
  
  question  = "Generate a quiz about science."
  expected_subjects = ["davinci", "telescope", "physics", "curie"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)

def test_geography_quiz():
  question  = "Generate a quiz about geography."
  expected_subjects = ["paris", "france", "louvre"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)


def test_assistant_science_quiz():
    assistant = assistant_chain()
    question = "Generate a quiz about science."
    answer = assistant.invoke({"question": question})
    expected_subjects = ["davinci", "telescope", "physics", "curie"]
    print(answer)
    assert any(
        subject.lower() in answer.lower() for subject in expected_subjects
    ), f"Expected the assistant questions to include '{expected_subjects}', but it did not"


def test_assistant_geography_quiz():
    assistant = assistant_chain()
    question = "Generate a quiz about geography."
    answer = assistant.invoke({"question": question})
    expected_subjects = ["paris", "france", "louvre"]
    print(answer)
    assert any(
        subject.lower() in answer.lower() for subject in expected_subjects
    ), f"Expected the assistant questions to include '{expected_subjects}', but it did not"


def test_assistant_unknown_subjects():
    assistant = assistant_chain()
    question = "Generate a quiz about Rome"
    answer = assistant.invoke({"question": question})
    print(answer)
    # We'll look for a substring of the message the bot prints when it gets a question about any
    decline_response = "I'm sorry"
    assert (
        decline_response.lower() in answer.lower()
    ), f"Expected the bot to decline with '{decline_response}' got {answer}"


