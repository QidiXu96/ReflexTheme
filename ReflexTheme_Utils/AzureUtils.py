"""
Module: AzureUtils.py
Description: Contains utility functions for interacting with the Azure OpenAI API and documents
"""
from openai import AzureOpenAI
from docx import Document

_client = None
def init_client(azure_endpoint, api_key, api_version):
    global _client
    _client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )

def get_completion(messages, model="gpt-4o-mini-Qidi", max_tokens=16384, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0):
    if _client is None:
        raise Exception("AzureOpenAI client not initialized. Please call init_client() with your parameters first.")
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response

def read_docx(file_path):
    """
    Extract text from a DOCX file.
    """
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)
