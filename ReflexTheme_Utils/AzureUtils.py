"""
Module: AzureUtils.py
Description: Contains utility functions for interacting with the Azure OpenAI API and documents
"""
from openai import AzureOpenAI
from docx import Document

def create_client(azure_endpoint, api_key, api_version):
    """
    Create an Azure OpenAI client.
    Parameters:
        azure_endpoint (str): Your Azure endpoint URL.
        api_key (str): Your API key.
        api_version (str): The API version to use.
    Return Azure OpenAI client
    """
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )

def get_completion(client, model, messages, max_tokens=16384, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0):
    """
    Get a chat completion from Azure OpenAI.

    Parameters:
        client (AzureOpenAI): Initialized Azure OpenAI client.
        model (str): Azure deployment/model name.

    Returns:
        ChatCompletion: Raw Azure OpenAI response.
    """
    response = client.chat.completions.create(
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
