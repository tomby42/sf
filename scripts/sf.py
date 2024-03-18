import os
from typing import Any, List, Mapping, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from transformers import pipeline

import requests
import json

from bs4 import BeautifulSoup
import mistune
import pypandoc

# TODO: specify custom model
class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

# TODO: init custom model
llm = ChatOpenAI()

def prep_url_links(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    links = soup.find_all('a')
    return "".join(map(lambda lnk: str(lnk), links))

def parse_links(links):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract pricing link from following HTML code. Answer should be only URL address."),
        ("user", "{input}")
    ])
    chain = prompt | llm
    return chain.invoke({"input": links})

def prep_prices(url):
    req = requests.get(url)
    return pypandoc.convert_text(req.text, 'md', format='html')

def parse_prices(prices):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract prices from following markdown document and create table in JSON format. Answer should be only the table in JSON format."),
        ("user", "{input}")
    ])
    chain = prompt | llm
    return json.loads(chain.invoke({"input": prices}))

def get_prices(url):
    links = prep_url_links(url)
    price_url = parse_links(links)
    prices = prep_prices(price_url)
    return parse_prices(prices)

### API

from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/prices')
def predict():
    content = request.get_json(silent=True)
    return get_prices(content.url)

if __name__ == '__main__':
    app.run(host= '0.0.0.0:8000',debug=True)
