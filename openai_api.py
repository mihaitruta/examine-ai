import streamlit as st
import openai
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import random
from utils import (parse_evaluation, get_color, text_to_html, calculate_average, 
                   get_random_score, num_tokens_from_string, api_details)
from prompts import safeguard_assessment
import time
from datetime import datetime
import logging
import pyperclip


class OpenAIResponder:
    """
    A class to handle responses from OpenAI's GPT model.
    """
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo-0613'):
        self._api_key = api_key
        self._model = model

    def get_response(self, messages: List[Dict[str, str]]):
        """Fetches a response from OpenAI using the given list of message dictionaries."""
        content = ''
        status = 'ERR'
        details = None
        try:
            total_toks = 0
            valid_messages = []
            msg_nr = 0
            first_msg_idx = 0
            for msg in messages:
                if msg['role'] in ['system', 'assistant', 'user', 'function']:
                    if status not in msg or msg['status'] in ['OK', 'WARN']:
                        valid_messages.append({'role': msg['role'], 'content': msg['content']})
                        total_toks += num_tokens_from_string(msg['content'], "cl100k_base")
                        msg_nr += 1
                        
            details = {}
            if total_toks + 100 >= api_details[self._model]['context']:
                details['chat_length_warning'] = True
                st.session_state.chat_length_warning = True
                extra_toks = total_toks + 100 - api_details[self._model]['context']
                start_toks = 0
                for idx, msg in enumerate(valid_messages):
                    start_toks += num_tokens_from_string(msg['content'], "cl100k_base")
                    if start_toks >= extra_toks:
                        first_msg_idx = idx + 1
                        break
                print('dropping first ', first_msg_idx, ' messsages')
            if first_msg_idx <= msg_nr - 1:
                valid_messages = valid_messages[first_msg_idx:]
            else:
                raise Exception("The message you entered does not fit into the context of the selected model.")      
            
            response = openai.ChatCompletion.create(
                model=self._model, messages=valid_messages
            )

            
            details['received'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details['finish_reason'] = response.choices[0].finish_reason
            details['created'] = response.created
            details['id'] = response.id 
            details['model'] = response.model
            details['usage'] = response.usage
            
            model = response.model
            usage = response.usage
            status = 'OK' if details['finish_reason'] == 'stop' else 'WARN'
            return response.choices[0].message['content'], status, details

        except openai.error.Timeout as e:
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIError as e:
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.APIConnectionError as e:
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.AuthenticationError as e:
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.PermissionError as e:
            content = f"OpenAI API request was not permitted: {e}"
        except openai.error.RateLimitError as e:
            content = f"OpenAI API request exceeded rate limit: {e}"
        except Exception as e:
            content = f"{e}"

        return content, status, details



def _test_responder():
    model = 'gpt-3.5-turbo-0613'

    prompt = "Q: Who starred in the 1996 blockbuster Independence Day?"
    prompt += "A: "
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    responder = OpenAIResponder(api_key=openai.api_key)
    
    response, status, details = responder.get_response([{'role': 'system', 'content': prompt}])
    
    print('response ', response)
    print('status ', status)
    print('details ', details)




if __name__ == "__main__":
    _test_responder()
    