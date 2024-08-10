
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers import GenerationConfig

from transformers import (AutoModelForSeq2SeqLM, AutoModelForCausalLM,
                          AutoTokenizer, GenerationConfig, TrainingArguments, Trainer)

device = 'cpu'
torch_device = torch.device(device)

model_name = 'google/flan-t5-base'
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

peft_model = PeftModel.from_pretrained(
    peft_model_base, "C:/Users/Acer1/PycharmProjects/pythonProject/checkpoint-1000"
).to(torch_device)


class TextSummarizer:
    st.title("Text Summarizer")
    st.write("Enter the text you want to summarize")


    def start_chat(self):

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        self.chat(st.chat_input("Enter or paste text here"))

    def chat(self, user_prompt: str):
        user_prompt = st.chat_input("Enter or paste text here")

        # accept user input
        if user_prompt := st.chat_input("Enter or paste text here"):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_prompt)

        reply = st.chat_input(self.summarize_prompt(user_prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(reply)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": reply})

        yield reply
    def summarize_prompt(self, user_prompt):
        user_prompt = st.text_area("Input Text", height=200)
        if st.button("Summarize"):
            if user_prompt:
                inputs = tokenizer(user_prompt, return_tensors='pt').input_ids
                summary = peft_model.generate(input_ids=inputs,
                                              generation_config=GenerationConfig
                                              (max_new_tokens=200, num_beams=1))
                output = tokenizer.decode(summary[0], skip_special_tokens=True)

                st.subheader("Summary:")
                st.write(output)
            else:
                st.write("Please enter some text to summarize.")

summarizer = TextSummarizer()
summarizer.start_chat()
