import os, re, time, json
import streamlit as st
from few_agent import Agent
import datetime

os.environ['HF_HOME']='/home/moebius/Projects/.cache/'

checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"

st.title("💬 Hi Jan, I am Agent Matt")
st.caption(f"🚀 An autonomous agent powered by {checkpoint}")

sound_config="off"
@st.cache_resource
def initialize_agent(checkpoint,sound_config):
    system = '''
                [INST]<<SYS>>
                            The assistant is Matt, created by Jan. 
                            
                            Matt's knowledge base was last updated on August 2023. 
                            It anwers questions about events prior to and after August 2023 the way a highly informed individual in August 2023
                            would if they were talking to someone from the above date, and can let the human know this when rlevant.
                            
                            It should give concise responses to very simple questions, but provide thourough responses to more complex and open-ended questions.
                            
                            Matt is jappy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks.
                            It uses markdown for coding.
                            
                            Matt does not mention this information about itself unless the information is directly pertinent to the human's query.
                      
                            You have access to the following tools:
                            [AVAILABLE_TOOLS]
                            {tools}
                            [/AVAILABLE_TOOLS]
                            Use the following format:\n                            
                            Question: the input question you must answer
                            Thought: you should always think about what to do
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            Observation: the result of the action
                            Final Answer: I know now the final answer
                       <</SYS>>
                        [/INST]
             '''

    return Agent(system, checkpoint,sound_config)

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]


if "agent" not in st.session_state:
    st.session_state["agent"] = initialize_agent(checkpoint,sound_config)


for msg in st.session_state['messages']:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    fInput = f"<s>[INST]{prompt}[/INST]"
    result = st.session_state["agent"].get_agent_response(fInput)
    msg = result['output']

    if sound_config == "on":
        wav_data = st.session_state["agent"].speak(msg, True)
        st.audio(f"./audio/{wav_data}", format="audio/wav")

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


