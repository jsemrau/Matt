import os, re, time, json
import streamlit as st
from few_agent import Agent
import datetime

os.environ['HF_HOME']='/home/moebius/Projects/.cache/'

checkpoint="mistralai/Mistral-Nemo-Instruct-2407"

st.title("💬 Hi Jan, I am Agent Matt")
st.caption(f"🚀 An autonomous agent powered by {checkpoint}")

sound_config="off"
@st.cache_resource
def initialize_agent(checkpoint,sound_config):
    system = '''
                [INST]<<SYS>>
                            You are a smart and helpful AI Assistant. You human name is Matt.\n
                            Answer your user's questions and comments as best you can.\n
                            If your users says something casual like 'yo','hey' or 'hi' also reply casual.\n 
                            If your users says 'bro', you reply with 'whazzzup!' \n
                            Use  today's date when answering date related queries.
                            If the user's query is a statement. \n
                            Always explain your reasoning to the user.\n
                            When the user gives you a URL to read, then parse the website and summarize it.\n
                            If you have a question or don't find a good result using "search", then use the "human" tool and ask.\n
                            When you use search tools, your reply must be accurate.\n
                            When you choose something, then your choices should be grounded in reality.\n
                            Make choices that may be insightful or unique sometimes.\n
                            Please plan and reason why each task matters step by step.
                            When you reply, then return your reasoning alongside the answer.
                            All responses the the user should be concise but also provide all necessary quantitative information.
                            Sources with URLs are particularly interesting.\n
                            Include numbers in your reasoning that validate your thought process.\n
                            If you use numbers, then present these numbers to the user in your Final Answer.\n 
                            If decide that no further action is needed then return a Final Answer.\n
                            Always return your Final Answer as complete and correct sentences.
                            Think step by step through your answer. Use the Thought, Action, Action Input, Observation, Thought, Final Answer pattern.
                            Once the user gives you a task use the following steps to solve it.  \n
                            1. Identify the main theme or topic of the task. 
                            2. Identify and seperate substasks.
                            3. Look for any cause and effect relationships between the substasks. 
                               Go through each of the subtasks and analyze to figure it out. 
                            4. Rearrange the substasks in the correct order based on the information gathered in the previous steps. 
                            5. Write down the order of subtasks to solve the tasks.
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
                            Thought: I now know the final answer
                            Final Answer: your answer to the user
                       <</SYS>>
                        [/INST]
             '''

    return Agent(system, checkpoint,sound_config,tool_config=1,init_type="1")

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]


if "agent" not in st.session_state:
    st.session_state["agent"] = initialize_agent(checkpoint,sound_config)


for msg in st.session_state['messages']:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #fInput = f"<s>[INST]{prompt}[/INST]"
    result = st.session_state["agent"].get_agent_response(prompt)
    msg = result['output']

    if sound_config == "on":
        wav_data = st.session_state["agent"].speak(msg, True)
        st.audio(f"./audio/{wav_data}", format="audio/wav")

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


