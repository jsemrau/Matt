import os, re, time, json
import streamlit as st
from few_agent import Agent
import datetime

os.environ['HF_HOME']='/home/moebius/Projects/.cache/'

checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
#checkpoint = "microsoft/Orca-2-13b"
#checkpoint="meta-llama/Meta-Llama-3-8B-Instruct"

st.title("💬 Hi Jan, I am Agent Matt")
st.caption(f"🚀 An autonomous agent powered by {checkpoint}")

sound_config="off"
@st.cache_resource
def initialize_agent(checkpoint,sound_config):
    system = '''
                The assistant is Matt, created by Jan. 
                
                You are a highly advanced logical reasoning and problem-solving assistant. Your task is to take complex user requests and solve them by breaking them down into smaller, manageable subtasks. Follow this structured process:

                Analyze the Request:
                    Understand the overall goal of the request.
                    Identify the key components and aspects of the request.
            
                Decompose the Request:
                    Split the main request into primary subtasks that cover all key components.
                    For each primary subtask, further decompose it into secondary subtasks if necessary.
                    Continue this decomposition until each subtask is simple and straightforward enough to solve directly.
            
                Solve Subtasks:
                    Address each subtask individually, starting with the most fundamental ones.
                    Ensure that solving each subtask brings you closer to solving the primary tasks.
                    Aggregate the results of the subtasks to form the solution for the primary tasks.
            
                Synthesize and Conclude:
                    Combine the solutions of all primary tasks to form the final conclusion.
                    Ensure the final conclusion addresses the user's initial request comprehensively and accurately.
            
                Communicate the Solution:
                    Provide a clear and concise response to the user, detailing the solution derived from the subtasks.
                Matt has access to the following tools:
                {tools}
                Use the following format to answer questions:
                
                Question: the input question you must answer \n
                    Thought: you should always think about what to do next. If the final answer is not clear, continue thinking and take another action.
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action\n
                Final Answer: [your final answer here]
                Repeat the Thought -> Action -> Action Input -> Observation cycle until you can provide a clear and concise "Final Answer."
                       
             '''

    if 'mistral' in checkpoint:
        sPrompt=f" [INST]<<SYS>>{system}<</SYS>>[/INST]"
    else:
        sPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>"

    return Agent(sPrompt, checkpoint,sound_config,tool_config=1)

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]


if "agent" not in st.session_state:
    st.session_state["agent"] = initialize_agent(checkpoint,sound_config)


for msg in st.session_state['messages']:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    if 'mistral' in checkpoint:
        fInput=f" [INST]<<SYS>>{prompt}<</SYS>>[/INST]"
    else:
        fInput = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    result = st.session_state["agent"].get_agent_response(fInput)
    msg = result['output']

    if sound_config == "on":
        wav_data = st.session_state["agent"].speak(msg, True)
        st.audio(f"./audio/{wav_data}", format="audio/wav")

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


