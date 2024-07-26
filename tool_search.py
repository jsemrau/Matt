import os, re, time, json
import streamlit as st
from few_agent import Agent
import datetime

os.environ['HF_HOME']='/home/moebius/Projects/.cache/'

#checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
#checkpoint = "microsoft/Orca-2-13b"
#checkpoint="meta-llama/Meta-Llama-3-8B-Instruct"
#checkpoint="internlm/internlm2_5-7b"
#checkpoint="stabilityai/StableBeluga-13B"
checkpoint="mistralai/Mistral-Nemo-Instruct-2407"
#checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct"

st.title("💬 Hi Jan, I am Agent Matt")
st.caption(f"🚀 An autonomous agent powered by {checkpoint}")

sound_config="off"
@st.cache_resource
def initialize_agent(checkpoint,sound_config):
    system = '''
               
                            The assistant is Matt, created by Jan. 
                            
                            It anwers questions about events the way a highly informed individual would if they were talking to someone from the today's date, and can let the human know this when relevant.
                            
                            It should give concise responses to very simple questions, but provide thourough responses to more complex and open-ended questions.
                            
                            Matt is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks.
                            It uses markdown for coding.
                            
                            Matt does not mention this information about itself unless the information is directly pertinent to the human's query.
                            
                            When using the "human" tool stop thinking formulate a question as your final answer.
                             
                            Matt reasons step by step through your answer. \n
                            Use the Thought, Action, Action Input, Observation, Thought, Final Answer pattern.
                            Once the user gives you a task use the following steps to solve it.  \n
                            1. Identify the main theme or topic of the task. 
                            2. Identify and seperate substasks.
                            3. Look for any cause and effect relationships between the substasks. 
                               Go through each of the subtasks and analyze to figure it out. 
                            4. Rearrange the substasks in the correct order based on the information gathered in the previous steps. 
                            5. Write down the order of subtasks to solve the tasks.
                             
                            Matt has access to the following tools:
                            {tools}
                            Use the following format to answer questions:
                            
                                Question: the input question you must answer \n
                                    Thought: you should always think about what to do next. If the final answer is not clear, continue thinking and take another action.
                                    Action: the action to take, should be one of [{tool_names}]
                                    Action Input: the input to the action
                                    Observation: the result of the action\n
                                Final Answer: [your observation and conclusion]
                                
                            Repeat the Thought -> Action -> Action Input -> Observation cycle until you can provide a clear and concise "Final Answer."
                    
                       
             '''

    if 'mistral' in checkpoint:
        sPrompt=f"[INST]<<SYS>>{system}<</SYS>>[/INST]"
    elif 'llama' in checkpoint or 'Orca' in checkpoint:
        sPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>"
    else:
        sPrompt=system

    return Agent(sPrompt, checkpoint,sound_config,tool_config=3)

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
    elif 'llama' in checkpoint or 'Orca' in checkpoint:
        fInput = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        fInput=prompt

    result = st.session_state["agent"].get_agent_response(fInput)
    msg = result['output']

    if sound_config == "on":
        wav_data = st.session_state["agent"].speak(msg, True)
        st.audio(f"./audio/{wav_data}", format="audio/wav")

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


