
import os, re
from typing import Dict, Union, Any, List
import datetime
import uuid

#MODEL RELATED
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_community.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
#AGENT RELATED
from langchain.agents import AgentType,AgentOutputParser,ZeroShotAgent,AgentExecutor
from langchain.schema import AgentAction, AgentFinish,OutputParserException
from langchain.agents import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,ReActSingleInputOutputParser
)

#PROMPT RELATED
from langchain import hub
from langchain_core.prompts import PromptTemplate,FewShotChatMessagePromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description

#MEMORY RELATED
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish

from langchain.agents.mrkl import prompt as react_prompt

#TOOL RELATED

#RETRIEVER


#OTHER
import time, torch, json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from uuid import UUID
from voice import SpeakAgent

from dotenv import load_dotenv

#******  Secrets   *********
os.environ["OPENAI_API_KEY"] =os.getenv["OPENAI_API_KEY"]
os.environ["SERPAPI_API_KEY"] =os.getenv["SERPAPI_API_KEY"]
os.environ["ALPHAVANTAGE_API_KEY"]=os.getenv["ALPHAVANTAGE_API_KEY"]

#******  Custom   *********

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:32"
os.environ['CUDA_VISIBLE_DEVICES']='4,1,2,0,3'
os.environ["PYTORCH_USE_CUDA_DSA"] = "0"
os.environ['HF_HOME']='/home/moebius/Projects/.cache/'
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish

        if "Final Answer:" in llm_output:

            return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": self.parse_finish(llm_output)},
                    log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:

            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": self.parse_finish(llm_output)},
                log=llm_output,
            )
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        #This can't be agent finish because otherwise the agent stops working.
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    def thought_process(self,input_string, new_uuid):

        # Extract the substring after "Begin!"
        start_idx = input_string.find("Begin!") + len("Begin!")
        sub_string = input_string[start_idx:].strip()

        # Define regex patterns to extract relevant parts
        patterns = {
            "memory": re.compile(r"\[.*\]"),
            "question": re.compile(r"Question:\s*(.*)"),
            "thought": re.compile(r"Thought:\s*(.*)"),
            "action": re.compile(r"Action:\s*(.*)"),
            "action_input": re.compile(r"Action Input:\s*(.*)"),
            "observation": re.compile(r"Observation:\s*(.*)"),
            "final_answer": re.compile(r"Final Answer:\s*(.*)")
        }

        # Extract data using regex
        data = {}
        for key, pattern in patterns.items():
            match = pattern.search(sub_string)
            if match:
                try:
                    data[key] = match.group(1).strip()
                except Exception as e:
                    pass
            else:
                data[key] = ""

        # Convert memory to a more readable format
        """
        try:
            data['memory'] = eval(data['memory'])
        except:
            data['memory'] = []
        """

        if "memory" in data:
            file_name = f"./output/memory/memory_{new_uuid}.txt"

            # Save the UUID to a text file
            with open(file_name, 'w') as file:
                file.write(str(data['memory']))

        # Output the dictionary
        #print(json.dumps(data, indent=4))
        jsondata=json.dumps(data, indent=4)
        file_name = f"./output/thoughts/thought_process_{new_uuid}.txt"

        # Save the UUID to a text file
        with open(file_name, 'w') as file:
            file.write(str(jsondata))

    def parse_finish(self,llm_output):
        new_uuid = uuid.uuid4()
        file_name = f"./output/full/full_{new_uuid}.txt"

        # Save the UUID to a text file
        with open(file_name, 'w') as file:
            file.write(str(llm_output))

        self.thought_process(llm_output,new_uuid)
        #return llm_output.split("Final Answer:")[-1].strip()

        parts = llm_output.split("Final Answer:")

        me=False

        if me==True:
            return parts
        else:
            # Get the part after the last "Final Answer:"
            if len(parts) > 1:
                final_answer_part = parts[-1].strip()
            else:
                final_answer_part = llm_output.strip()

            # Find the index of the first newline in this part
            newline_index = final_answer_part.find('\n')

            # If a newline is found, slice the string up to the newline
            # Otherwise, take the whole string
            if newline_index != -1:
                result = final_answer_part[:newline_index].strip()
            else:
                result = final_answer_part

            return result
#******  Langchain   *********


# Initialize Langchain components

#******  Custom   *********


#******  Agent   *********

class Agent():

    def __init__(self, system_prefix, checkpoint, agent_type= "default") -> None:

        print("******************  Initializing Matt...  ************************")
        self.checkpoint=checkpoint
        self.speaker = SpeakAgent()
        filename = "./data/few_shot_mistral.json"
        with open(filename, 'r') as file:
            fewshot_examples = json.load(file)

        quantization_config = BitsAndBytesConfig(llm_int8_threshold=50.0,load_in_8bit=True,llm_int8_enable_fp32_cpu_offload=True)

        # First load the "brain" of the organization
        #https://docs.mistral.ai/api/#operation/createChatCompletion

        model_kwargs = {
            "max_length": 8192,
            "device_map": "auto",
            "offload_folder": "offload",
            "max_memory": {4: "12GB", 1: "8GB", 2: "8GB", 0: "8GB", 3: "11GB"},
            "quantization_config": quantization_config,

        }

        llm = HuggingFacePipeline.from_model_id(
            model_id=checkpoint,
            task="text-generation",
            device=None,
            pipeline_kwargs={
                "max_new_tokens": 2048,  # changed from 2048
                "top_p": 0.15,  # changed from 0.15
                "do_sample": True,  # changed from true
                "torch_dtype": torch.bfloat16,  # bfloat16
                "use_fast": True,
            },
            model_kwargs=model_kwargs
        )

        tools = load_tools(["human", "serpapi","google-finance","wikipedia","llm-math"], llm=llm)

        date_suffix = f" Today's date is { datetime.now().strftime('%A')},{ datetime.now() }."
        # Third define the way the agent should act
        basic_suffix =  """
                        Begin!
                        {chat_history}
                        Question: {input}
                        {agent_scratchpad}
                       """

        example_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ('human', "{input}"),
                ('ai', "{output}")
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=fewshot_examples,
            example_prompt=example_prompt,
            input_variables=["input",
                             "agent_scratchpad"],
        )

        format_instructions = f"{react_prompt.FORMAT_INSTRUCTIONS}\n " \
                              f"Here are some examples of user inputs and " \
                              f"their corresponding ReAct responses:\n"

        # definition_prompts="You are so smart"
        template = "\n\n".join(
            [
                system_prefix + " " + date_suffix,
                "{tools}",
                format_instructions,
                few_shot_prompt.format(),
                basic_suffix
            ]
        )

        prompt = PromptTemplate.from_template(template=template)
        prompt = prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools]),
        )
        #tell me a bunny joke

        # Fourth ensure that the agent has access to memory
        memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix='Assistant',human_prefix='User',input_key='input', output_key="output", return_messages=True)

        # Fifth define when the brain should stop thinking
        chat_model_with_stop = llm.bind(stop=["\nFinal Answer:", "\nHuman:","\nObservation:"])

        # Sixth define the agent with inout and scratchpad
        agent = create_react_agent(
            llm=chat_model_with_stop,
            tools=tools,
            prompt=prompt,
            output_parser=CustomOutputParser()
        )

        #"agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        # Seventh define the executor with the agent, its tools, memory and an execution stop.
        self.agent_executor = AgentExecutor(agent=agent, tools=tools,memory=memory,handle_parsing_errors=True,verbose=False,max_iterations=5)
        #handle_parsing_errors=False,

    def get_agent_response(self,prompt_text):
        torch.cuda.empty_cache()
        input_text={"input": prompt_text.strip()}
        response = self.agent_executor.invoke(input_text)
        #print(dict(response)['output'])
        return dict(response)


    def speak(self, txt, store_local=True):
        wav_data=self.speaker.speak(txt,store_local)
        return wav_data
