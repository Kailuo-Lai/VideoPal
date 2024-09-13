from langchain.agents import Tool
from langchain import LLMChain
from langchain.agents import BaseSingleActionAgent, AgentOutputParser
from typing import List, Tuple, Any, Union, Dict
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.callbacks.manager import Callbacks
import re

agent_template = """
Background:{background}

{instruction}

Query: "{query}"

{response}

"""

# agent_template = """
# 你现在是{role}。

# {instruction}

# {tool_desctibe}

# 你目前的信息有:{info}。

# {question_guide}:{input}

# {answer_format}

# Your Response:

# """

# Define the agent class
class LLMSingleActionAgent(BaseSingleActionAgent):
    """Base class for single action agents."""

    llm_chain: LLMChain
    """LLMChain to use for agent."""
    output_parser: AgentOutputParser
    """Output parser to use for agent."""
    stop: List[str]
    """List of strings to stop on."""

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        Returns:
            List of input keys.
        """
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        # import pdb; pdb.set_trace()
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output,input=kwargs["query"])

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output,input=kwargs["query"])

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }
        
# Define the custom prompt template and output parser
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
    
        if len(intermediate_steps) == 0:
            background = """The user is querying a database of videos. The intent of the query needs to be identified to retrieve the relevant video information. It's crucial to accurately classify the query to provide the most relevant information to the user."""
            instruction = """
            Clarifications:
            - 'Time-based Retrieval' refers only to queries that ask for information about specific timestamps or durations within the video content itself, such as seeking what happens at a particular minute and second.
            - 'Entity-based Retrieval' involves queries focused on specific entities within the video, such as people, places, things, or concepts, and seeks detailed information about these entities.
            - 'Summary-based Retrieval' queries require an overarching summary of the video's content.

            Examples of Correct Classifications:
            1. "What are the four basic principles of successful survival?" - This query seeks detailed information about an entity (principles, survival) and should be classified as Entity-based Retrieval, not Time-based Retrieval, as it does not seek information tied to a specific timestamp or duration within the video.
            2. "What is happening between 34 minutes 23 seconds and 34 minutes 32 seconds in the video?" - Time-based Retrieval
            3. "Who is the director of 'Inception'?" - Entity-based Retrieval
            4. "Summarize the entire video in detail." - Summary-based Retrieval
            5. "What speed is the person skydiving in the video?" - This query seeks detailed information about an entity's (the person skydiving) specific aspect (speed) and should be classified as Entity-based Retrieval, not Time-based Retrieval, as it does not seek information tied to a specific timestamp or duration within the video.
            6. "What is the terminal velocity of the protagonist when he parachutes and lands?" - This query seeks detailed information about an entity (velocity) and should be classified as Entity-based Retrieval, not Time-based Retrieval, as it does not seek information tied to a specific timestamp or duration within the video.
            
            Now, classify the following query into one of three categories based on the above definitions and clarifications. Remember, the classification depends on the focus of the query - whether it is on specific times, entities within the video, or requires a summary of content."""
            
            
            response = """
            Categories:
            1. Time-based Retrieval
            2. Entity-based Retrieval
            3. Summary-based Retrieval

            Intent:
            """
                    
        else:
            background = """The user is querying a database of videos. You should answer the question based on the following information:"""
            instruction = "Information: "
            action, observation = intermediate_steps[0]
            instruction += f"{observation}\n"
            
            response = """Your Response is:"""

        kwargs["background"] = background
        kwargs["instruction"] = instruction
        kwargs["response"] = response

        return self.template.format(**kwargs)

## Define the custom output parser, which classifies the query into one of three categories    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str,input:str) -> Union[AgentAction, AgentFinish]:
        
        pattern1 = r"Time-based Retrieval"
        pattern2 = r"Entity-based Retrieval"
        pattern3 = r"Summary-based Retrieval"

        match1 = re.search(pattern1, llm_output, re.IGNORECASE)
        match2 = re.search(pattern2, llm_output, re.IGNORECASE)
        match3 = re.search(pattern3, llm_output, re.IGNORECASE)

        found1 = bool(match1)
        found2 = bool(match2)
        found3 = bool(match3)

        if found1:
            action = "get_information_from_detailed_time"
            action_input = input

        elif found2:
            action = "get_information_from_detailed_objects"
            action_input = input
            
        elif found3:
            action = "get_information_from_video_summary"
            action_input = input
        else:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )

        return AgentAction(tool=action, tool_input=action_input, log=llm_output)