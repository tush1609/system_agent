from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from app.bridge import Bridge

class Models(ABC):
    """
    Abstract base class defining the contract for all LLM provider classes.

    This class acts as an interface that every LLM provider (e.g. HuggingFace,
    OpenAI, Anthropic) must implement. It enforces a consistent API across
    all providers so they can be used interchangeably throughout the codebase.

    Design Pattern:
        Strategy Pattern — each provider subclass is a concrete strategy
        that implements the same model() interface, allowing the caller
        to swap providers without changing any downstream code.

    Usage:
        Never instantiate Models directly. Use a concrete subclass:

            llm = Huggingface().qwen_2_5_72b_instruct().model()
            llm = OpenAIProvider().gpt_4o().model()

    Raises:
        TypeError: If a subclass does not implement the model() method,
                   Python will raise TypeError on instantiation.
    """

    @abstractmethod
    def model(self) -> ChatOpenAI:
        """
        Build and return a configured LLM instance.

        Every provider subclass must implement this method and return
        a LangChain-compatible ChatOpenAI (or compatible) instance
        ready for invocation, tool binding, or use in LangGraph agents.

        Returns:
            ChatOpenAI: A configured LangChain LLM instance.

        Raises:
            NotImplementedError: Implicitly raised if subclass does not
                                 override this method.

        Example (in a concrete subclass):
            def model(self) -> ChatOpenAI:
                return ChatOpenAI(
                    model=self.gpt_model,
                    api_key=config.api_key,
                    base_url='https://api.example.com/v1',
                    temperature=0
                )
        """
        pass


    async def tool_response(self, msg_for_tool: str,  applicable_tools: list):
        response = self.model().bind_tools(applicable_tools).invoke(msg_for_tool)
        if response.content:
            await self.stream_message(response.content)

        return response.content, response.tool_calls


    async def tool_execution_response(self, tool_node, tool_query):
        response = tool_node.invoke(tool_query)
        for resp in response["messages"]:
            await self.stream_message(resp.content)

        return response["messages"]


    @staticmethod
    async def stream_message(messages):
        await Bridge().stream_start()
        for chunk in messages:
            await Bridge().stream_message(chunk)
        await Bridge().stream_end()


    async def stream_llm_response(self, llm_input):
        await Bridge().stream_start()
        full_text = ''
        async for chunk in self.model().astream(llm_input):
            await Bridge().stream_message(chunk.content)
            print('chunk is ', chunk)
            full_text += chunk.content or ""
        await Bridge().stream_end()

        return full_text
