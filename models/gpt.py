from langchain_openai import ChatOpenAI
from configuration.config_loader import config
from .models import Models


class Gpt(Models):
    """
    A provider class for interacting with OpenAI's GPT models via
    the official OpenAI API using LangChain's ChatOpenAI interface.

    Inherits from the base Models class and follows a fluent builder
    pattern, allowing model selection and instantiation to be chained.

    Available Models:
        - gpt-4o       → Most capable, multimodal (text + vision)
        - gpt-4o-mini  → Lightweight, faster, cost-efficient version of 4o
        - gpt-4-turbo  → High capability with large 128k context window
        - gpt-3.5-turbo → Fast and cost-effective for simpler tasks

    Usage:
        llm = Gpt().gpt_4o().model()
        llm = Gpt().gpt_4o_mini().model()

    Attributes:
        gpt_model (str): The OpenAI model identifier to use for inference.
    """

    def __init__(self):
        """
        Initialize the Gpt provider with an empty model identifier.
        Call a model selector method (e.g. gpt_4o()) before calling
        model() to build the LLM instance.
        """
        self.gpt_model = ''

    def gpt_4o(self) -> "Gpt":
        """
        GPT-4o is OpenAI's most capable multimodal model, supporting
        text and vision inputs with strong reasoning and instruction
        following capabilities.
        """
        self.gpt_model = 'gpt-4o'
        return self

    def gpt_4o_mini(self) -> "Gpt":
        """
        GPT-4o Mini is a smaller, faster, and more cost-efficient
        variant of GPT-4o. Suitable for tasks that don't require
        the full capability of GPT-4o.
        """
        self.gpt_model = 'gpt-4o-mini'
        return self

    def gpt_4o_turbo(self) -> "Gpt":
        """
        GPT-4 Turbo offers high capability with a large 128k token
        context window, making it suitable for tasks that require
        processing long documents or extended conversations.
        """
        self.gpt_model = 'gpt-4-turbo'
        return self

    def gpt_3_5_turbo(self) -> "Gpt":
        """
        GPT-3.5 Turbo is a fast and cost-effective model suitable
        for simpler tasks like summarization, classification, and
        basic question answering where GPT-4 is not required.
        """
        self.gpt_model = 'gpt-3.5-turbo'
        return self

    def model(self) -> ChatOpenAI:
        """
        Build and return a ChatOpenAI instance configured with the
        selected GPT model and OpenAI API credentials.

        Returns:
            ChatOpenAI: A LangChain-compatible LLM instance ready for
                        invocation, tool binding, or use in agents/graphs.

        Raises:
            ValueError: If gpt_model is empty (no model selected),
                        the API will reject the request. Always call a
                        model selector method before calling model().

        Note:
            max_tokens is set to 20 — increase this for tasks requiring
            longer responses such as summarization or code generation.

        """
        return ChatOpenAI(
            model=self.gpt_model,
            api_key=config.gpt_key,
            temperature=0,
            max_tokens=20
        )
