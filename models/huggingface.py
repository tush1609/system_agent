from .models import Models
from langchain_openai import ChatOpenAI
from configuration.config_loader import config


class Huggingface(Models):
    """
    A provider class for interacting with HuggingFace-hosted LLMs
    via the HuggingFace router API using the OpenAI-compatible interface.

    Inherits from the base Models class and follows a fluent builder pattern,
    allowing model selection and instantiation to be chained together.

    Usage:
        llm = Huggingface().qwen_2_5_72b_instruct().model()

    Attributes:
        gpt_model (str): The HuggingFace model identifier to use.
    """

    def __init__(self):
        """
        Initialize the Huggingface provider with an empty model identifier.
        Call a model selector method (e.g. qwen_2_5_72b_instruct()) before
        calling model() to build the LLM instance.
        """
        self.gpt_model = ''

    def qwen_2_5_72b_instruct(self) -> "Huggingface":
        """
        Select the Qwen 2.5 72B Instruct model as the active model.

        Qwen2.5-72B-Instruct is a large instruction-tuned language model
        by Alibaba Cloud, capable of complex reasoning, coding, and
        multi-turn conversations.

        Returns:
            Huggingface: Returns self for method chaining.

        Example:
            llm = Huggingface().qwen_2_5_72b_instruct().model()
        """
        self.gpt_model = 'Qwen/Qwen2.5-72B-Instruct'
        return self

    def model(self) -> ChatOpenAI:
        """
        Build and return a ChatOpenAI instance configured to use the
        HuggingFace router API with the selected model.

        The HuggingFace router API is OpenAI-compatible, so ChatOpenAI
        is used with a custom base_url pointing to HuggingFace's endpoint.

        Returns:
            ChatOpenAI: A LangChain-compatible LLM instance ready for
                        invocation, chaining, or use in agents/graphs.

        Raises:
            ValueError: If gpt_model is not set (empty string), the API
                        will return a 404. Always call a model selector
                        method before calling model().

        Example:
            llm = Huggingface().qwen_2_5_72b_instruct().model()
            response = llm.invoke("What is the capital of France?")
        """
        return ChatOpenAI(
            model=self.gpt_model,
            api_key=config.hugging_face_key,
            base_url='https://router.huggingface.co/v1',
            temperature=0
        )
