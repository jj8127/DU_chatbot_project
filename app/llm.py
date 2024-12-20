from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
<<<<<<< HEAD
llm = ChatOllama(model="ollama 모델 이름을 넣으세요", stop=["</s>"])
=======
llm = ChatOllama(model="DUchatbot:latest", stop=["</s>"])
>>>>>>> cac02d4 (add push)
