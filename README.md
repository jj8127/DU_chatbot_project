# DU Chatbot Project

## 프로젝트 개요 (Project Overview)
동서울대학교 학생과 직원들이 학교 정보를 손쉽게 접근하고 검색할 수 있도록 돕는 전용 챗봇을 개발한 프로젝트입니다. 현재 다양한 Cloud 기반의 LLM이 존재하지만, 보안이 중요한 기관(기업, 정부 등)에서는 사용이 어려운 상황입니다. 이러한 문제를 해결하기 위해, 내부 환경에서 구동 가능한 LLM을 구축하여 학교 정보 접근성의 개선을 목표로 하였습니다.

## 주요 기능
- **학교 정보 검색 및 처리**: 학과, 부서와 관련된 정보를 직접 검색하거나 문의하지 않고도, 챗봇을 통해 즉시 필요한 정보를 제공받을 수 있습니다.
- **보안성 유지**: 내부 환경에서 안전하게 운용할 수 있도록 경량화와 성능 유지에 중점을 두어 개발하였습니다.

## 프로젝트 로드맵 (Project Roadmap)
![스크린샷 2024-10-22 195158](https://github.com/user-attachments/assets/9db9a29b-bbd3-489e-a319-7ef2bafd4c6d)

![1730364025513-14a09d46-e92f-4e2c-a941-02df5b6c8e22_1](https://github.com/user-attachments/assets/746edd9d-6b58-48e7-8642-77128679d929)

1. **모델 경량화**: Ollama 모델을 활용하여 성능을 최대한 유지하면서도 Quantization을 통해 모델을 경량화하였습니다.
2. **모델 Fine-Tuning**: 학교 정보를 학습할 수 있도록 PEFT(QLoRA) 방법론을 통해 Fine-Tuning을 수행했습니다.
3. **성능 향상**: 파라미터 변경 없이도 성능 향상을 도모할 수 있는 RAG(학습 및 검색) 구조를 사용하여 LLM의 성능을 강화하였습니다.

## 직면한 문제들과 해결방식 (Challenges & Solutions)
1. **GPU 제약**: 
   - 문제: 한정된 4060Ti GPU로는 모델 응답 시간이 오래 걸려 상용화에 어려움이 있었습니다.
   - 해결책: 모델 구동용 PC와 웹 및 Tokenizer 실행용 PC를 분리하여 GPU 부하를 분산하였습니다. 이를 통해 Inference 시간을 60초에서 8초로 대폭 단축할 수 있었습니다.

2. **환경 설정 (Environment Setting)**:
   - 문제: Windows 환경에서 4060Ti에 맞는 CUDA 설정과 Streamlit 버전 조정에 예상보다 많은 시간이 소요되었습니다.
   - 해결책: 여러 차례의 테스트를 거쳐 최적의 설정값을 도출하였으며, 아래에 최종 사용한 환경 설정을 기록하였습니다.

## 사용 환경 (Environment)
GPU: 4060Ti
Cuda capavility: 8.9
Cuda SDK: 11.8
pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
[requirements.txt](https://github.com/user-attachments/files/17605967/requirements.txt)aiohttp==3.9.4
aiosignal==1.3.1
altair==5.3.0
annotated-types==0.6.0
antlr4-python3-runtime==4.9.3
anyio==4.3.0
attrs==23.2.0
backoff==2.2.1
beautifulsoup4==4.12.3
blinker==1.7.0
Brotli @ file:///C:/b/abs_3d36mno480/croot/brotli-split_1714483178642/work
cachetools==5.3.3
certifi==2024.2.2
cffi==1.16.0
chardet==5.2.0
charset-normalizer @ file:///croot/charset-normalizer_1721748349566/work
click==8.1.7
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.2.1
cryptography==42.0.5
cycler==0.12.1
dataclasses-json==0.6.4
dataclasses-json-speakeasy==0.5.11
Deprecated==1.2.14
distro==1.9.0
effdet==0.4.1
emoji==2.11.0
faiss-cpu==1.8.0
fastapi==0.110.1
filelock==3.13.4
filetype==1.2.0
Final2==0.0.1
flatbuffers==24.3.25
fonttools==4.51.0
frozenlist==1.4.1
fsspec==2024.3.1
gitdb==4.0.11
GitPython==3.1.43
gmpy2 @ file:///C:/ci_311/gmpy2_1677743390134/work
greenlet==3.0.3
h11==0.14.0
httpcore==1.0.5
httpx==0.27.0
httpx-sse==0.4.0
huggingface-hub==0.26.0
humanfriendly==10.0
idna @ file:///C:/b/abs_aad84bnnw5/croot/idna_1714398896795/work
iopath==0.1.10
Jinja2==3.1.3
joblib==1.4.0
jsonpatch==1.33
jsonpath-python==1.0.6
jsonpointer==2.4
jsonschema==4.21.1
jsonschema-specifications==2023.12.1
kiwisolver==1.4.5
langchain==0.1.16
langchain-community==0.0.32
langchain-core==0.1.42
langchain-openai==0.1.3
langchain-text-splitters==0.0.1
langchainhub==0.1.15
langdetect==1.0.9
langserve==0.0.51
langsmith==0.1.45
layoutparser==0.3.4
lxml==5.2.1
markdown-it-py==3.0.0
MarkupSafe==2.1.5
marshmallow==3.21.1
matplotlib==3.8.4
mdurl==0.1.2
mkl-service==2.4.0
mkl_fft @ file:///C:/b/abs_f55mv94vyg/croot/mkl_fft_1725370278455/work
mkl_random @ file:///C:/b/abs_21ydbzdu8d/croot/mkl_random_1725370276095/work
mpmath @ file:///C:/b/abs_7833jrbiox/croot/mpmath_1690848321154/work
multidict==6.0.5
mypy-extensions==1.0.0
networkx==3.3
nltk==3.8.1
numpy==1.26.4
omegaconf==2.3.0
onnx==1.16.0
onnxruntime==1.15.1
openai==1.17.0
opencv-python==4.9.0.80
orjson==3.10.0
packaging==23.2
pandas==2.2.2
pdf2image==1.17.0
pdfminer.six==20231228
pdfplumber==0.11.0
pikepdf==8.15.0
pillow==10.3.0
pillow_heif==0.16.0
portalocker==2.8.2
protobuf==4.25.3
pyarrow==15.0.2
pycocotools==2.0.7
pycparser==2.22
pydantic==2.7.0
pydantic_core==2.18.1
pydeck==0.8.0
Pygments==2.17.2
PyMuPDF==1.24.11
pyparsing==3.1.2
pypdf==4.2.0
pypdfium2==4.29.0
pyreadline3==3.4.1
PySocks @ file:///C:/ci_311/pysocks_1676425991111/work
pytesseract==0.3.10
python-dateutil==2.9.0.post0
python-iso639==2024.2.7
python-magic==0.4.27
python-multipart==0.0.9
pytz==2024.1
pywin32==306
PyYAML==6.0.1
rapidfuzz==3.8.1
referencing==0.34.0
regex==2023.12.25
requests==2.31.0
rich==13.7.1
rpds-py==0.18.0
safetensors==0.4.2
scikit-learn==1.5.2
scipy==1.13.0
sentence-transformers==3.2.0
six==1.16.0
smmap==5.0.1
sniffio==1.3.1
soupsieve==2.5
SQLAlchemy==2.0.29
sse-starlette==2.1.0
starlette==0.37.2
streamlit==1.33.0
sympy==1.12
tabulate==0.9.0
tenacity==8.2.3
threadpoolctl==3.5.0
tiktoken==0.6.0
timm==0.9.16
tokenizers==0.20.1
toml==0.10.2
toolz==0.12.1
torch==2.2.2
torchaudio==2.2.2
torchvision==0.17.2
tornado==6.4
tqdm==4.66.2
transformers==4.45.2
types-requests==2.31.0.20240406
typing-inspect==0.9.0
typing_extensions @ file:///C:/b/abs_0as9mdbkfl/croot/typing_extensions_1715268906610/work
tzdata==2024.1
unstructured==0.13.2
unstructured-client==0.18.0
unstructured-inference==0.7.25
unstructured.pytesseract==0.3.12
urllib3==2.2.1
uvicorn==0.29.0
watchdog==4.0.0
win-inet-pton @ file:///C:/ci_311/win_inet_pton_1676425458225/work
wrapt==1.16.0
yarl==1.9.4


## 느낀점 (Reflection)
본 프로젝트는 인공지능을 학습하면서 처음으로 수행한 프로젝트로, 프로젝트의 흐름을 이해하는 데 다소 시간이 걸렸으나 팀원들과의 협력으로 만족스러운 결과를 달성할 수 있었습니다. 이번 경험을 통해 향후 AI 발전 방향과 기술 접목에 대한 아이디어가 보다 명확해졌습니다.

## 참고 자료 (Resources)
프로젝트와 관련된 데모 영상을 통해 DU Chatbot의 작동 방식을 확인할 수 있습니다.
[![데모 영상](https://img.youtube.com/vi/VkcaigvTrug/0.jpg)](https://youtu.be/VkcaigvTrug)

## License
MIT License

Copyright (c) 2024, 테디노트

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of

