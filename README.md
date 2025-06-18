# DU Chatbot Project

## 프로젝트 개요 (Project Overview)
동서울대학교 학생과 직원들이 학교 정보를 쉽고 빠르게 접근할 수 있도록 전용 안내 챗봇을 개발한 프로젝트입니다. 기존의 GPT 기반 LLM에서 발생하는 환각(Hallucination), 실시간 데이터 반영 어려움, 높은 파라미터 수로 인한 부담을 해소하기 위해 PEFT(LoRA)와 RAG를 결합한 하이브리드 모델을 구축했습니다.

모바일 환경에서의 접근성을 극대화하기 위해 음성 인식(STT) 및 음성 합성(TTS) 기능을 통합하여 안드로이드 및 iOS 환경에서도 동작하도록 설계했습니다.

## 주요 기능
- **학교 정보 검색 및 즉각적인 응답**: 학과 및 부서 정보 등을 챗봇을 통해 실시간으로 검색하고 접근할 수 있습니다.
- **모바일 환경 최적화**: 음성 인터페이스(STT 및 TTS)를 탑재하여 모바일 사용자의 편의성을 높였습니다.
- **보안과 성능 최적화**: 내부 환경에서 독립적으로 운용 가능하도록 모델 경량화 및 파라미터 효율적 조정을 수행했습니다.

## 프로젝트 로드맵 (Project Roadmap)
![스크린샷 2024-10-22 195158](https://github.com/user-attachments/assets/9db9a29b-bbd3-489e-a319-7ef2bafd4c6d)
![image](https://github.com/user-attachments/assets/f07af4b2-81f6-4700-bb9f-63705a682ae3)


## 기술적 접근 방법 (Technical Approach)

### PEFT (Parameter Efficient Fine-Tuning)
- 기존 LLM 전체를 재학습하는 대신 소규모 행렬(LoRA)을 추가하여 파라미터 수를 약 3.35% 수준으로 효율적으로 줄였습니다.
- ![image](https://github.com/user-attachments/assets/2c7c71a6-e4ee-4c84-a7e0-e92b69d65e17)


### RAG (Retrieval-Augmented Generation)
- 외부 데이터베이스로부터 문서를 검색하고, 이를 기반으로 보다 정확하고 신뢰할 수 있는 답변을 생성합니다.
- 데이터는 동서울대학교 웹사이트에서 크롤링한 Q&A 데이터(총 720개)로 구성되었습니다.
- ![image](https://github.com/user-attachments/assets/301404df-9f9c-4e73-9fc8-9b03fb4bc663)


## 성능 평가 (Evaluation)
모델의 성능 평가를 위해 다음과 같은 N-gram 기반 메트릭을 사용했습니다:
- **BLEU**: 응답 품질을 참조 문장과의 N-gram 겹침 비율로 평가
- **METEOR**: 단어 정확도 및 구조적 일관성 평가
- **ROUGE**: 참조 문장의 정보가 얼마나 잘 포착되었는지 평가

오버피팅을 방지하기 위해 적절한 epoch 수를 최적화하였습니다.
![image](https://github.com/user-attachments/assets/caca49b8-2696-4c49-84f1-3867488dc450)

실험 결과, PEFT와 RAG의 혼합형 모델이 최고 성능을 나타냈습니다.
![image](https://github.com/user-attachments/assets/374ab6ef-a38f-44c1-b6bd-c747c086a4b0)


## 모바일 애플리케이션 (Mobile Application)
안드로이드와 iOS 환경에서 모두 사용 가능하며, 다음과 같은 도구와 언어를 사용하여 개발하였습니다:
- **Android**: Kotlin, Android Studio
- **iOS**: Swift, Xcode
- ![image](https://github.com/user-attachments/assets/fb646b48-eb67-4edb-a2c2-38a54c88ed8d)


음성 기반 인터페이스(STT 및 TTS)를 제공하여 사용자의 접근성을 높였습니다.

## 직면한 문제들과 해결방식 (Challenges & Solutions)
1. **GPU 제약**: 
   - 문제: 한정된 4060Ti GPU로는 모델 응답 시간이 오래 걸려 상용화에 어려움이 있었습니다.
   - 해결책: 모델 구동용 PC와 웹 및 Tokenizer 실행용 PC를 분리하여 GPU 부하를 분산하였습니다. 이를 통해 Inference 시간을 60초에서 8초로 대폭 단축할 수 있었습니다.

2. **환경 설정 (Environment Setting)**:
   - 문제: Windows 환경에서 4060Ti에 맞는 CUDA 설정과 Streamlit 버전 조정에 예상보다 많은 시간이 소요되었습니다.
   - 해결책: 여러 차례의 테스트를 거쳐 최적의 설정값을 도출하였으며, 아래에 최종 사용한 환경 설정을 기록하였습니다.

## 사용 환경 (Environment)
GPU: V100 / 4060Ti
Cuda capavility: 8.9
Cuda SDK: 11.8
pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# 사용법 (How to Use)

1. **필수 패키지 설치**  
   `requirements.txt` 파일을 기반으로 필요한 패키지를 설치합니다.  
   ```bash
   pip install -r requirements.txt
   ```

2. **모델 다운로드**  
   Hugging Face CLI를 사용하여 모델 파일을 다운로드합니다.  
   ```bash
   huggingface-cli download \
     byung-jun/DUChatbot-CapstoneDesign-GGUF \
     llama3.2_3B_q4.gguf \
     --local-dir <다운로드할_폴더_경로> \
     --local-dir-use-symlinks False
   ```

3. **Ollama 모델 생성**  
   다운로드한 모델 파일을 이용해 Ollama에서 모델을 생성합니다.  
   ```bash
   ollama create "<모델_이름>" -f <모델_파일>
   ```

4. **모델 확인**  
   아래 명령어를 실행했을 때, 생성한 모델 이름이 출력되면 정상적으로 생성된 것입니다.  
   ```bash
   ollama list
   ```

5. **모델 테스트**  
   모델이 정상 작동하는지 확인합니다.  
   ```bash
   ollama run "<모델_이름>"
   ```

6. **앱 설정**  
   `app/llm.py`와 `app/chat.py` 파일에서 생성한 모델 이름을 입력합니다.

7. **서버 실행**  
   서버를 실행합니다.  
   ```bash
   python app/server.py
   ```

8. **포트 포워딩 설정**  
   `ngrok`을 사용해 포트 포워딩을 설정합니다.  
   ```bash
   ngrok http 8000
   ```

9. **원격 주소 설정**  
   생성된 `ngrok` 주소를 `example/main.py`의 `RemoteRunnable` 함수에 추가합니다.

10. **앱 실행**  
    Streamlit을 통해 앱을 실행합니다.  
    ```bash
    streamlit run example/main.py
    ```

---

위 단계를 순차적으로 진행하면 모델과 앱이 정상적으로 작동합니다! 🚀


## 느낀점 (Reflection)
본 프로젝트는 인공지능을 학습하면서 처음으로 수행한 프로젝트로, 프로젝트의 흐름을 이해하는 데 다소 시간이 걸렸으나 팀원들과의 협력으로 만족스러운 결과를 달성할 수 있었습니다. 이번 경험을 통해 향후 AI 발전 방향과 기술 접목에 대한 아이디어가 보다 명확해졌습니다.

