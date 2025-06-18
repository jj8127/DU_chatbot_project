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
GPU: V100
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

