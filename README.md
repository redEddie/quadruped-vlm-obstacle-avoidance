# Quadruped VLM Obstacle Avoidance

본 프로젝트는 VLM(Vision-Language Model)을 활용해 사족보행 로봇이 장애물을 인식하고 회피하도록 학습/테스트하는 파이프라인을 제공합니다.

<!-- TOC -->
- [프로젝트 개요](#프로젝트-개요)
- [논문 및 참조](#논문-및-참조)
- [설치 및 실행 방법](#설치-및-실행-방법)
  - [필수 요구사항](#필수-요구사항)
  - [코드 다운로드](#코드-다운로드)
  - [환경 설정](#환경-설정)
  - [실행 예제](#실행-예제)
- [데모 및 결과](#데모-및-결과)
- [구성 및 디렉터리 구조](#구성-및-디렉터리-구조)
- [연락처](#연락처)
<!-- /TOC -->


# 프로젝트 개요
사족보행 로봇 GO2가 GPT-4.1을 이용해 자연어로 주어진 초기 명령어의 목표를 달성하기 위해 툴을 생성하고, 환경 이미지를 분석하고 실제 제어 명령으로 변환하여 시물레이션 환경에서 장애물 회피 동작을 수행한다.

# 논문 및 참조
**논문 제목**: 시각-언어 모델을 활용한 사족보행 로봇의 장애물 회피(Obstacle Avoidance for Quadrupedal Robots Using Vision-Language Models)  
**저자**: 서명준, 전찬욱, 이상문

# 필수 요구사항
IsaacLab

# 환경 설정
 - API키 설정  
프로젝트 최상위 디렉터리(예: README.md가 있는 곳)에 .env 파일을 만듭니다.  
`.env` 파일 예시

```
# .env 파일 예시
# 실제 값은 본인 환경에 맞게 수정하세요

GOOGLE_CSE_ID=oooo
GOOGLE_API_KEY=oooo
OPENAI_API_KEY=oooo

DATABASE_URL=postgresql://username:password@hostname:port/dbname
SECRET_KEY=your-secret-key-here
```

# 실행 예제
환경 변수 설정이 완료되었다면, 바로 `main.py`를 실행하면 됩니다.
```
python main.py
```

# 데모 및 결과
![output](https://github.com/user-attachments/assets/de52a860-8b92-4864-a9fc-9940c4b678de)


# 연락처
eddie3618@knu.ac.kr
