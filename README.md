# 🧠 Brain MRI AI Classifier

Streamlit 기반 뇌 MRI 4클래스 분류 시스템

## 주요 기능
- 다단계 분류: 정상/종양 → 평면 분류 → 종양 타입 분류
- 4가지 클래스: 정상, glioma_tumor, meningioma_tumor, pituitary_tumor
- 웹 인터페이스: Streamlit 기반
- S3 통합: AWS S3에서 데이터 로드 및 모델 저장

## 설치 및 실행

### 1. 의존성 설치
```bash
# requirements.txt 사용 (권장)
pip install -r requirements.txt

# 또는 개별 설치
pip install torch torchvision streamlit Pillow numpy matplotlib seaborn boto3
```

### 2. 모델 파일 준비
`models/` 폴더에 다음 모델 파일들이 필요합니다:
- `2class.pth` - 종양 여부 분류 모델
- `plane_classifier.pth` - 평면 분류 모델
- `sagittal_classifier.pth` - 시상면 종양 분류 모델
- `coronal_classifier.pth` - 관상면 종양 분류 모델
- `axial_classifier.pth` - 수평면 종양 분류 모델

### 3. 설정 파일 수정
`config.py` 파일에서 AWS 설정을 확인하세요:
```python
AWS_ACCESS_KEY_ID = 'your_access_key'
AWS_SECRET_ACCESS_KEY = 'your_secret_key'
AWS_REGION = 'ap-northeast-2'
S3_BUCKET_NAME = 'your_bucket_name'
```

### 4. 애플리케이션 실행
```bash
streamlit run stream_web.py
```

## 프로젝트 구조
```
brain_app/
├── stream_web.py          # Streamlit 웹 애플리케이션
├── brain_mri_classifier.py # 메인 분류 클래스
├── models.py              # 모델 아키텍처 정의
├── config.py              # 설정 파일
├── models/                # 모델 파일들 (.pth)
├── requirements.txt       # 의존성 목록
└── README.md             # 이 파일
```

## 사용법
1. 웹 브라우저에서 `http://localhost:8501` 접속
2. "Choose File" 버튼으로 MRI 이미지 업로드
3. "Analyze" 버튼 클릭하여 분석 시작
4. 결과 확인: 정상/종양 판정 및 종양 유형 분류

## 기술 스택
- **Backend**: PyTorch, TorchVision
- **Frontend**: Streamlit
- **Image Processing**: Pillow, OpenCV
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Cloud**: AWS S3 (boto3)

## 주의사항
- 교육/연구 목적으로만 사용
- 실제 의료 진단에는 사용하지 마세요
- AWS 자격 증명은 안전하게 관리하세요
- Apple Silicon Mac에서는 MPS 가속 지원

## 라이선스
연구 및 교육 목적으로만 사용 가능합니다.
