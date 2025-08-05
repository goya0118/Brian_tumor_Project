# 🧠 Brain MRI AI Classifier

Streamlit 기반 뇌 MRI 4클래스 분류 시스템

## 주요 기능
- 다단계 분류: 정상/종양 → 평면 분류 → 종양 타입 분류
- 4가지 클래스: 정상, glioma_tumor, meningioma_tumor, pituitary_tumor
- 웹 인터페이스: Streamlit 기반

## 설치 및 실행
```bash
pip install torch torchvision streamlit Pillow numpy matplotlib seaborn scikit-learn
streamlit run stream_web.py

#주의사항

config.py 파일에 AWS 설정 필요
models/ 폴더에 .pth 모델 파일들 필요
교육/연구 목적으로만 사용
