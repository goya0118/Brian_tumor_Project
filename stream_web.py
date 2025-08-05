import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import os
from datetime import datetime
import pandas as pd

try:
    from brain_mri_classifier import BrainMRIMultiStageClassifier, create_brain_mri_classifier
    st.success("✅ brain_mri_classifier.py 파일 import 성공!")
except ImportError as e:
    st.error(f"❌ brain_mri_classifier.py import 실패: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링 (기존과 동일)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 🔧 모델 로드 함수 수정 - ./models/ 경로 직접 사용
@st.cache_resource
def load_model():
    """EC2/로컬 환경 모두 지원하는 모델 로더"""
    try:
        with st.spinner("🤖 AI 모델 로딩 중..."):
            # 현재 디렉토리 확인
            current_dir = os.getcwd()
            st.write(f"📁 현재 디렉토리: {current_dir}")
            
            # 로컬 models 폴더 생성
            models_path = "./models/"
            os.makedirs(models_path, exist_ok=True)
            
            # 필수 모델 파일 목록
            required_files = [
                "2class.pth", 
                "plane_classifier.pth", 
                "sagittal_classifier.pth", 
                "coronal_classifier.pth", 
                "axial_classifier.pth"
            ]
            
            # 로컬에 모든 파일이 있는지 확인
            missing_files = []
            for file_name in required_files:
                local_path = os.path.join(models_path, file_name)
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    missing_files.append(file_name)
                else:
                    file_size = os.path.getsize(local_path) / (1024 * 1024)
                    st.write(f"✅ 로컬 파일: {file_name} ({file_size:.1f}MB)")
            
            # 누락된 파일이 있으면 S3에서 다운로드 시도
            if missing_files:
                st.write(f"📥 누락된 파일 {len(missing_files)}개를 S3에서 다운로드 시도...")
                
                success = download_models_from_s3(missing_files, models_path)
                if not success:
                    st.error("❌ S3에서 모델 다운로드 실패. 로컬 파일을 확인하거나 S3 설정을 점검하세요.")
                    return None, "모델 파일 다운로드 실패"
            
            # 최종 파일 존재 확인
            final_missing = []
            for file_name in required_files:
                local_path = os.path.join(models_path, file_name)
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    final_missing.append(file_name)
            
            if final_missing:
                st.error(f"❌ 모델 파일 누락: {final_missing}")
                st.info("💡 해결 방법: 1) 로컬에 파일 복사 2) config.py에 S3 설정 3) AWS 자격증명 설정")
                return None, f"모델 파일 누락: {final_missing}"
            
            st.write("✅ 모든 모델 파일 준비 완료")
            
            # 🎯 기존 파일의 함수 직접 사용
            st.write("🚀 create_brain_mri_classifier 함수 호출 중...")
            model = create_brain_mri_classifier(models_path)
            
            st.write(f"✅ 모델 로드 완료!")
            st.write(f"📊 클래스 수: {model.get_num_classes()}")
            st.write(f"📋 클래스 이름: {model.get_class_names()}")
            st.write(f"🖥️ 디바이스: {model.device}")
            
        return model, None
        
    except Exception as e:
        error_msg = f"❌ 모델 로드 실패: {str(e)}"
        st.error(error_msg)
        import traceback
        st.code(traceback.format_exc())
        return None, error_msg

def download_models_from_s3(missing_files, models_path):
    """S3에서 모델 파일 다운로드"""
    try:
        # 방법 1: config.py에서 설정 가져오기
        try:
            from config import S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            bucket_name = S3_BUCKET_NAME
            st.write(f"✅ config.py에서 S3 설정 로드")
            
        except ImportError:
            # 방법 2: 환경 변수에서 가져오기
            bucket_name = os.environ.get('S3_BUCKET_NAME')
            if not bucket_name:
                st.warning("⚠️ S3 설정이 없습니다. 로컬 파일을 사용하거나 config.py/환경변수를 설정하세요.")
                return False
            
            s3_client = boto3.client('s3')  # AWS CLI 설정 또는 IAM Role 사용
            st.write(f"✅ 환경변수/AWS CLI에서 S3 설정 로드")
        
        # S3에서 파일 다운로드
        for file_name in missing_files:
            try:
                # S3 키 경로 (실제 S3 구조에 맞게 수정)
                s3_key = f"models/{file_name}"
                local_path = os.path.join(models_path, file_name)
                
                st.write(f"⬇️ 다운로드: {s3_key}")
                
                # 다운로드 실행
                s3_client.download_file(bucket_name, s3_key, local_path)
                
                # 파일 크기 확인
                file_size = os.path.getsize(local_path) / (1024 * 1024)
                st.write(f"✅ 완료: {file_name} ({file_size:.1f}MB)")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchKey':
                    st.error(f"❌ S3에 파일이 없음: {s3_key}")
                elif error_code == 'NoSuchBucket':
                    st.error(f"❌ S3 버킷이 없음: {bucket_name}")
                else:
                    st.error(f"❌ S3 오류: {str(e)}")
                return False
                
            except Exception as e:
                st.error(f"❌ 다운로드 실패: {file_name} - {str(e)}")
                return False
        
        return True
        
    except NoCredentialsError:
        st.error("❌ AWS 자격증명이 설정되지 않았습니다.")
        st.info("💡 해결 방법: 1) aws configure 실행 2) config.py 설정 3) 환경변수 설정")
        return False
        
    except Exception as e:
        st.error(f"❌ S3 연결 실패: {str(e)}")
        return False

def get_s3_info():
    """S3 설정 정보 표시 (디버깅용)"""
    st.subheader("🔍 S3 설정 정보")
    
    # config.py 확인
    try:
        from config import S3_BUCKET_NAME
        st.write(f"✅ config.py 버킷: {S3_BUCKET_NAME}")
    except ImportError:
        st.write("❌ config.py 없음")
    
    # 환경변수 확인
    env_bucket = os.environ.get('S3_BUCKET_NAME')
    if env_bucket:
        st.write(f"✅ 환경변수 버킷: {env_bucket}")
    else:
        st.write("❌ 환경변수 S3_BUCKET_NAME 없음")
    
    # AWS 자격증명 확인
    try:
        boto3.client('s3')
        st.write("✅ AWS 자격증명 설정됨")
    except:
        st.write("❌ AWS 자격증명 없음")

# 나머지 함수들은 기존과 동일
def get_prediction_color(prediction_result):
    """예측 결과에 따른 색상 반환"""
    if prediction_result == "정상":
        return "#28a745"  # 초록색
    elif "glioma" in prediction_result:
        return "#dc3545"  # 빨간색
    elif "meningioma" in prediction_result:
        return "#fd7e14"  # 주황색
    elif "pituitary" in prediction_result:
        return "#6f42c1"  # 보라색
    else:
        return "#6c757d"  # 회색

def create_confidence_chart(probabilities, class_names):
    """신뢰도 차트 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#28a745', '#dc3545', '#fd7e14', '#6f42c1']
    bars = ax.bar(class_names, probabilities, color=colors, alpha=0.7)
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Class Prediction Confidence')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

# stream_web.py에서 이 함수로 교체하세요

def process_image_prediction(model, uploaded_file):
    """이미지 예측 처리 - 기존 모델의 forward 메서드 직접 사용"""
    try:
        # 이미지 로드
        image = Image.open(uploaded_file)
        
        with st.spinner("🔍 AI가 이미지를 분석 중입니다..."):
            start_time = time.time()
            
            # PIL Image를 RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 모델의 전처리 사용
            image_tensor = model.transform_rgb(image).unsqueeze(0).to(model.device)
            
            # 모델의 forward 메서드 직접 호출
            with torch.no_grad():
                results = model.forward(image_tensor)
            
            # 결과 정리 (기존 predict_single_image_from_pil과 동일한 형태로)
            result = {
                "result": results['final_class_names'][0],
                "tumor_probability": results['tumor_probs'][0],
                "plane_prediction": results['plane_preds'][0],
                "tumor_type_prediction": results['tumor_type_preds'][0],
                "final_class_index": results['final_class_indices'][0].item(),
                "confidence": results['tumor_probs'][0] if results['final_class_indices'][0] == 0 else (1 - results['tumor_probs'][0])
            }
            
            # 평면 이름 추가
            if results['plane_preds'][0] != -1:
                plane_names = ["수평면", "시상면", "관상면"]
                result["plane_name"] = plane_names[results['plane_preds'][0]]
            else:
                result["plane_name"] = "해당없음"
            
            # 단계별 정보 생성
            stage_info = []
            
            # 1단계 정보
            tumor_prob = results['tumor_probs'][0]
            if results['final_class_indices'][0] == 0:  # 정상
                stage_info.append(f"1단계: 정상 감지 (확률: {tumor_prob:.3f})")
            else:  # 종양
                stage_info.append(f"1단계: 종양 감지 (확률: {1-tumor_prob:.3f})")
                
                # 2단계 정보
                plane_pred = results['plane_preds'][0]
                plane_names = ["수평면(Axial)", "시상면(Sagittal)", "관상면(Coronal)"]
                stage_info.append(f"2단계: {plane_names[plane_pred]} 감지")
                
                # 3단계 정보
                tumor_type = results['final_class_names'][0]
                stage_info.append(f"3단계: {tumor_type} 분류 ({plane_names[plane_pred]} 모델 사용)")
            
            result["stage_info"] = stage_info
            
            end_time = time.time()
            
        processing_time = end_time - start_time
        
        return result, processing_time, None
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        st.error("상세 오류 정보:")
        st.code(error_detail)
        return None, 0, f"예측 중 오류: {str(e)}"

def display_prediction_results(result, processing_time, original_image):
    """예측 결과 표시 - 다단계 정보 포함"""
    prediction = result['result']
    tumor_prob = result['tumor_probability']
    
    # 메인 예측 결과
    color = get_prediction_color(prediction)
    
    st.markdown(f"""
    <div class="prediction-box">
        <h2 style="color: {color}; margin: 0;">🎯 최종 예측: {prediction}</h2>
        <p style="margin: 0.5rem 0;"><strong>처리 시간:</strong> {processing_time:.2f}초</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 상세 정보를 컬럼으로 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 다단계 분석 결과")
        
        # 1단계: 종양 확률
        if prediction == "정상":
            st.metric("1단계 - 정상 확률", f"{tumor_prob:.1%}", "✅ 정상으로 분류")
        else:
            st.metric("1단계 - 종양 확률", f"{(1-tumor_prob):.1%}", "⚠️ 종양 감지")
        
        # 2단계: 평면 정보
        if 'plane_name' in result and result['plane_name'] != "해당없음":
            st.info(f"🧭 2단계 - 감지된 평면: {result['plane_name']}")
        
        # 3단계: 종양 타입 (종양인 경우만)
        if prediction != "정상":
            st.info(f"🔬 3단계 - 종양 종류: {prediction}")
        
        # 단계별 상세 정보
        if 'stage_info' in result:
            st.subheader("🔍 단계별 처리 과정")
            for i, stage in enumerate(result['stage_info'], 1):
                st.write(f"**{stage}**")
        
        # 신뢰도 지표
        confidence = result.get('confidence', 0)
        if confidence > 0.8:
            st.success(f"🎯 높은 신뢰도: {confidence:.1%}")
        elif confidence > 0.6:
            st.warning(f"⚠️ 중간 신뢰도: {confidence:.1%}")
        else:
            st.error(f"❗ 낮은 신뢰도: {confidence:.1%}")
    
    with col2:
        st.subheader("🖼️ 원본 이미지")
        st.image(original_image, caption="업로드된 MRI 이미지", use_column_width=True)

def main():
    """메인 앱"""
    
    # 헤더
    st.markdown('<h1 class="main-header">🧠 Brain MRI AI Classifier</h1>', unsafe_allow_html=True)
    
    # 모델 로드 (세션 상태 사용해서 한 번만 로드)
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.error = load_model()
    
    model = st.session_state.model
    error = st.session_state.error
    
    if model is None:
        st.markdown(f'<div class="error-box"><h3>❌ 모델 로드 실패</h3><p>{error}</p></div>', 
                   unsafe_allow_html=True)
        st.stop()
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        
        # 모델 정보
        st.subheader("📋 모델 정보")
        device_info = str(model.device) if hasattr(model, 'device') else "알 수 없음"
        st.info(f"""
        **모델 타입:** 다단계 분류 모델  
        **클래스:** 4개 (정상, glioma, meningioma, pituitary)  
        **입력 크기:** 224x224 픽셀  
        **디바이스:** {device_info}
        **처리 단계:** 3단계 (종양감지 → 평면분류 → 종양분류)
        """)
        
        # 사용법 안내
        st.subheader("📖 사용법")
        st.markdown("""
        1. **이미지 업로드**: MRI 이미지 파일을 선택하세요
        2. **분석 시작**: '분석 시작' 버튼을 클릭하세요  
        3. **결과 확인**: AI의 3단계 진단 결과를 확인하세요
        
        **지원 형식:** PNG, JPG, JPEG
        """)
        
        # 주의사항
        st.subheader("⚠️ 주의사항")
        st.warning("""
        이 AI 모델은 **연구/교육 목적**으로만 사용하세요.  
        실제 의료 진단은 반드시 **전문의와 상담**하시기 바랍니다.
        """)
    
    # 메인 컨텐츠
    st.header("📤 MRI 이미지 업로드")
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "MRI 이미지를 선택하세요",
        type=['png', 'jpg', 'jpeg'],
        help="PNG, JPG, JPEG 형식의 MRI 이미지를 업로드하세요."
    )
    
    if uploaded_file is not None:
        # 이미지 미리보기
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_column_width=True)
        
        # 분석 버튼
        if st.button("🔍 AI 분석 시작", type="primary", use_container_width=True):
            
            # 예측 수행
            result, processing_time, error = process_image_prediction(model, uploaded_file)
            
            if error:
                st.markdown(f'<div class="error-box"><h3>❌ 분석 실패</h3><p>{error}</p></div>', 
                           unsafe_allow_html=True)
            else:
                # 결과 표시
                display_prediction_results(result, processing_time, image)
                
                # 결과 다운로드
                st.subheader("💾 결과 저장")
                
                # 결과를 텍스트로 변환
                result_text = f"""
Brain MRI AI 분석 결과 (다단계 모델)
=====================================
분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
파일명: {uploaded_file.name}
처리 시간: {processing_time:.2f}초

🎯 최종 예측: {result['result']}
📊 종양 확률: {result['tumor_probability']:.1%}
🧭 감지된 평면: {result.get('plane_name', 'N/A')}
🔍 신뢰도: {result.get('confidence', 0):.1%}

📋 단계별 처리 과정:
"""
                if 'stage_info' in result:
                    for stage in result['stage_info']:
                        result_text += f"- {stage}\n"
                
                result_text += f"""
⚠️ 주의: 이 결과는 AI 모델의 예측이며, 실제 의료 진단을 대체할 수 없습니다.
반드시 전문의와 상담하시기 바랍니다.
"""
                
                st.download_button(
                    label="📄 결과 텍스트 다운로드",
                    data=result_text,
                    file_name=f"mri_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        # 가이드 표시
        st.markdown("""
        <div class="prediction-box">
            <h3>🚀 시작하기</h3>
            <p>위의 파일 업로더를 사용하여 MRI 이미지를 업로드하세요.</p>
            <p>AI가 3단계 과정을 통해 뇌 종양을 분석하고 분류해드립니다.</p>
            <ul>
                <li><strong>1단계:</strong> 정상/종양 여부 판별</li>
                <li><strong>2단계:</strong> MRI 평면 분류 (수평면/시상면/관상면)</li>
                <li><strong>3단계:</strong> 종양 종류 분류 (glioma/meningioma/pituitary)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧠 Brain MRI AI Classifier | 다단계 분류 모델 | Powered by PyTorch & Streamlit</p>
        <p>⚠️ 교육/연구 목적으로만 사용하세요. 실제 의료 진단은 전문의와 상담하시기 바랍니다.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
