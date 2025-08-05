import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import models

class BrainMRIMultiStageClassifier(nn.Module):
    """
    다단계 Brain MRI 분류 모델
    1단계: Tumor 여부 판별
    2단계: 평면 분류 (axial, sagittal, coronal)
    3단계: 종양 종류 분류 (glioma, meningioma, pituitary)
    """
    
    def __init__(self, model_paths, device=None):
        super(BrainMRIMultiStageClassifier, self).__init__()
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device
        
        # 모델 경로 저장
        self.model_paths = model_paths
        
        # 전처리 파이프라인 설정
        self.transform_rgb = transforms.Compose([
            transforms.Resize(224),        
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB")), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_gray = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("L")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 클래스 이름 정의
        self.tumor_classes = ["정상", "종양"]
        self.plane_classes = ["axial", "sagittal", "coronal"]
        self.tumor_type_classes = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
        
        # 최종 4개 클래스 (평면 정보 제외)
        self.all_classes = ["정상", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
        
        # 모델들 로드
        self._load_models()
        
    def _load_model(self, path, model_name, num_classes):
        """개별 모델 로드 함수"""
        model = models.BrainMRIClassifier(num_classes=num_classes, model_name=model_name)
        state_dict = torch.load(path, map_location=self.device)
        
        if isinstance(state_dict, dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('backbone.') and not key.startswith('classifier.'):
                    new_key = 'backbone.' + key
                    new_state_dict[new_key] = value
                elif model_name == 'resnet18' and key.startswith('fc.'):
                    new_key = 'classifier.' + key[3:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
                    
            if model_name == 'resnet18' and 'backbone.conv1.weight' in new_state_dict:
                conv1_weight = new_state_dict['backbone.conv1.weight']
                if conv1_weight.shape[1] == 1:
                    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                elif conv1_weight.shape[1] == 3:
                    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model = state_dict
            
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_models(self):
        """모든 모델 로드"""
        print("모델들을 로드하는 중...")
        
        self.tumor_classifier = self._load_model(
            self.model_paths['tumor'], 'efficientnet', 1
        )
        self.plane_classifier = self._load_model(
            self.model_paths['plane'], 'efficientnet', 3
        )
        self.sagittal_classifier = self._load_model(
            self.model_paths['sagittal'], 'efficientnet', 3
        )
        self.coronal_classifier = self._load_model(
            self.model_paths['coronal'], 'efficientnet', 3
        )
        self.axial_classifier = self._load_model(
            self.model_paths['axial'], 'efficientnet', 3
        )
        
        print("모든 모델 로드 완료!")
    
    def forward(self, x):
        """
        순전파 - 다단계 분류 수행
        
        Args:
            x: 전처리된 이미지 텐서 (batch_size, 3, 224, 224)
            
        Returns:
            dict: 각 단계별 예측 결과와 최종 클래스 인덱스
        """
        batch_size = x.size(0)
        results = {
            'tumor_probs': [],
            'plane_preds': [],
            'tumor_type_preds': [],
            'final_class_indices': [],
            'final_class_names': []
        }
        
        with torch.no_grad():
            # 1단계: Tumor 여부 판별
            tumor_outputs = self.tumor_classifier(x)
            tumor_probs = torch.sigmoid(tumor_outputs).squeeze()
            tumor_preds = (tumor_probs > 0.5).long()
            
            for i in range(batch_size):
                tumor_prob = tumor_probs[i].item() if batch_size > 1 else tumor_probs.item()
                tumor_pred = tumor_preds[i].item() if batch_size > 1 else tumor_preds.item()
                
                results['tumor_probs'].append(tumor_prob)
                
                if tumor_pred == 1:  # No Tumor (정상)
                    results['plane_preds'].append(-1)
                    results['tumor_type_preds'].append(-1)
                    results['final_class_indices'].append(0)  # "정상" 클래스
                    results['final_class_names'].append("정상")
                else:
                    # 2단계: 평면 분류
                    single_image = x[i:i+1]
                    plane_output = self.plane_classifier(single_image)
                    plane_pred = torch.argmax(plane_output, dim=1).item()
                    results['plane_preds'].append(plane_pred)
                    
                    # 3단계: 종양 종류 분류
                    if plane_pred == 0:  # axial
                        final_model = self.axial_classifier
                    elif plane_pred == 1:  # sagittal
                        final_model = self.sagittal_classifier
                    else:  # coronal
                        final_model = self.coronal_classifier
                    
                    tumor_type_output = final_model(single_image)
                    tumor_type_pred = torch.argmax(tumor_type_output, dim=1).item()
                    results['tumor_type_preds'].append(tumor_type_pred)
                    
                    # 최종 클래스 결정 (평면 정보 제외, 종양 종류만)
                    final_class_name = self.tumor_type_classes[tumor_type_pred]
                    final_class_idx = tumor_type_pred + 1  # 0: 정상, 1-3: 종양 종류들
                    
                    results['final_class_indices'].append(final_class_idx)
                    results['final_class_names'].append(final_class_name)
        
        # 텐서로 변환
        results['final_class_indices'] = torch.tensor(results['final_class_indices'])
        
        return results
    
    def predict_single_image(self, image_path):
        """
        단일 이미지 예측 (기존 함수와 호환)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            dict: 예측 결과
        """
        image = Image.open(image_path)
        image_tensor = self.transform_rgb(image).unsqueeze(0).to(self.device)
        
        result = self.forward(image_tensor)
        
        return {
            "result": result['final_class_names'][0],
            "tumor_probability": result['tumor_probs'][0],
            "plane_prediction": result['plane_preds'][0],  # 내부적으로만 사용
            "tumor_type_prediction": result['tumor_type_preds'][0],
            "final_class_index": result['final_class_indices'][0].item()
        }
    
    def get_class_names(self):
        """평가용 클래스 이름 반환"""
        return self.all_classes
    
    def get_num_classes(self):
        """클래스 수 반환"""
        return len(self.all_classes)

# 모델 초기화 함수
def create_brain_mri_classifier(base_path="/Users/skku_aws17/Desktop/projects/brain_pj/app/models/"):
    """
    Brain MRI 분류 모델 생성
    
    Args:
        base_path: 모델 파일들이 있는 기본 경로
        
    Returns:
        BrainMRIMultiStageClassifier: 초기화된 모델
    """
    model_paths = {
        'tumor': f"{base_path}2class.pth",
        'plane': f"{base_path}plane_classifier.pth",
        'sagittal': f"{base_path}sagittal_classifier.pth",
        'coronal': f"{base_path}coronal_classifier.pth",
        'axial': f"{base_path}axial_classifier.pth"
    }
    
    return BrainMRIMultiStageClassifier(model_paths)

# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = create_brain_mri_classifier()
    
    # 단일 이미지 예측 (기존 방식과 호환)
    result = model.predict_single_image("/Users/skku_aws17/Desktop/projects/brain_pj/app/test_images/ax_14.png")
    print("예측 결과:", result)
    
    # 클래스 정보 확인
    print("전체 클래스:", model.get_class_names())
    print("클래스 수:", model.get_num_classes())
    print("클래스 매핑:")
    for i, class_name in enumerate(model.get_class_names()):
        print(f"  {i}: {class_name}")