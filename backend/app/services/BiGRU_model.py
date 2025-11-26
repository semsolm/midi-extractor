"""
드럼 타격 검출 모델 (Drum Onset Detection Model) - 개선 버전
CNN + BiGRU 구조를 사용한 멀티레이블 분류 모델

주요 개선사항:
1. CNN 구조 강화: Conv-Conv-Pool 패턴으로 변경 (특징 추출 능력 향상)
2. GRU hidden size 증가: 256 → 384 (temporal dependency 강화)
3. Residual connection 추가 (선택적)
"""

import torch
import torch.nn as nn


class DrumOnsetDetector(nn.Module):
    """
    드럼 타격 검출 모델 (개선 버전)

    아키텍처:
    1. CNN layers: Conv-Conv-Pool 패턴으로 특징 추출 능력 향상
    2. BiGRU layers: Hidden size 증가로 시간적 문맥 정보 학습 강화
    3. Fully connected layers: 멀티레이블 분류 (3개 드럼 타입)
    """

    def __init__(
            self,
            n_mels=128,
            n_classes=3,
            cnn_channels=[32, 64, 128],
            gru_hidden=384,  # 256 → 384로 증가
            gru_layers=2,
            dropout=0.3
    ):
        super(DrumOnsetDetector, self).__init__()

        self.n_mels = n_mels
        self.n_classes = n_classes

        # ============================================
        # 1. CNN 레이어: Conv-Conv-Pool 패턴
        # ============================================
        self.cnn = nn.Sequential(
            # Block 1: Conv-Conv-Pool
            nn.Conv2d(1, cnn_channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 주파수축만 pooling
            nn.Dropout2d(dropout * 0.5),
            # Output: (batch, 32, seq_len, n_mels//2=64)

            # Block 2: Conv-Conv-Pool
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout * 0.5),
            # Output: (batch, 64, seq_len, n_mels//4=32)

            # Block 3: Conv-Conv-Pool
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[2], cnn_channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout * 0.5),
            # Output: (batch, 128, seq_len, n_mels//8=16)
        )

        self.freq_bins_after_cnn = n_mels // (2 ** 3)  # 128 // 8 = 16
        self.cnn_output_dim = cnn_channels[2] * self.freq_bins_after_cnn  # 128 * 16 = 2048

        # ============================================
        # 2. BiGRU 레이어: Hidden size 증가
        # ============================================
        self.gru = nn.GRU(
            input_size=self.cnn_output_dim,
            hidden_size=gru_hidden,  # 384로 증가
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.gru_output_dim = gru_hidden * 2  # 384 * 2 = 768

        # ============================================
        # 3. Fully Connected 레이어
        # ============================================
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

        # ============================================
        # 4. 가중치 초기화
        # ============================================
        self._init_weights()

    def _init_weights(self):
        """Xavier/He 초기화로 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 멜스펙트로그램 (batch_size, seq_len, n_mels)

        Returns:
            logits: 각 드럼 타입별 로짓 (batch_size, seq_len, n_classes)
        """
        batch_size, seq_len, n_mels = x.shape

        # CNN을 위한 shape 변환
        x = x.unsqueeze(1)  # (batch, 1, seq_len, n_mels)

        # CNN으로 특징 추출
        x = self.cnn(x)  # (batch, 128, seq_len, 16)

        # GRU를 위한 shape 변환
        batch_size, channels, seq_len, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, seq_len, channels, freq_bins)
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 2048)

        # BiGRU로 시간적 문맥 학습
        x, _ = self.gru(x)  # (batch, seq_len, 768)

        # Fully Connected로 분류
        logits = self.fc(x)  # (batch, seq_len, 3)

        return logits


def get_model(device='cuda', **kwargs):
    """모델 생성 및 디바이스 할당 헬퍼 함수"""
    model = DrumOnsetDetector(**kwargs)
    model = model.to(device)
    return model

# 로컬에서 테스트 하기 위한 코드 : 주석 처리
# if __name__ == "__main__":
#     """모델 구조 테스트"""
#     batch_size = 4
#     seq_len = 500
#     n_mels = 128

#     x = torch.randn(batch_size, seq_len, n_mels)

#     model = DrumOnsetDetector(
#         n_mels=128,
#         n_classes=3,
#         cnn_channels=[32, 64, 128],
#         gru_hidden=384,  # 증가됨
#         gru_layers=2,
#         dropout=0.3
#     )

#     with torch.no_grad():
#         logits = model(x)

#     print("=" * 60)
#     print("드럼 타격 검출 모델 구조 테스트 (개선 버전)")
#     print("=" * 60)
#     print(f"입력 shape: {x.shape}")
#     print(f"출력 shape (logits): {logits.shape}")
#     print(f"\n모델 총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
#     print(f"학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#     print("=" * 60)

#     probs = torch.sigmoid(logits)
#     print(f"\n확률 범위 확인 (sigmoid 후):")
#     print(f"  최소값: {probs.min().item():.4f}")
#     print(f"  최대값: {probs.max().item():.4f}")
#     print(f"  평균값: {probs.mean().item():.4f}")