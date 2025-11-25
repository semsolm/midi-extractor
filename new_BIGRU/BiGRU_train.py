"""
드럼 타격 검출 모델 학습 스크립트 (최종 개선 버전)

주요 개선사항:
1. ✅ ±1 프레임 확장 라벨
2. ✅ SpecAugment 추가
3. ✅ Silent sample augmentation
4. ✅ WeightedFocalBCEWithLogitsLoss 구현
5. ✅ 클래스별 threshold grid search
6. ✅ Best threshold를 checkpoint에 저장
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import json
from datetime import datetime
import gc
import time
from itertools import product

from BiGRU_model import DrumOnsetDetector
from BiGRU_datautilr import DrumDatasetConfig, get_dataloaders


class CompleteTrainConfig:
    """완전한 학습 설정 클래스 (최종 개선 버전)"""

    def __init__(self):
        # 경로 설정
        self.save_dir = "./checkpoints_final"
        self.log_dir = "./logs_final"

        # 데이터 로딩 모드 선택
        self.use_precomputed = True
        self.precomputed_root = "./precomputed_bigru_data_hop256_improved"

        # 학습 하이퍼파라미터
        self.epochs = 50
        self.batch_size = 4
        self.accumulation_steps = 4  # 실질 배치: 16
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.num_workers = 4

        # 최적화 설정
        self.use_mixed_precision = True
        self.use_gradient_checkpointing = False
        self.empty_cache_every_n_batches = 20
        self.max_grad_norm = 1.0

        # 모델 하이퍼파라미터
        self.n_mels = 128
        self.n_classes = 3
        self.cnn_channels = [32, 64, 128]
        self.gru_hidden = 384
        self.gru_layers = 2
        self.dropout = 0.3

        # ============================================
        # 손실함수 선택 및 가중치
        # ============================================
        self.use_focal_loss = True  # True: Focal Loss, False: 일반 BCE
        self.focal_alpha = 0.25  # Focal Loss alpha
        self.focal_gamma = 2.0   # Focal Loss gamma

        # 클래스별 가중치 (kick/snare가 적게 나오므로)
        self.class_weights = [2.0, 1.5, 1.0]  # [kick, snare, hihat]

        # ============================================
        # 클래스별 Threshold 설정
        # ============================================
        # 초기값 (학습 중 자동으로 최적화됨)
        self.thresholds = [0.5, 0.5, 0.5]  # [kick, snare, hihat]

        # Threshold 탐색 설정
        self.search_thresholds = True  # validation 시 threshold 탐색
        self.threshold_search_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        # Early stopping 설정
        self.patience = 10

        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._print_config()

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _print_config(self):
        """설정 출력"""
        print("\n" + "=" * 80)
        print("🚀 학습 설정 (최종 개선 버전)")
        print("=" * 80)
        print(f"📂 저장 경로: {self.save_dir}")
        print(f"📊 로그 경로: {self.log_dir}")
        print(f"\n⚡ 데이터 로딩: {'고속 모드 (사전 계산)' if self.use_precomputed else '일반 모드'}")
        print(f"🔢 에포크: {self.epochs}")
        print(f"📦 배치 크기: {self.batch_size}")
        print(f"🔄 Accumulation steps: {self.accumulation_steps} (실질 배치: {self.batch_size * self.accumulation_steps})")
        print(f"🎯 학습률: {self.learning_rate}")
        print(f"👷 워커 수: {self.num_workers}")
        print(f"🎨 Mixed precision: {self.use_mixed_precision}")

        loss_type = "Focal BCE Loss" if self.use_focal_loss else "Weighted BCE Loss"
        print(f"\n📐 손실함수: {loss_type}")
        if self.use_focal_loss:
            print(f"   Focal alpha: {self.focal_alpha}, gamma: {self.focal_gamma}")
        print(f"🎛️  클래스 가중치: kick={self.class_weights[0]}, snare={self.class_weights[1]}, hihat={self.class_weights[2]}")

        print(f"\n🎯 Threshold 초기값: kick={self.thresholds[0]}, snare={self.thresholds[1]}, hihat={self.thresholds[2]}")
        print(f"🔍 Threshold 자동 탐색: {self.search_thresholds}")

        print(f"\n🖥️  디바이스: {self.device}")
        if self.device.type == 'cuda':
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("=" * 80 + "\n")


def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class WeightedFocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss + 클래스별 가중치를 적용한 Binary Cross Entropy Loss

    Focal Loss는 쉬운 샘플의 기여도를 줄이고 어려운 샘플에 집중하도록 함
    - alpha: positive/negative 샘플의 균형 조절
    - gamma: 쉬운 샘플의 loss 감소 정도 (gamma=0이면 일반 BCE)

    Formula: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    """

    def __init__(self, pos_weights=None, alpha=0.25, gamma=2.0):
        super().__init__()
        self.pos_weights = pos_weights
        self.alpha = alpha
        self.gamma = gamma

        if pos_weights is not None:
            self.pos_weights = torch.FloatTensor(pos_weights)

    def forward(self, logits, targets, lengths=None):
        """
        Args:
            logits: (B, T, C) - 모델 출력
            targets: (B, T, C) - 정답 레이블
            lengths: (B,) - 각 시퀀스의 실제 길이 (optional)
        """
        if self.pos_weights is not None:
            self.pos_weights = self.pos_weights.to(logits.device)

        # BCE loss (reduction='none')
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weights,
            reduction='none'  # (B, T, C)
        )

        # Focal loss 계산
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)  # p_t
        focal_weight = (1 - p_t) ** self.gamma  # (1-p_t)^gamma

        # Alpha balancing
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        # Masking 적용 (padding 영역 제외)
        if lengths is not None:
            B, T, C = focal_loss.shape
            mask = torch.arange(T, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).expand(B, T, C)

            focal_loss = focal_loss * mask.float()

            # 유효한 프레임 수로 평균
            valid_count = mask.sum()
            if valid_count > 0:
                focal_loss = focal_loss.sum() / valid_count
            else:
                focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.mean()

        return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """클래스별 가중치를 적용한 Binary Cross Entropy Loss (Masking 포함)"""

    def __init__(self, pos_weights=None):
        super().__init__()
        self.pos_weights = pos_weights
        if pos_weights is not None:
            self.pos_weights = torch.FloatTensor(pos_weights)

    def forward(self, logits, targets, lengths=None):
        """
        Args:
            logits: (B, T, C)
            targets: (B, T, C)
            lengths: (B,) - 각 시퀀스의 실제 길이 (optional)
        """
        if self.pos_weights is not None:
            self.pos_weights = self.pos_weights.to(logits.device)

        # 기본 BCE loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weights,
            reduction='none'  # (B, T, C)
        )

        # Masking 적용 (padding 영역 제외)
        if lengths is not None:
            B, T, C = loss.shape
            mask = torch.arange(T, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).expand(B, T, C)

            loss = loss * mask.float()

            # 유효한 프레임 수로 평균
            valid_count = mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()

        return loss


def search_best_thresholds(all_probs, all_labels, all_lengths, config):
    """
    Grid search로 클래스별 최적 threshold 탐색

    Args:
        all_probs: List of (B, T, C) probability tensors
        all_labels: List of (B, T, C) label tensors
        all_lengths: List of (B,) length tensors
        config: 설정

    Returns:
        best_thresholds: [kick_th, snare_th, hihat_th]
        best_f1: 최고 F1 score
    """
    # 데이터 flat하게 모으기
    flat_probs = []
    flat_labels = []

    for probs_batch, labels_batch, lengths_batch in zip(all_probs, all_labels, all_lengths):
        B, T, C = probs_batch.shape
        lengths_batch = lengths_batch.cpu()

        for i in range(B):
            L = int(lengths_batch[i].item())
            probs_valid = probs_batch[i, :L, :].cpu().numpy()
            labels_valid = labels_batch[i, :L, :].cpu().numpy()

            flat_probs.append(probs_valid)
            flat_labels.append(labels_valid)

    if len(flat_probs) == 0:
        return config.thresholds, 0.0

    all_probs_np = np.vstack(flat_probs)  # (N, C)
    all_labels_np = np.vstack(flat_labels)  # (N, C)

    # Grid search
    best_f1 = 0.0
    best_thresholds = config.thresholds.copy()

    grid = config.threshold_search_grid

    print(f"\n🔍 Threshold 탐색 중... (grid: {grid})")

    # 모든 조합 탐색
    for th_kick, th_snare, th_hihat in product(grid, grid, grid):
        thresholds = [th_kick, th_snare, th_hihat]

        # 예측
        preds = np.zeros_like(all_probs_np, dtype=int)
        for class_idx in range(config.n_classes):
            preds[:, class_idx] = (all_probs_np[:, class_idx] >= thresholds[class_idx]).astype(int)

        # F1 계산
        f1_scores = []
        for class_idx in range(config.n_classes):
            f1 = f1_score(
                all_labels_np[:, class_idx],
                preds[:, class_idx],
                zero_division=0
            )
            f1_scores.append(f1)

        avg_f1 = np.mean(f1_scores)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thresholds = thresholds

    print(f"✅ 최적 Threshold 발견:")
    print(f"   kick={best_thresholds[0]}, snare={best_thresholds[1]}, hihat={best_thresholds[2]}")
    print(f"   F1={best_f1:.4f}")

    return best_thresholds, best_f1


def compute_metrics_from_lists(all_preds, all_labels, all_lengths, thresholds):
    """
    리스트로 수집된 예측/레이블/길이에서 메트릭 계산
    클래스별 threshold 적용

    Args:
        all_preds: List of (B, T, C) probability tensors (sigmoid 통과)
        all_labels: List of (B, T, C) label tensors
        all_lengths: List of (B,) length tensors
        thresholds: [kick_th, snare_th, hihat_th]
    """
    flat_preds = []
    flat_labels = []

    for preds_batch, labels_batch, lengths_batch in zip(all_preds, all_labels, all_lengths):
        probs_batch = preds_batch
        B, T, C = probs_batch.shape

        lengths_batch = lengths_batch.cpu()

        for i in range(B):
            L = int(lengths_batch[i].item())
            probs_valid = probs_batch[i, :L, :].cpu().numpy()
            labels_valid = labels_batch[i, :L, :].cpu().numpy()

            flat_preds.append(probs_valid)
            flat_labels.append(labels_valid)

    if len(flat_preds) == 0:
        return {
            'f1_kick': 0.0,
            'f1_snare': 0.0,
            'f1_hihat': 0.0,
            'f1_avg': 0.0,
            'precision_avg': 0.0,
            'recall_avg': 0.0
        }

    all_preds_np = np.vstack(flat_preds)  # (N, C)
    all_labels_np = np.vstack(flat_labels)  # (N, C)

    # 클래스별 threshold 적용
    all_preds_binary = np.zeros_like(all_preds_np, dtype=int)
    for class_idx in range(len(thresholds)):
        threshold = thresholds[class_idx]
        all_preds_binary[:, class_idx] = (all_preds_np[:, class_idx] >= threshold).astype(int)

    f1_per_class, precision_per_class, recall_per_class = [], [], []

    for class_idx in range(len(thresholds)):
        f1 = f1_score(
            all_labels_np[:, class_idx],
            all_preds_binary[:, class_idx],
            zero_division=0
        )
        precision = precision_score(
            all_labels_np[:, class_idx],
            all_preds_binary[:, class_idx],
            zero_division=0
        )
        recall = recall_score(
            all_labels_np[:, class_idx],
            all_preds_binary[:, class_idx],
            zero_division=0
        )

        f1_per_class.append(f1)
        precision_per_class.append(precision)
        recall_per_class.append(recall)

    metrics = {
        'f1_kick': f1_per_class[0],
        'f1_snare': f1_per_class[1],
        'f1_hihat': f1_per_class[2],
        'f1_avg': np.mean(f1_per_class),
        'precision_avg': np.mean(precision_per_class),
        'recall_avg': np.mean(recall_per_class),
        'precision_kick': precision_per_class[0],
        'precision_snare': precision_per_class[1],
        'precision_hihat': precision_per_class[2],
        'recall_kick': recall_per_class[0],
        'recall_snare': recall_per_class[1],
        'recall_hihat': recall_per_class[2]
    }

    return metrics


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, config):
    """1 에포크 학습 (Masking 적용)"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_lengths = []

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for batch_idx, (mel_specs, labels, lengths) in enumerate(pbar):
        mel_specs = mel_specs.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        lengths = lengths.to(config.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=config.use_mixed_precision):
            logits = model(mel_specs)
            loss = criterion(logits, labels, lengths)
            loss = loss / config.accumulation_steps

        scaler.scale(loss).backward()

        # 예측 저장
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_lengths.append(lengths.cpu())

        total_loss += loss.item() * config.accumulation_steps

        pbar.set_postfix({'loss': f"{loss.item() * config.accumulation_steps:.4f}"})

        # Gradient accumulation
        if (batch_idx + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 메모리 정리
        if (batch_idx + 1) % config.empty_cache_every_n_batches == 0:
            clear_gpu_memory()

    # 남은 gradient 처리
    if len(train_loader) % config.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)

    metrics = compute_metrics_from_lists(all_preds, all_labels, all_lengths, config.thresholds)
    metrics['loss'] = avg_loss
    return metrics


def validate(model, val_loader, criterion, config):
    """
    검증 (Masking 적용 + Threshold 탐색)

    Returns:
        metrics: 메트릭 딕셔너리
        best_thresholds: 탐색된 최적 threshold (search_thresholds=True일 때)
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_lengths = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", ncols=100)
        for batch_idx, (mel_specs, labels, lengths) in enumerate(pbar):
            mel_specs = mel_specs.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            lengths = lengths.to(config.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=config.use_mixed_precision):
                logits = model(mel_specs)
                loss = criterion(logits, labels, lengths)

            total_loss += loss.item()

            # Logits 저장 (threshold 탐색용)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_lengths.append(lengths.cpu())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # 메모리 정리
            if (batch_idx + 1) % config.empty_cache_every_n_batches == 0:
                clear_gpu_memory()

    avg_loss = total_loss / len(val_loader)

    # Sigmoid 적용
    all_probs = [torch.sigmoid(logits) for logits in all_logits]

    # Threshold 탐색
    best_thresholds = config.thresholds
    if config.search_thresholds:
        best_thresholds, _ = search_best_thresholds(
            all_probs, all_labels, all_lengths, config
        )

    # 메트릭 계산
    metrics = compute_metrics_from_lists(all_probs, all_labels, all_lengths, best_thresholds)
    metrics['loss'] = avg_loss

    return metrics, best_thresholds
"""
드럼 타격 검출 모델 학습 스크립트 - Part 2
Checkpoint 저장 및 메인 학습 루프
"""

import os
import torch
import json


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, filename, thresholds=None):
    """
    체크포인트 저장 (최적 threshold 포함)

    Args:
        model: 모델
        optimizer: Optimizer
        scheduler: Scheduler
        epoch: 현재 에포크
        metrics: 메트릭 딕셔너리
        config: 설정
        filename: 저장 파일명
        thresholds: 최적 threshold [kick, snare, hihat] (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': {
            'n_mels': config.n_mels,
            'n_classes': config.n_classes,
            'cnn_channels': config.cnn_channels,
            'gru_hidden': config.gru_hidden,
            'gru_layers': config.gru_layers,
            'dropout': config.dropout,
            'thresholds': thresholds if thresholds is not None else config.thresholds,
            'class_weights': config.class_weights,
            'use_focal_loss': config.use_focal_loss,
            'focal_alpha': config.focal_alpha if config.use_focal_loss else None,
            'focal_gamma': config.focal_gamma if config.use_focal_loss else None,
        }
    }

    save_path = os.path.join(config.save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"💾 체크포인트 저장: {save_path}")
    if thresholds is not None:
        print(f"   최적 Threshold: kick={thresholds[0]:.2f}, snare={thresholds[1]:.2f}, hihat={thresholds[2]:.2f}")


def train(config):
    """전체 학습 루프"""
    print("\n" + "=" * 80)
    print("🚀 학습 시작 (최종 개선 버전)")
    print("=" * 80)

    # 데이터셋 설정
    data_config = DrumDatasetConfig()
    data_config.use_precomputed = config.use_precomputed
    data_config.precomputed_root = config.precomputed_root

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = get_dataloaders(
        data_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # 모델 생성
    print("\n📦 모델 초기화 (최종 개선 버전)...")
    model = DrumOnsetDetector(
        n_mels=config.n_mels,
        n_classes=config.n_classes,
        cnn_channels=config.cnn_channels,
        gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers,
        dropout=config.dropout
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   총 파라미터: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")

    # 손실함수 선택
    if config.use_focal_loss:
        print(f"\n📐 손실함수: Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        criterion = WeightedFocalBCEWithLogitsLoss(
            pos_weights=config.class_weights,
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )
    else:
        print(f"\n📐 손실함수: Weighted BCE Loss")
        criterion = WeightedBCEWithLogitsLoss(pos_weights=config.class_weights)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_mixed_precision)

    # 학습 기록
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [],
        'thresholds_history': []  # Threshold 변화 추적
    }

    best_f1 = 0.0
    best_thresholds = config.thresholds.copy()
    patience_counter = 0

    print("\n" + "=" * 80)
    print("🎯 학습 시작!")
    print("=" * 80)

    for epoch in range(1, config.epochs + 1):
        print(f"\n📍 Epoch [{epoch}/{config.epochs}]")
        print("-" * 80)

        # 학습
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config
        )

        # 검증 (Threshold 탐색 포함)
        val_metrics, searched_thresholds = validate(model, val_loader, criterion, config)

        # Threshold 업데이트
        if config.search_thresholds:
            config.thresholds = searched_thresholds

        # Scheduler step
        scheduler.step(val_metrics['f1_avg'])

        # 결과 출력
        print(f"\n📊 [TRAIN] Loss: {train_metrics['loss']:.4f} | "
              f"F1_avg: {train_metrics['f1_avg']:.4f} | "
              f"F1_kick: {train_metrics['f1_kick']:.4f} | "
              f"F1_snare: {train_metrics['f1_snare']:.4f} | "
              f"F1_hihat: {train_metrics['f1_hihat']:.4f}")

        print(f"📊 [VAL]   Loss: {val_metrics['loss']:.4f} | "
              f"F1_avg: {val_metrics['f1_avg']:.4f} | "
              f"F1_kick: {val_metrics['f1_kick']:.4f} | "
              f"F1_snare: {val_metrics['f1_snare']:.4f} | "
              f"F1_hihat: {val_metrics['f1_hihat']:.4f}")

        print(f"   P_kick: {val_metrics['precision_kick']:.3f} | "
              f"R_kick: {val_metrics['recall_kick']:.3f} | "
              f"P_snare: {val_metrics['precision_snare']:.3f} | "
              f"R_snare: {val_metrics['recall_snare']:.3f} | "
              f"P_hihat: {val_metrics['precision_hihat']:.3f} | "
              f"R_hihat: {val_metrics['recall_hihat']:.3f}")

        # 기록 저장
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1_avg'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1_avg'])
        history['thresholds_history'].append(config.thresholds.copy())

        # Best model 저장
        if val_metrics['f1_avg'] > best_f1:
            best_f1 = val_metrics['f1_avg']
            best_thresholds = config.thresholds.copy()
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, 'best_model.pt', thresholds=best_thresholds
            )
            print(f"🎉 새로운 Best F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{config.patience}")

        # 주기적 저장
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, f'checkpoint_epoch_{epoch}.pt', thresholds=config.thresholds
            )

        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⛔ Early stopping at epoch {epoch}")
            break

        # 메모리 정리
        clear_gpu_memory()

    # 최종 평가
    print("\n" + "=" * 80)
    print("🎯 최종 테스트 평가")
    print("=" * 80)

    best_model_path = os.path.join(config.save_dir, 'best_model.pt')
    checkpoint = torch.load(
        best_model_path,
        map_location=config.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Best threshold 로드
    loaded_thresholds = checkpoint['config']['thresholds']
    print(f"\n✅ Best threshold 로드: kick={loaded_thresholds[0]:.2f}, snare={loaded_thresholds[1]:.2f}, hihat={loaded_thresholds[2]:.2f}")

    # 테스트 평가 (threshold 탐색 안함)
    config.search_thresholds = False
    config.thresholds = loaded_thresholds

    test_metrics, _ = validate(model, test_loader, criterion, config)

    print(f"\n📊 [TEST] Loss: {test_metrics['loss']:.4f} | "
          f"F1_avg: {test_metrics['f1_avg']:.4f} | "
          f"F1_kick: {test_metrics['f1_kick']:.4f} | "
          f"F1_snare: {test_metrics['f1_snare']:.4f} | "
          f"F1_hihat: {test_metrics['f1_hihat']:.4f}")

    print(f"   P_kick: {test_metrics['precision_kick']:.3f} | "
          f"R_kick: {test_metrics['recall_kick']:.3f} | "
          f"P_snare: {test_metrics['precision_snare']:.3f} | "
          f"R_snare: {test_metrics['recall_snare']:.3f} | "
          f"P_hihat: {test_metrics['precision_hihat']:.3f} | "
          f"R_hihat: {test_metrics['recall_hihat']:.3f}")

    # 학습 기록 저장
    history['test_metrics'] = test_metrics
    history['best_thresholds'] = best_thresholds

    history_path = os.path.join(config.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n💾 학습 기록 저장: {history_path}")

    # 최종 결과 요약
    print("\n" + "=" * 80)
    print("🎊 학습 완료 - 최종 결과 요약")
    print("=" * 80)
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Test F1: {test_metrics['f1_avg']:.4f}")
    print(f"Best Thresholds: kick={best_thresholds[0]:.2f}, snare={best_thresholds[1]:.2f}, hihat={best_thresholds[2]:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    config = CompleteTrainConfig()
    clear_gpu_memory()
    train(config)