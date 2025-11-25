# modeling/src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics


def build_cnn_model(input_shape, num_classes):
    """드럼 사운드 분류를 위한 2D CNN 모델을 생성합니다."""
    model = Sequential([
        # 첫 번째 Conv-Pool 블록
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # 두 번째 Conv-Pool 블록
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # 세 번째 Conv-Pool 블록
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # 분류기 (Classifier)
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.3),  # 표현력 감소에 맞춘 드롭아웃
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid')  # 최종 출력층
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            metrics.BinaryAccuracy(threshold=0.5, name='binary_acc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(multi_label=True, name='auc')
        ]
    )

    return model
