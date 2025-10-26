# modeling/src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


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
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # 과적합 방지를 위한 드롭아웃
        Dense(num_classes, activation='softmax')  # 최종 출력층
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model