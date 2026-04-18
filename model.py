import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                              confusion_matrix)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM,
                                        Dense, Dropout,
                                        SpatialDropout1D)
from tensorflow.keras.callbacks import EarlyStopping

def build_model(vocab_size=10000, max_len=50,
                num_classes=3):
    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.summary()

    es = EarlyStopping(monitor='val_loss',
                        patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=10, batch_size=64,
        validation_split=0.1,
        callbacks=[es], verbose=1
    )

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_curves.png', dpi=150)
    plt.show()

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc:.2%}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred,
          target_names=['negative','neutral','positive']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['neg','neu','pos'],
               yticklabels=['neg','neu','pos'])
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.show()
    return model