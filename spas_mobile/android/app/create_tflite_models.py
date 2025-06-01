# Скрипт для создания моделей TensorFlow Lite для Flutter-приложения
# Запустите этот скрипт, чтобы создать модели для шумоподавления и распознавания речи

import tensorflow as tf
import numpy as np
import os

# Создаем директорию для моделей
os.makedirs('assets/models', exist_ok=True)

# Создание модели шумоподавления
def create_noise_reduction_model():
# Определяем входные слои
input_audio = tf.keras.layers.Input(shape=(None,), name='audio_input')
input_noise = tf.keras.layers.Input(shape=(None,), name='noise_profile')

# Преобразуем входные данные в спектрограммы
# Используем короткое оконное преобразование Фурье (STFT)
audio_stft = tf.signal.stft(input_audio, frame_length=512, frame_step=128)
noise_stft = tf.signal.stft(input_noise, frame_length=512, frame_step=128)

# Получаем магнитуду и фазу
audio_spec = tf.abs(audio_stft)
audio_phase = tf.math.angle(audio_stft)
noise_spec = tf.abs(noise_stft)

# Конкатенируем спектры аудио и шума
concat_input = tf.keras.layers.Concatenate()([audio_spec, noise_spec])

# Создаем маску подавления шума с использованием RNN
x = tf.keras.layers.Reshape((-1, concat_input.shape[-1]))(concat_input)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
mask = tf.keras.layers.Dense(257, activation='sigmoid')(x)

# Применяем маску к спектру аудио
enhanced_spec = tf.multiply(audio_spec, mask)

# Восстанавливаем сигнал из спектра и фазы
enhanced_stft = tf.complex(enhanced_spec, tf.zeros_like(enhanced_spec))
enhanced_stft = enhanced_stft * tf.exp(tf.complex(0., audio_phase))
output = tf.signal.inverse_stft(enhanced_stft, frame_length=512, frame_step=128)

# Создаем модель
model = tf.keras.Model(inputs=[input_audio, input_noise], outputs=output)
model.compile(optimizer='adam', loss='mse')

print("Модель шумоподавления создана")

# Конвертируем в TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Сохраняем модель
with open('assets/models/noise_reduction_model.tflite', 'wb') as f:
f.write(tflite_model)

print("Модель шумоподавления сохранена в assets/models/noise_reduction_model.tflite")

# Создание модели распознавания речи
def create_speech_recognition_model():
# Определяем входные слои
input_audio = tf.keras.layers.Input(shape=(16000,), name='audio_input')  # 1 сек аудио при 16кГц
input_accent = tf.keras.layers.Input(shape=(64,), name='accent_profile')  # Профиль акцента

# Извлечение MFCC признаков
# Преобразуем аудио в спектрограмму
stft = tf.signal.stft(input_audio, frame_length=512, frame_step=256)
spectrogram = tf.abs(stft)

# Преобразуем в мел-спектрограмму
num_mel_bins = 40
mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
num_mel_bins=num_mel_bins,
num_spectrogram_bins=spectrogram.shape[-1],
sample_rate=16000,
lower_edge_hertz=20,
upper_edge_hertz=8000
)
mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)

# Применяем логарифм к мел-спектрограмме
log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

# Нормализуем
mfccs = tf.keras.layers.LayerNormalization()(log_mel_spectrogram)

# Извлечение признаков с помощью CNN
x = tf.keras.layers.Reshape((-1, num_mel_bins, 1))(mfccs)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)

# Конкатенируем с профилем акцента
x = tf.keras.layers.Concatenate()([x, input_accent])

# Полносвязные слои
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Выходной слой - вероятности для каждой команды
num_commands = 6  # "открыть", "закрыть", "позвонить", "сообщение", "музыка", "погода"
output = tf.keras.layers.Dense(num_commands, activation='softmax')(x)

# Создаем модель
model = tf.keras.Model(inputs=[input_audio, input_accent], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Модель распознавания речи создана")

# Конвертируем в TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Сохраняем модель
with open('assets/models/speech_recognition_model.tflite', 'wb') as f:
f.write(tflite_model)

print("Модель распознавания речи сохранена в assets/models/speech_recognition_model.tflite")

if __name__ == "__main__":
print("Создание моделей TensorFlow Lite для Flutter-приложения...")
create_noise_reduction_model()
create_speech_recognition_model()
print("Готово! Обе модели созданы и сохранены в директории assets/models/")