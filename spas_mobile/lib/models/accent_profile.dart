import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:fftea/fftea.dart';
import 'dart:convert';

import '../models/accent_profile.dart';
import 'recognition_result.dart';

/// Сервис для работы с TensorFlow Lite моделями
class TFLiteService {
  static const String _noiseReductionModelPath = 'assets/models/noise_reduction_model.tflite';
  static const String _speechRecognitionModelPath = 'assets/models/speech_recognition_model.tflite';
  static const String _accentProfileKey = 'user_accent_profile';

  Interpreter? _noiseReductionInterpreter;
  Interpreter? _speechRecognitionInterpreter;

  final List<String> _supportedCommands = [
    "открыть", "закрыть", "позвонить", "сообщение", "музыка", "погода"
  ];

  /// Инициализирует TensorFlow Lite модели
  Future<void> initModels() async {
    try {
      // Загружаем модель шумоподавления
      final noiseReductionModelData = await _loadModelData(_noiseReductionModelPath);
      _noiseReductionInterpreter = Interpreter.fromBuffer(noiseReductionModelData);

      // Загружаем модель распознавания речи
      final speechRecognitionModelData = await _loadModelData(_speechRecognitionModelPath);
      _speechRecognitionInterpreter = Interpreter.fromBuffer(speechRecognitionModelData);
    } catch (e) {
      print('Ошибка инициализации TFLite моделей: $e');
      throw Exception('Не удалось инициализировать модели: $e');
    }
  }

  /// Загружает данные модели из ассетов
  Future<Uint8List> _loadModelData(String assetPath) async {
    try {
      final ByteData modelData = await rootBundle.load(assetPath);
      return modelData.buffer.asUint8List();
    } catch (e) {
      throw Exception('Не удалось загрузить модель $assetPath: $e');
    }
  }

  /// Применяет шумоподавление к аудиосигналу
  Future<List<double>> applyNoiseReduction(
      List<double> audioData,
      List<double> backgroundNoise
      ) async {
    if (_noiseReductionInterpreter == null) {
      throw Exception('Модель шумоподавления не инициализирована');
    }

    // Нормализуем входные данные
    final int frameSize = 512;
    final int hopSize = 128;

    // Создаем входные и выходные тензоры
    final inputAudio = _convertToTensor(audioData);
    final inputNoise = _convertToTensor(backgroundNoise);
    final outputBuffer = List<double>.filled(audioData.length, 0.0);

    // Создаем входные и выходные карты
    final inputs = [inputAudio, inputNoise];
    final outputs = {0: outputBuffer};

    // Запускаем модель шумоподавления
    await _noiseReductionInterpreter!.runForMultipleInputs(inputs, outputs);

    return outputBuffer;
  }

  /// Распознает речь из аудиосигнала
  Future<RecognitionResult> recognizeSpeech(
      List<double> cleanedAudio,
      AccentProfile? accentProfile
      ) async {
    if (_speechRecognitionInterpreter == null) {
      throw Exception('Модель распознавания речи не инициализирована');
    }

    // Подготавливаем входные данные
    final inputSize = 16000; // 1 секунда аудио при 16кГц

    // Нормализуем длину входных данных
    List<double> normalizedAudio;
    if (cleanedAudio.length >= inputSize) {
      normalizedAudio = cleanedAudio.sublist(0, inputSize);
    } else {
      normalizedAudio = List<double>.filled(inputSize, 0.0);
      for (int i = 0; i < cleanedAudio.length; i++) {
        normalizedAudio[i] = cleanedAudio[i];
      }
    }

    // Извлекаем MFCC признаки
    final features = _extractMFCCFeatures(normalizedAudio);

    // Преобразуем признаки в тензор
    final inputFeatures = _convertToTensor(features);

    // Подготавливаем профиль акцента или создаем нейтральный
    final accentTensor = _createAccentTensor(accentProfile);

    // Создаем выходной буфер для вероятностей команд
    final outputSize = _supportedCommands.length;
    final outputBuffer = List<double>.filled(outputSize, 0.0);

    // Запускаем модель распознавания речи
    final inputs = [inputFeatures, accentTensor];
    final outputs = {0: outputBuffer};

    await _speechRecognitionInterpreter!.runForMultipleInputs(inputs, outputs);

    // Находим команду с максимальной вероятностью
    int maxIndex = 0;
    double maxProb = outputBuffer[0];

    for (int i = 1; i < outputBuffer.length; i++) {
      if (outputBuffer[i] > maxProb) {
        maxProb = outputBuffer[i];
        maxIndex = i;
      }
    }

    // Возвращаем результат распознавания
    return RecognitionResult(
        command: _supportedCommands[maxIndex],
        confidence: maxProb
    );
  }

  /// Обновляет профиль акцента пользователя
  Future<AccentProfile> updateAccentProfile(
      List<double> audioData,
      String recognizedCommand,
      double confidence,
      AccentProfile? currentProfile
      ) async {
    // Если уверенность низкая, не обновляем профиль
    if (confidence < 0.7) {
      return currentProfile ?? AccentProfile(features: List<double>.filled(64, 0.0));
    }

    // Создаем новый профиль, если он еще не существует
    final profile = currentProfile ?? AccentProfile(features: List<double>.filled(64, 0.0));

    // Извлекаем признаки акцента из аудио
    final accentFeatures = _extractAccentFeatures(audioData, recognizedCommand);

    // Обновляем профиль с использованием экспоненциального скользящего среднего
    final alpha = 0.1; // Коэффициент обучения

    for (int i = 0; i < profile.features.length; i++) {
      if (i < accentFeatures.length) {
        profile.features[i] = (1 - alpha) * profile.features[i] + alpha * accentFeatures[i];
      }
    }

    // Сохраняем обновленный профиль
    await _saveAccentProfile(profile);

    return profile;
  }

  /// Загружает сохраненный профиль акцента
  Future<AccentProfile?> loadAccentProfile() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final String? profileJson = prefs.getString(_accentProfileKey);

      if (profileJson != null) {
        final Map<String, dynamic> profileMap = json.decode(profileJson);
        return AccentProfile.fromJson(profileMap);
      }
    } catch (e) {
      print('Ошибка загрузки профиля акцента: $e');
    }

    return null;
  }

  /// Сохраняет профиль акцента
  Future<void> _saveAccentProfile(AccentProfile profile) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final String profileJson = json.encode(profile.toJson());
      await prefs.setString(_accentProfileKey, profileJson);
    } catch (e) {
      print('Ошибка сохранения профиля акцента: $e');
    }
  }

  /// Преобразует список чисел в тензор для TFLite
  List<double> _convertToTensor(List<double> data) {
    return data;
  }

  /// Создает тензор профиля акцента
  List<double> _createAccentTensor(AccentProfile? profile) {
    if (profile == null) {
      return List<double>.filled(64, 0.0);
    }
    return profile.features;
  }

  /// Извлекает MFCC признаки из аудиосигнала
  List<double> _extractMFCCFeatures(List<double> audioData) {
    // Размер окна и шаг для STFT
    final int frameSize = 512;
    final int hopSize = 256;
    final int numFrames = ((audioData.length - frameSize) ~/ hopSize) + 1;

    // Размер MFCC вектора
    final int numMfcc = 40;

    // Результирующий вектор признаков
    final features = List<double>.filled(numFrames * numMfcc, 0.0);

    // Создаем объект FFT для спектрального анализа
    final fft = FFT(frameSize);

    // Окно Хэмминга для уменьшения эффекта утечки спектра
    final hammingWindow = List<double>.generate(frameSize,
            (i) => 0.54 - 0.46 * cos(2 * pi * i / (frameSize - 1))
    );

    // Обрабатываем каждый фрейм
    for (int frameIndex = 0; frameIndex < numFrames; frameIndex++) {
      final startIndex = frameIndex * hopSize;
      final endIndex = min(startIndex + frameSize, audioData.length);

      // Выбираем фрейм и применяем окно
      final frame = List<double>.filled(frameSize, 0.0);
      for (int i = 0; i < endIndex - startIndex; i++) {
        frame[i] = audioData[startIndex + i] * hammingWindow[i];
      }

      // Вычисляем FFT
      final fftResult = fft.realFft(frame);

      // Вычисляем спектр мощности
      final powerSpectrum = List<double>.filled(frameSize ~/ 2 + 1, 0.0);
      for (int i = 0; i < powerSpectrum.length; i++) {
        final real = fftResult[i * 2];
        final imag = i > 0 && i < fftResult.length ~/ 2 ? fftResult[i * 2 + 1] : 0.0;
        powerSpectrum[i] = real * real + imag * imag;
      }

      // Применяем мел-фильтры и вычисляем MFCC
      final mfcc = _computeMFCC(powerSpectrum, numMfcc);

      // Сохраняем MFCC для текущего фрейма
      for (int i = 0; i < numMfcc; i++) {
        features[frameIndex * numMfcc + i] = mfcc[i];
      }
    }

    return features;
  }

  /// Вычисляет MFCC для спектра мощности
  List<double> _computeMFCC(List<double> powerSpectrum, int numMfcc) {
    final int numFilters = 40;
    final double sampleRate = 16000.0;

    // Создаем мел-фильтры
    final melFilters = _createMelFilterbank(powerSpectrum.length, numFilters, sampleRate);

    // Применяем мел-фильтры к спектру мощности
    final melEnergies = List<double>.filled(numFilters, 0.0);
    for (int i = 0; i < numFilters; i++) {
      for (int j = 0; j < powerSpectrum.length; j++) {
        melEnergies[i] += powerSpectrum[j] * melFilters[i][j];
      }
      // Применяем логарифм к энергиям
      melEnergies[i] = melEnergies[i] > 0.0 ? log(melEnergies[i]) : 0.0;
    }

    // Применяем DCT для получения MFCC
    final mfcc = _applyDCT(melEnergies, numMfcc);

    return mfcc;
  }

  /// Создает мел-фильтры для преобразования спектра
  List<List<double>> _createMelFilterbank(int fftSize, int numFilters, double sampleRate) {
    final double lowFreq = 0.0;
    final double highFreq = sampleRate / 2.0;

    // Преобразуем частоты в мел-шкалу
    final double lowMel = _hzToMel(lowFreq);
    final double highMel = _hzToMel(highFreq);

    // Равномерно распределяем точки в мел-шкале
    final List<double> melPoints = List<double>.filled(numFilters + 2, 0.0);
    for (int i = 0; i < numFilters + 2; i++) {
      melPoints[i] = lowMel + i * (highMel - lowMel) / (numFilters + 1);
    }

    // Преобразуем обратно в Гц
    final List<double> hzPoints = melPoints.map((mel) => _melToHz(mel)).toList();

    // Преобразуем в индексы бинов FFT
    final List<int> bins = hzPoints.map((hz) =>
        ((fftSize - 1) * hz / sampleRate).round()
    ).toList();

    // Создаем фильтры
    final List<List<double>> filterbank = List.generate(
        numFilters, (_) => List<double>.filled(fftSize, 0.0)
    );

    for (int i = 0; i < numFilters; i++) {
      for (int j = bins[i]; j < bins[i+1]; j++) {
        filterbank[i][j] = (j - bins[i]) / (bins[i+1] - bins[i]);
      }
      for (int j = bins[i+1]; j < bins[i+2]; j++) {
        filterbank[i][j] = (bins[i+2] - j) / (bins[i+2] - bins[i+1]);
      }
    }

    return filterbank;
  }

  /// Применяет дискретное косинусное преобразование (DCT)
  List<double> _applyDCT(List<double> melEnergies, int numCoefficients) {
    final int numFilters = melEnergies.length;
    final List<double> mfcc = List<double>.filled(numCoefficients, 0.0);

    for (int i = 0; i < numCoefficients; i++) {
      for (int j = 0; j < numFilters; j++) {
        mfcc[i] += melEnergies[j] * cos(pi * i * (2 * j + 1) / (2 * numFilters));
      }
    }

    return mfcc;
  }

  /// Преобразует Гц в мел-шкалу
  double _hzToMel(double hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
  }

  /// Преобразует мел-шкалу в Гц
  double _melToHz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
  }

  /// Извлекает признаки акцента из аудиосигнала
  List<double> _extractAccentFeatures(List<double> audioData, String command) {
    // Индекс команды
    final int commandIdx = _supportedCommands.indexOf(command);

    // Извлекаем MFCC признаки
    final List<double> mfccFeatures = _extractMFCCFeatures(audioData);

    // Создаем вектор признаков акцента
    final List<double> accentFeatures = List<double>.filled(64, 0.0);

    // Рассчитываем статистические характеристики MFCC
    double mean = 0.0;
    for (final feature in mfccFeatures) {
      mean += feature;
    }
    mean /= mfccFeatures.length;

    double variance = 0.0;
    for (final feature in mfccFeatures) {
      variance += (feature - mean) * (feature - mean);
    }
    variance /= mfccFeatures.length;

    // Заполняем вектор признаков
    int featureIdx = 0;

    // Добавляем статистические характеристики
    accentFeatures[featureIdx++] = mean;
    accentFeatures[featureIdx++] = variance;
    accentFeatures[featureIdx++] = sqrt(variance);

    // Добавляем команду как one-hot вектор
    for (int i = 0; i < _supportedCommands.length; i++) {
      accentFeatures[featureIdx++] = (i == commandIdx) ? 1.0 : 0.0;
    }

    return accentFeatures;
  }

  /// Освобождает ресурсы
  void dispose() {
    _noiseReductionInterpreter?.close();
    _speechRecognitionInterpreter?.close();
  }
}