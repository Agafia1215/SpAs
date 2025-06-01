import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:async';

import '../services/audio_service.dart';
import '../services/tflite_service.dart';
import '../models/accent_profile.dart';

class VoiceAssistantPage extends StatefulWidget {
  const VoiceAssistantPage({Key? key}) : super(key: key);

  @override
  _VoiceAssistantPageState createState() => _VoiceAssistantPageState();
}

class _VoiceAssistantPageState extends State<VoiceAssistantPage> {
  final AudioService _audioService = AudioService();
  bool _isRecording = false;
  String _statusText = "Готов к работе";
  String _recognizedText = "Нажмите 'Начать запись' и произнесите команду";
  bool _isInitialized = false;

  // Профиль акцента пользователя
  AccentProfile? _userAccentProfile;

  // Поддерживаемые команды
  final List<String> _supportedCommands = [
    "открыть", "закрыть", "позвонить", "сообщение", "музыка", "погода"
  ];

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _initializeServices();
  }

  @override
  void dispose() {
    _audioService.dispose();
    super.dispose();
  }

  Future<void> _requestPermissions() async {
    // Запрашиваем разрешение на запись аудио
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() {
        _statusText = "Для работы приложения необходим доступ к микрофону";
      });
    }
  }

  Future<void> _initializeServices() async {
    try {
      final tfliteService = Provider.of<TFLiteService>(context, listen: false);

      // Инициализируем аудио сервис
      await _audioService.initialize();

      // Загружаем сохраненный профиль акцента, если есть
      _userAccentProfile = await tfliteService.loadAccentProfile();

      setState(() {
        _isInitialized = true;
        _statusText = "Модели загружены, ассистент готов к работе";
      });
    } catch (e) {
      setState(() {
        _statusText = "Ошибка инициализации: $e";
      });
    }
  }

  Future<void> _startRecording() async {
    if (_isRecording || !_isInitialized) return;

    setState(() {
      _isRecording = true;
      _statusText = "Запись...";
    });

    try {
      // Начинаем запись аудио
      await _audioService.startRecording();

      // Собираем фоновый шум в течение 1 секунды
      await Future.delayed(const Duration(seconds: 1));
      final backgroundNoise = await _audioService.collectBackgroundNoise();

      // Продолжаем запись и обработку в реальном времени
      _audioService.audioStream.listen((audioData) async {
        if (!_isRecording) return;

        // Проверяем уровень громкости
        final rmsLevel = _calculateRMS(audioData);

        // Если уровень выше порога, обрабатываем
        if (rmsLevel > -40) { // -40 дБ как порог для активной речи
          final tfliteService = Provider.of<TFLiteService>(context, listen: false);

          // Применяем шумоподавление
          final cleanedAudio = await tfliteService.applyNoiseReduction(
              audioData,
              backgroundNoise
          );

          // Распознаем речь
          final result = await tfliteService.recognizeSpeech(
              cleanedAudio,
              _userAccentProfile
          );

          // Если результат достаточно уверенный, обновляем UI
          if (result.confidence > 0.7) {
            setState(() {
              _recognizedText = result.command;
            });

            // Обрабатываем команду
            _processCommand(result.command);

            // Обновляем профиль акцента
            _userAccentProfile = await tfliteService.updateAccentProfile(
                cleanedAudio,
                result.command,
                result.confidence,
                _userAccentProfile
            );
          }
        }
      });
    } catch (e) {
      setState(() {
        _statusText = "Ошибка записи: $e";
        _isRecording = false;
      });
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    setState(() {
      _isRecording = false;
      _statusText = "Запись остановлена";
    });

    await _audioService.stopRecording();
  }

  void _processCommand(String command) {
    setState(() {
      _statusText = "Выполняю команду: $command";
    });

    // Здесь будет логика обработки различных команд
    switch (command) {
      case "открыть":
      // Логика для открытия приложения
        break;
      case "закрыть":
      // Логика для закрытия приложения
        break;
      case "позвонить":
      // Логика для звонка
        break;
      case "сообщение":
      // Логика для отправки сообщения
        break;
      case "музыка":
      // Логика для запуска музыки
        break;
      case "погода":
      // Логика для отображения погоды
        break;
    }
  }

  // Расчет среднеквадратичного значения (RMS) для аудиоданных
  double _calculateRMS(List<double> audioData) {
    double sum = 0;
    for (var sample in audioData) {
      sum += sample * sample;
    }
    final rms = (sum / audioData.length).sqrt();

    // Преобразование в дБ
    return rms > 0 ? 20 * (rms / 1.0).log() / 2.303 : -100.0;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Речевой ассистент с адаптацией'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Этот ассистент распознает речь в шумной обстановке и адаптируется к вашему акценту без отправки данных в облако.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),

            // Карточка с распознанным текстом
            Card(
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              child: Container(
                height: 200,
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Распознанный текст:',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Expanded(
                      child: Center(
                        child: Text(
                          _recognizedText,
                          style: const TextStyle(fontSize: 24),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Поддерживаемые команды
            const Text(
              'Поддерживаемые команды:',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              _supportedCommands.join(', '),
              style: const TextStyle(fontSize: 16),
            ),

            const SizedBox(height: 24),

            // Текущий статус
            Text(
              _statusText,
              style: const TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),

            const Spacer(),

            // Кнопки управления
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isInitialized && !_isRecording
                        ? _startRecording
                        : null,
                    child: const Text('Начать запись'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isRecording ? _stopRecording : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white,
                    ),
                    child: const Text('Остановить'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}