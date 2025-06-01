import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:spas_mobile/android/app/lib/services/tflite_service.dart';
import 'package:spas_mobile/android/app/lib/pages/voice_assistant_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Инициализируем сервис TensorFlow Lite
  final tfliteService = TFLiteService();
  await tfliteService.initModels();

  runApp(
    MultiProvider(
      providers: [
        Provider<TFLiteService>.value(value: tfliteService),
      ],
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Речевой ассистент',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: const VoiceAssistantPage(),
    );
  }
}