// lib/services/location_callback_handler.dart
import 'dart:convert';
import 'package:background_locator_2/location_dto.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_messaging/firebase_messaging.dart';

// This is a special top-level function that runs in a separate process.
// It MUST be outside of any class.
@pragma('vm:entry-point')
void callback(LocationDto locationDto) async {
  print('Location Ping Received in Background: ${locationDto.latitude}, ${locationDto.longitude}');
  
  // For the prototype, we'll use a hardcoded user_id.
  const String userId = 'prototype_user_01';

  // We need to get the FCM token to identify the device.
  final String? fcmToken = await FirebaseMessaging.instance.getToken();

  if (fcmToken == null) {
    print("FATAL: Could not get FCM token in background.");
    return;
  }
  
  // IMPORTANT: Replace <YOUR_SERVER_IP> with your computer's actual local IP address.
  // Example: 'http://192.168.1.10:5000/location_ping'
  // Do NOT use 'localhost' or '127.0.0.1' as that will not work from an emulator/phone.
  final url = Uri.parse('http://172.18.213.12:5000/location_ping');
  
  try {
    await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'user_id': userId,
        'lat': locationDto.latitude,
        'lon': locationDto.longitude,
        'fcm_token': fcmToken,
      }),
    );
    print('Successfully sent location ping to server.');
  } catch (e) {
    print('Error sending location to server: $e');
  }
}
