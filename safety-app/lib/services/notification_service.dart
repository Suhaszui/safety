// lib/services/notification_service.dart
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

// A GlobalKey for navigation from outside the widget tree
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

class NotificationService {
  final FirebaseMessaging _fcm = FirebaseMessaging.instance;
  final FlutterLocalNotificationsPlugin _localNotifications = FlutterLocalNotificationsPlugin();

  // 1. Initialize permissions, listeners, and the local notifications plugin
  Future<void> initialize() async {
    await _fcm.requestPermission();
    final fcmToken = await _fcm.getToken();
    print('🔑 FCM Registration Token: $fcmToken');

    await _initializeLocalNotifications();
    _setupListeners();
  }
  
  // 2. Set up the local notifications plugin
  Future<void> _initializeLocalNotifications() async {
    // Define high-importance channel for Android
    const AndroidNotificationChannel channel = AndroidNotificationChannel(
      'high_importance_channel', // id
      'High Importance Notifications', // title
      description: 'This channel is used for important safety alerts.', // description
      importance: Importance.max,
    );
    
    await _localNotifications.resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(channel);

    // Initialize the plugin for Android and iOS
    const AndroidInitializationSettings initializationSettingsAndroid = AndroidInitializationSettings('@mipmap/ic_launcher');
    const DarwinInitializationSettings initializationSettingsIOS = DarwinInitializationSettings();
    const InitializationSettings initializationSettings = InitializationSettings(
      android: initializationSettingsAndroid,
      iOS: initializationSettingsIOS,
    );
    await _localNotifications.initialize(initializationSettings);
  }

  // 3. Set up the FCM listeners
  void _setupListeners() {
    // This handles messages that arrive when the app is in the FOREGROUND
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      print('Got a message whilst in the foreground!');
      final notification = message.notification;
      final android = message.notification?.android;

      if (notification != null && android != null) {
        // If a notification arrives while the app is open, show it as a local notification
        _localNotifications.show(
          notification.hashCode,
          notification.title,
          notification.body,
          NotificationDetails(
            android: AndroidNotificationDetails(
              'high_importance_channel',
              'High Importance Notifications',
              channelDescription: 'This channel is used for important safety alerts.',
              icon: android.smallIcon,
            ),
          ),
        );
      }
    });

    // This handles what happens when the user TAPS on a notification
    FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
      print('User tapped on a notification to open the app!');
      // Navigate to the alert screen, passing the data payload
      navigatorKey.currentState?.pushNamed('/deviation', arguments: message.data);
    });
  }
}