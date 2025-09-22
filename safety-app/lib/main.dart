import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// Firebase and Services
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'services/notification_service.dart';

// Screens
import 'package:smart_safety_app/screens/splash_screen.dart';
import 'package:smart_safety_app/screens/language_screen.dart';
import 'package:smart_safety_app/screens/login_screen.dart';
import 'package:smart_safety_app/screens/signup_screen.dart';
import 'package:smart_safety_app/screens/terms_screen.dart';
import 'package:smart_safety_app/screens/instructions_screen.dart';
import 'package:smart_safety_app/screens/dashboard_screen.dart';
import 'package:smart_safety_app/screens/itinerary_screen.dart';
import 'package:smart_safety_app/screens/map_screen.dart';
import 'package:smart_safety_app/screens/panic_screen.dart';
import 'package:smart_safety_app/screens/profile_screen.dart';
import 'package:smart_safety_app/screens/settings_screen.dart';
import 'package:smart_safety_app/screens/privacy_screen.dart';
import 'package:smart_safety_app/screens/help_screen.dart';
import 'package:smart_safety_app/screens/end_trip_screen.dart';
import 'package:smart_safety_app/screens/deviation_alert_screen.dart';
import 'package:smart_safety_app/screens/location_status_screen.dart';
import 'package:smart_safety_app/screens/safety_stats_screen.dart';
import 'package:smart_safety_app/screens/emergency_contacts_screen.dart';
import 'package:smart_safety_app/screens/safety_tips_screen.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  await NotificationService().initialize();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  final Color primaryColor = const Color(0xFF2563EB);
  final Color backgroundColor = const Color(0xFFF9FAFB);
  final Color textColor = const Color(0xFF1F2937);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      navigatorKey: navigatorKey,
      title: 'Smart Safety App',
      theme: ThemeData(
        scaffoldBackgroundColor: backgroundColor,
        primaryColor: primaryColor,
        textTheme: GoogleFonts.interTextTheme().apply(
          bodyColor: textColor,
          displayColor: textColor,
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: primaryColor,
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            textStyle: GoogleFonts.inter(fontWeight: FontWeight.w600),
          ),
        ),
      ),
      initialRoute: '/',
      routes: {
        // MODIFIED: Removed 'const' from all routes
        '/': (context) => SplashScreen(),
        '/language': (context) => LanguageScreen(),
        '/login': (context) => LoginScreen(),
        '/signup': (context) => SignupScreen(),
        '/terms': (context) => TermsScreen(),
        '/instructions': (context) => InstructionsScreen(),
        '/dashboard': (context) => DashboardScreen(),
        '/itinerary': (context) => ItineraryScreen(),
        '/map': (context) => MapScreen(),
        '/panic': (context) => PanicScreen(),
        '/settings': (context) => SettingsScreen(),
        '/privacy': (context) => PrivacyScreen(),
        '/profile': (context) => ProfileScreen(),
        '/help': (context) => HelpScreen(),
        '/end': (context) => EndTripScreen(),
        '/deviation': (context) => DeviationAlertScreen(),
        '/location': (context) => LocationStatusScreen(),
        '/stats': (context) => SafetyStatsScreen(),
        '/contacts': (context) => EmergencyContactsScreen(),
        '/tips': (context) => SafetyTipsScreen(),
        '/start': (context) => MapScreen(),
      },
      debugShowCheckedModeBanner: false,
    );
  }
}
