// lib/screens/splash_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async';

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _logoController;
  late Animation<double> _logoAnimation;

  final List<String> quotes = [
    '“Safety isn’t expensive, it’s priceless.”',
    '“Smart travel begins with smart protection.”',
    '“Preparedness is the ultimate confidence.”',
  ];

  int quoteIndex = 0;
  Timer? quoteTimer;

  @override
  void initState() {
    super.initState();

    // Logo animation setup
    _logoController = AnimationController(
      vsync: this,
      duration: Duration(seconds: 2),
    );
    _logoAnimation = CurvedAnimation(
      parent: _logoController,
      curve: Curves.easeOutBack,
    );
    _logoController.forward();

    // Quote rotation
    quoteTimer = Timer.periodic(Duration(seconds: 2), (timer) {
      setState(() {
        quoteIndex = (quoteIndex + 1) % quotes.length;
      });
    });

    // Navigate after delay
    Future.delayed(Duration(seconds: 5), () {
      Navigator.pushReplacementNamed(context, '/language');
    });
  }

  @override
  void dispose() {
    _logoController.dispose();
    quoteTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF1E3A8A), Color(0xFF2563EB)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ScaleTransition(
                scale: _logoAnimation,
                child: Container(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.white.withOpacity(0.3),
                        blurRadius: 20,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Icon(Icons.shield_rounded,
                      size: 100, color: Colors.white),
                ),
              ),
              SizedBox(height: 20),
              Text(
                'Smart Safety',
                style: GoogleFonts.inter(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  letterSpacing: 1.2,
                ),
              ),
              SizedBox(height: 8),
              AnimatedSwitcher(
                duration: Duration(milliseconds: 600),
                child: SizedBox(
                  key: ValueKey(quoteIndex),
                  width: 300,
                  child: Text(
                    quotes[quoteIndex],
                    style: GoogleFonts.inter(
                      fontSize: 14,
                      color: Colors.white70,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
