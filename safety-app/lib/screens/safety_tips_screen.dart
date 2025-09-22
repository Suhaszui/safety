import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SafetyTipsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFCBD5E1),
      appBar: AppBar(title: Text('Safety Tips')),
      body: Center(
        child: Text('Tip: Always share your itinerary.',
            style: GoogleFonts.inter(fontSize: 16, color: Color(0xFF1F2937))),
      ),
    );
  }
}
