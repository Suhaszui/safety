import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class EmergencyContactsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFCBD5E1),
      appBar: AppBar(title: Text('Emergency Contacts')),
      body: Center(
        child: Text('You have 3 emergency contacts saved.',
            style: GoogleFonts.inter(fontSize: 16, color: Color(0xFF1F2937))),
      ),
    );
  }
}
