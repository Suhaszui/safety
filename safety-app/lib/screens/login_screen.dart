import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController email = TextEditingController();
  final TextEditingController password = TextEditingController();

  late AnimationController _controller;
  late Animation<double> _fadeIn;

  @override
  void initState() {
    super.initState();
    _controller =
        AnimationController(vsync: this, duration: Duration(milliseconds: 800));
    _fadeIn = CurvedAnimation(parent: _controller, curve: Curves.easeIn);
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Widget buildTextField(String label, TextEditingController controller,
      {bool obscure = false}) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      style: GoogleFonts.inter(),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: GoogleFonts.inter(),
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  Future<void> saveUserSession(int userId, String name) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('user_id', userId);
    await prefs.setString('user_name', name);
  }

  void login() async {
    // Simulated login — replace with real API call
    final userId = 1;
    final userName = 'Manisha'; // Replace with actual name from backend

    await saveUserSession(userId, userName);
    Navigator.pushReplacementNamed(context, '/terms'); // ✅ Navigate to Terms
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF2563EB), Color(0xFF1E3A8A)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: FadeTransition(
          opacity: _fadeIn,
          child: Center(
            child: SingleChildScrollView(
              child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: 500),
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      Icon(Icons.shield_rounded, size: 60, color: Colors.white),
                      SizedBox(height: 12),
                      Text('Welcome Back',
                          style: GoogleFonts.inter(
                              fontSize: 26,
                              fontWeight: FontWeight.bold,
                              color: Colors.white)),
                      SizedBox(height: 4),
                      Text('Login to continue your journey',
                          style: GoogleFonts.inter(
                              fontSize: 14, color: Colors.white70)),
                      SizedBox(height: 32),
                      buildTextField('Email', email),
                      SizedBox(height: 16),
                      buildTextField('Password', password, obscure: true),
                      SizedBox(height: 24),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.white,
                            foregroundColor: Color(0xFF2563EB),
                            padding: EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                          ),
                          onPressed: login,
                          child: Text('LOGIN',
                              style: GoogleFonts.inter(
                                  fontWeight: FontWeight.bold)),
                        ),
                      ),
                      SizedBox(height: 12),
                      TextButton(
                        onPressed: () {
                          Navigator.pushNamed(context, '/signup');
                        },
                        child: Text('Don’t have an account? Sign up',
                            style: GoogleFonts.inter(color: Colors.white)),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
