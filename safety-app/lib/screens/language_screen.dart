import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LanguageScreen extends StatefulWidget {
  const LanguageScreen({super.key});

  @override
  State<LanguageScreen> createState() => _LanguageScreenState();
}

class _LanguageScreenState extends State<LanguageScreen>
    with SingleTickerProviderStateMixin {
  String selectedLanguage = '';
  late AnimationController _controller;
  late Animation<double> _fadeIn;

  final List<String> languages = [
    'English',
    'हिन्दी',
    'తెలుగు',
    'தமிழ்',
    'বাংলা',
    'ಕನ್ನಡ',
    'മലയാളം',
    'ગુજરાતી',
    'मराठी',
    'ਪੰਜਾਬੀ',
    'اردو',
  ];

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: Duration(milliseconds: 1000));
    _fadeIn = CurvedAnimation(parent: _controller, curve: Curves.easeIn);
    _controller.forward();
  }

  Future<void> saveLanguage(String language) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('selected_language', language);
  }

  void showConfirmationDialog(String language) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.check_circle_outline,
                  size: 40, color: Color(0xFF9DC183)),
              SizedBox(height: 16),
              Text('Language Selected',
                  style: GoogleFonts.inter(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.black)),
              SizedBox(height: 12),
              Text(
                'You’ve chosen "$language". Great choice!',
                style: GoogleFonts.inter(color: Colors.black87),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 8),
              Text(
                'Let’s make your journey safer and smarter.',
                style: GoogleFonts.inter(
                    fontSize: 14,
                    color: Colors.black54,
                    fontStyle: FontStyle.italic),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 20),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF2563EB),
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                ),
                onPressed: () {
                  Navigator.pop(context);
                  Navigator.pushReplacementNamed(context, '/login');
                },
                child: Text('Continue', style: GoogleFonts.inter(fontSize: 16)),
              ),
            ],
          ),
        ),
      ),
    );
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
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.language, size: 60, color: Colors.white),
                  SizedBox(height: 20),
                  Text(
                    'Choose Your Language',
                    style: GoogleFonts.inter(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Your journey starts in your language.',
                    style: GoogleFonts.inter(
                      fontSize: 14,
                      color: Colors.white70,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  SizedBox(height: 30),
                  Wrap(
                    spacing: 16,
                    runSpacing: 16,
                    children: languages.map((lang) {
                      final isSelected = selectedLanguage == lang;
                      return GestureDetector(
                        onTap: () {
                          setState(() => selectedLanguage = lang);
                          saveLanguage(lang);
                          Future.delayed(Duration(milliseconds: 300), () {
                            showConfirmationDialog(lang);
                          });
                        },
                        child: AnimatedContainer(
                          duration: Duration(milliseconds: 300),
                          width: 140,
                          height: 60,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                            color: isSelected
                                ? Color(0xFF9DC183) // Sage green
                                : Colors.white,
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black12,
                                blurRadius: 6,
                                offset: Offset(0, 3),
                              ),
                            ],
                            border: Border.all(
                              color: isSelected
                                  ? Color(0xFF9DC183)
                                  : Colors.transparent,
                              width: 2,
                            ),
                          ),
                          child: Text(
                            lang,
                            style: GoogleFonts.inter(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                              color: Colors.black,
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                  SizedBox(height: 40),
                  Text(
                    '“Preparedness is the ultimate confidence.”',
                    style: GoogleFonts.inter(
                      fontSize: 13,
                      color: Colors.white60,
                      fontStyle: FontStyle.italic,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
