import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class TermsScreen extends StatefulWidget {
  @override
  _TermsScreenState createState() => _TermsScreenState();
}

class _TermsScreenState extends State<TermsScreen>
    with SingleTickerProviderStateMixin {
  bool accepted = false;
  late AnimationController _controller;
  late Animation<double> _fadeIn;

  final List<String> terms = [
    'You agree to share your location for emergency tracking.',
    'You will use the panic alert feature responsibly.',
    'You acknowledge that Smart Safety is not a substitute for local law enforcement.',
    'You will keep your itinerary updated for accurate monitoring.',
    'You agree to receive safety notifications and alerts.',
    'You understand that data is stored securely and used only for safety purposes.',
  ];

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: Duration(milliseconds: 1000));
    _fadeIn = CurvedAnimation(parent: _controller, curve: Curves.easeIn);
    _controller.forward();
  }

  void showConfirmationPopup() {
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
              Icon(Icons.verified_user, size: 40, color: Color(0xFF9DC183)),
              SizedBox(height: 16),
              Text('Terms Accepted',
                  style: GoogleFonts.inter(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.black)),
              SizedBox(height: 12),
              Text(
                'Thanks for committing to safety and responsibility.',
                style: GoogleFonts.inter(color: Colors.black87),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 8),
              Text(
                'Let’s continue building a safer journey together.',
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
                  Navigator.pushNamed(context, '/instructions');
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
      backgroundColor: Color(0xFFCBD5E1), // Slightly darker slate background
      body: FadeTransition(
        opacity: _fadeIn,
        child: Center(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: 600),
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                children: [
                  Text(
                    'Terms & Conditions',
                    style: GoogleFonts.inter(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1F2937),
                    ),
                  ),
                  SizedBox(height: 12),
                  Text(
                    'Please review and accept our safety guidelines.',
                    style: GoogleFonts.inter(
                      fontSize: 14,
                      color: Color(0xFF6B7280),
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  SizedBox(height: 20),
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black12,
                            blurRadius: 12,
                            offset: Offset(0, 6),
                          ),
                        ],
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: ListView.separated(
                          itemCount: terms.length,
                          separatorBuilder: (_, __) =>
                              Divider(height: 24, color: Colors.grey[300]),
                          itemBuilder: (context, index) {
                            return Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Icon(Icons.check_circle_outline,
                                    color: Color(0xFF9DC183)),
                                SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    terms[index],
                                    style: GoogleFonts.inter(
                                        fontSize: 15,
                                        color: Color(0xFF4B5563),
                                        height: 1.4),
                                  ),
                                ),
                              ],
                            );
                          },
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: 20),
                  Row(
                    children: [
                      Checkbox(
                        value: accepted,
                        activeColor: Color(0xFF9DC183),
                        onChanged: (value) =>
                            setState(() => accepted = value ?? false),
                      ),
                      Expanded(
                        child: Text(
                          'I accept the Terms & Conditions',
                          style: GoogleFonts.inter(
                              fontSize: 14, color: Color(0xFF374151)),
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor:
                            accepted ? Color(0xFF2563EB) : Colors.grey[300],
                        foregroundColor: Colors.white,
                        padding:
                            EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                      ),
                      onPressed: accepted ? showConfirmationPopup : null,
                      child: Text('CONTINUE',
                          style: GoogleFonts.inter(
                              fontSize: 16, fontWeight: FontWeight.w600)),
                    ),
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
