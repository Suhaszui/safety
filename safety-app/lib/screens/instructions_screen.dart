import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class InstructionsScreen extends StatefulWidget {
  @override
  _InstructionsScreenState createState() => _InstructionsScreenState();
}

class _InstructionsScreenState extends State<InstructionsScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeIn;

  final List<Map<String, dynamic>> instructions = [
    {
      'text': 'Keep your location services turned on at all times.',
      'icon': Icons.location_on_outlined,
    },
    {
      'text': 'Update your itinerary before traveling to new places.',
      'icon': Icons.edit_calendar_outlined,
    },
    {
      'text': 'Use the panic alert only in genuine emergencies.',
      'icon': Icons.warning_amber_outlined,
    },
    {
      'text': 'Ensure your emergency contacts are up to date.',
      'icon': Icons.contacts_outlined,
    },
    {
      'text': 'Respond to safety notifications promptly.',
      'icon': Icons.notifications_active_outlined,
    },
    {
      'text': 'Review your safety dashboard regularly.',
      'icon': Icons.dashboard_customize_outlined,
    },
  ];

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: Duration(milliseconds: 1000));
    _fadeIn = CurvedAnimation(parent: _controller, curve: Curves.easeIn);
    _controller.forward();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFCBD5E1), // Slightly dark slate background
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
                    'Safety Instructions',
                    style: GoogleFonts.inter(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1F2937),
                    ),
                  ),
                  SizedBox(height: 12),
                  Text(
                    'Follow these steps to ensure your safety is always prioritized.',
                    style: GoogleFonts.inter(
                      fontSize: 14,
                      color: Color(0xFF6B7280),
                      fontStyle: FontStyle.italic,
                    ),
                    textAlign: TextAlign.center,
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
                          itemCount: instructions.length,
                          separatorBuilder: (_, __) =>
                              Divider(height: 24, color: Colors.grey[300]),
                          itemBuilder: (context, index) {
                            final item = instructions[index];
                            return Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Container(
                                  decoration: BoxDecoration(
                                    color: Color(0xFF9DC183).withOpacity(0.2),
                                    shape: BoxShape.circle,
                                  ),
                                  padding: EdgeInsets.all(8),
                                  child: Icon(
                                    item['icon'],
                                    color: Color(0xFF9DC183),
                                    size: 24,
                                  ),
                                ),
                                SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    item['text'],
                                    style: GoogleFonts.inter(
                                      fontSize: 15,
                                      color: Color(0xFF4B5563),
                                      height: 1.4,
                                    ),
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
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Color(0xFF2563EB),
                        foregroundColor: Colors.white,
                        padding:
                            EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                      ),
                      onPressed: () {
                        Navigator.pushNamed(context, '/dashboard');
                      },
                      child: Text('GO TO DASHBOARD',
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
