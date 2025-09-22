import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/trip_manager.dart'; // 1. Import your TripManager

// ... inside your widget's build method ...

// 2. Create an instance of the TripManager
final tripManager = TripManager();

class DashboardScreen extends StatefulWidget {
  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeIn;

  final List<Map<String, String>> safetyTips = [
    {
      'title': 'Stay Visible',
      'desc': 'Wear bright clothing when traveling alone.'
    },
    {'title': 'Share Itinerary', 'desc': 'Let someone know your travel plans.'},
    {
      'title': 'Emergency Contacts',
      'desc': 'Keep them updated and accessible.'
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

  Widget buildCard(
      {required IconData icon,
      required String title,
      required String subtitle,
      VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 10),
        padding: EdgeInsets.all(20),
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
        child: Row(
          children: [
            Container(
              decoration: BoxDecoration(
                color: Color(0xFF9DC183).withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              padding: EdgeInsets.all(12),
              child: Icon(icon, color: Color(0xFF9DC183), size: 28),
            ),
            SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: GoogleFonts.inter(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF1F2937))),
                  SizedBox(height: 4),
                  Text(subtitle,
                      style: GoogleFonts.inter(
                          fontSize: 14, color: Color(0xFF6B7280))),
                ],
              ),
            ),
            Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey[400]),
          ],
        ),
      ),
    );
  }

  Widget buildStatCard(String label, String value) {
    return Container(
      margin: EdgeInsets.all(8),
      padding: EdgeInsets.symmetric(vertical: 16, horizontal: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(color: Colors.black12, blurRadius: 8, offset: Offset(0, 4)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: GoogleFonts.inter(fontSize: 13, color: Color(0xFF6B7280))),
          SizedBox(height: 4),
          Text(value,
              style: GoogleFonts.inter(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1F2937))),
        ],
      ),
    );
  }

  Widget buildTipCard(Map<String, String> tip) {
    return Container(
      margin: EdgeInsets.symmetric(horizontal: 4),
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Color(0xFF9DC183).withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(tip['title']!,
              style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1F2937))),
          SizedBox(height: 6),
          Text(tip['desc']!,
              style: GoogleFonts.inter(fontSize: 14, color: Color(0xFF4B5563))),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFCBD5E1),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Color(0xFF9DC183),
        child: Icon(Icons.warning_amber_outlined, color: Colors.white),
        onPressed: () {
          // Trigger panic alert
        },
      ),
      body: FadeTransition(
        opacity: _fadeIn,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: ListView(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('Welcome back, Priya 👋',
                        style: GoogleFonts.inter(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF1F2937))),
                    CircleAvatar(
                      backgroundImage: AssetImage('assets/avatar.png'),
                      radius: 20,
                    ),
                  ],
                ),
                SizedBox(height: 8),
                Text('Here’s your safety dashboard overview.',
                    style: GoogleFonts.inter(
                        fontSize: 14,
                        color: Color(0xFF6B7280),
                        fontStyle: FontStyle.italic)),
                SizedBox(height: 24),
                Text('Quick Stats',
                    style: GoogleFonts.inter(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF6B7280))),
                GridView.count(
                  crossAxisCount: 2,
                  shrinkWrap: true,
                  physics: NeverScrollableScrollPhysics(),
                  childAspectRatio: 3,
                  children: [
                    buildStatCard('Trips Completed', '12'),
                    buildStatCard('Alerts Sent', '3'),
                  ],
                ),
                SizedBox(height: 24),
                Text('Quick Access',
                    style: GoogleFonts.inter(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF6B7280))),
                buildCard(
                  icon: Icons.location_on_outlined,
                  title: 'Location Status',
                  subtitle: 'Tracking is active',
                  onTap: () => Navigator.pushNamed(context, '/location'),
                ),
                buildCard(
                  icon: Icons.bar_chart_outlined,
                  title: 'Safety Stats',
                  subtitle: 'View recent activity',
                  onTap: () => Navigator.pushNamed(context, '/stats'),
                ),
                buildCard(
                  icon: Icons.calendar_today_outlined,
                  title: 'Upcoming Itinerary',
                  subtitle: '2 trips scheduled',
                ),
                buildCard(
                  icon: Icons.play_circle_outline,
                  title: 'Start Trip',
                  subtitle: 'Begin your journey',
                  onTap: () {
                  // 3. Call the startTrip() method
                  print("Start Trip button tapped!");
                  tripManager.startTrip();

                  // 4. Then navigate to the map screen
                  Navigator.pushNamed(context, '/map');
                },
                ),
               buildCard(
                  icon: Icons.stop_circle_outlined,
                  title: 'End Trip',
                  subtitle: 'Finish and log your trip',
                  onTap: () {
                  // 3. Call the stopTrip() method
                  print("End Trip button tapped!");
                  tripManager.stopTrip();

                  // 4. Then navigate to the end trip screen
                  Navigator.pushNamed(context, '/end');
                },
               ),
                buildCard(
                  icon: Icons.contacts_outlined,
                  title: 'Emergency Contacts',
                  subtitle: '3 saved contacts',
                  onTap: () => Navigator.pushNamed(context, '/contacts'),
                ),
                buildCard(
                  icon: Icons.shield_outlined,
                  title: 'Safety Tips',
                  subtitle: 'Stay informed and prepared',
                  onTap: () => Navigator.pushNamed(context, '/tips'),
                ),
                SizedBox(height: 24),
                Text('Safety Tips',
                    style: GoogleFonts.inter(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF6B7280))),
                SizedBox(
                  height: 120,
                  child: PageView.builder(
                    itemCount: safetyTips.length,
                    controller: PageController(viewportFraction: 0.9),
                    itemBuilder: (context, index) =>
                        buildTipCard(safetyTips[index]),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
