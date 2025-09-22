import 'package:flutter/material.dart';

class PrivacyScreen extends StatefulWidget {
  const PrivacyScreen({super.key});

  @override
  State<PrivacyScreen> createState() => _PrivacyScreenState();
}

class _PrivacyScreenState extends State<PrivacyScreen> {
  bool shareLocation = true;
  bool shareTripData = true;
  bool allowEmergencyAccess = true;

  void toggle(String key, bool value) {
    setState(() {
      if (key == 'location') shareLocation = value;
      if (key == 'trip') shareTripData = value;
      if (key == 'emergency') allowEmergencyAccess = value;
    });
  }

  void showConsentDetails(String title, String description) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text(title),
        content: Text(description),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context), child: Text('Close'))
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Privacy & Consent')),
      body: ListView(
        padding: EdgeInsets.all(20),
        children: [
          SwitchListTile(
            title: Text('Share Live Location'),
            subtitle: Text('Used for geo-fencing and emergency alerts'),
            value: shareLocation,
            onChanged: (val) => toggle('location', val),
          ),
          SwitchListTile(
            title: Text('Share Trip Itinerary'),
            subtitle: Text('Used for safety score and route monitoring'),
            value: shareTripData,
            onChanged: (val) => toggle('trip', val),
          ),
          SwitchListTile(
            title: Text('Allow Emergency Access'),
            subtitle: Text('Police can access your data during SOS'),
            value: allowEmergencyAccess,
            onChanged: (val) => toggle('emergency', val),
          ),
          Divider(),
          ListTile(
            title: Text('View Consent Terms'),
            trailing: Icon(Icons.info_outline),
            onTap: () => showConsentDetails(
              'Consent Terms',
              'By enabling these options, you agree to share selected data with safety authorities and emergency responders. You can revoke access anytime from this screen.',
            ),
          ),
          ListTile(
            title: Text('Download Data Policy'),
            trailing: Icon(Icons.download),
            onTap: () {
              // TODO: Trigger download or open PDF
            },
          ),
        ],
      ),
    );
  }
}
