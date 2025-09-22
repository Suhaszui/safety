import 'package:flutter/material.dart';

class DeviationAlertScreen extends StatelessWidget {
  const DeviationAlertScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // This line safely extracts the data sent with the push notification.
    // If no data is sent, it uses a default map to prevent errors.
    final Map<String, dynamic> alertData = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {
      'emoji': '⚠️',
      'tagline': 'Safety Alert',
      'advice': 'General caution is advised.'
    };

    // Use the data to build the UI, with fallback values.
    final String emoji = alertData['emoji'] ?? '⚠️';
    final String tagline = alertData['tagline'] ?? 'Safety Alert';
    final String advice = alertData['advice'] ?? 'Please be cautious.';
    
    // Determine the icon and color based on the alert level or type if available
    final IconData icon = (alertData['level'] != null && int.parse(alertData['level']!) < 2) 
                          ? Icons.error 
                          : Icons.warning;
    final Color iconColor = (alertData['level'] != null && int.parse(alertData['level']!) < 2) 
                          ? Colors.red[700]! 
                          : Colors.orange[700]!;

    return Scaffold(
      appBar: AppBar(
        title: Text(alertData['level_name'] ?? 'Safety Alert'),
        backgroundColor: iconColor,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                emoji,
                style: const TextStyle(fontSize: 80),
              ),
              const SizedBox(height: 24),
              Text(
                tagline,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                advice,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.grey[700],
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 40),
              ElevatedButton(
                onPressed: () {
                  // Simply closes the alert screen and returns to the previous page.
                  Navigator.of(context).pop();
                },
                style: ElevatedButton.styleFrom(
                    backgroundColor: Theme.of(context).primaryColor),
                child: const Text('Acknowledge'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

