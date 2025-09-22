import 'package:flutter/material.dart';

class HelpScreen extends StatelessWidget {
  const HelpScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final List<Map<String, String>> helpOptions = [
      {'title': 'Lost Phone', 'desc': 'Report and lock your account remotely'},
      {
        'title': 'Emergency Contact',
        'desc': 'Call or message your saved contacts'
      },
      {'title': 'Local Police', 'desc': 'Connect to nearest police station'},
      {'title': 'Tourist Helpline', 'desc': 'Get travel and safety assistance'},
    ];

    return Scaffold(
      appBar: AppBar(title: Text('Help & Support')),
      body: ListView(
        padding: EdgeInsets.all(20),
        children: helpOptions
            .map((item) => Card(
                  child: ListTile(
                    title: Text(item['title']!),
                    subtitle: Text(item['desc']!),
                    trailing: Icon(Icons.arrow_forward),
                    onTap: () {
                      // TODO: Trigger action or call
                    },
                  ),
                ))
            .toList(),
      ),
    );
  }
}
