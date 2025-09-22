import 'package:flutter/material.dart';

class SafetyScoreScreen extends StatelessWidget {
  const SafetyScoreScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final int score = 82;
    final Color scoreColor = score >= 80
        ? Colors.green
        : score >= 50
            ? Colors.orange
            : Colors.red;

    final List<Map<String, dynamic>> breakdown = [
      {'label': 'Area Risk Level', 'value': 'Low', 'impact': '+20'},
      {'label': 'Time of Travel', 'value': 'Daytime', 'impact': '+15'},
      {'label': 'Activity Consistency', 'value': 'Stable', 'impact': '+10'},
    ];

    final List<String> tips = [
      'Avoid traveling alone at night',
      'Stay within green zones',
      'Check in regularly at itinerary points',
    ];

    final List<Map<String, dynamic>> history = [
      {'date': 'Sep 18', 'score': 78},
      {'date': 'Sep 19', 'score': 80},
      {'date': 'Sep 20', 'score': 82},
    ];

    return Scaffold(
      appBar: AppBar(title: Text('Safety Score')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: ListView(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text('Current Score: ',
                    style:
                        TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: scoreColor,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text('$score',
                      style: TextStyle(color: Colors.white, fontSize: 16)),
                ),
              ],
            ),
            SizedBox(height: 20),
            Text('Breakdown',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            ...breakdown.map((item) => ListTile(
                  title: Text(item['label']),
                  subtitle: Text(item['value']),
                  trailing: Text(item['impact']),
                )),
            Divider(),
            Text('Tips to Improve',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            ...tips.map((tip) => ListTile(
                  leading: Icon(Icons.lightbulb),
                  title: Text(tip),
                )),
            Divider(),
            Text('Score History',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            ...history.map((entry) => ListTile(
                  title: Text(entry['date']),
                  trailing: Text('${entry['score']}'),
                )),
          ],
        ),
      ),
    );
  }
}
