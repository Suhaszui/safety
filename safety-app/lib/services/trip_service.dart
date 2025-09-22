import 'dart:convert';
import 'package:http/http.dart' as http;

class TripService {
  TripService._(); // private constructor
  static final TripService instance = TripService._();

  final String baseUrl = 'http://localhost:3000'; // update if deployed

  Future<List<Map<String, dynamic>>> getTrips({
    required int userId,
    int page = 1,
    int limit = 20,
    bool forceRefresh = false,
  }) async {
    final uri =
        Uri.parse('$baseUrl/trips?user_id=$userId&page=$page&limit=$limit');
    final response = await http.get(uri);

    if (response.statusCode == 200) {
      final List data = jsonDecode(response.body);
      return List<Map<String, dynamic>>.from(data);
    } else {
      throw Exception('Failed to load trips: ${response.statusCode}');
    }
  }

  Future<bool> addTrip(Map<String, dynamic> trip) async {
    final uri = Uri.parse('http://localhost:3000/trips');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(trip),
    );
    return response.statusCode == 201;
  }

  Future<bool> updateTrip(int id, Map<String, dynamic> trip) async {
    final uri = Uri.parse('$baseUrl/trips/$id');
    final response = await http.put(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(trip),
    );
    return response.statusCode == 200;
  }

  Future<bool> deleteTrip(int id) async {
    final uri = Uri.parse('$baseUrl/trips/$id');
    final response = await http.delete(uri);
    return response.statusCode == 200;
  }
}
