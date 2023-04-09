#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <random>
#include<time.h>

using namespace std;

struct DataPoint {
    double x;
    double y;
    int cluster;
};

// Compute the Euclidean distance between two points
double euclidean_distance(DataPoint a, DataPoint b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

vector<DataPoint> generate_data() {
    vector<DataPoint> data;

    // Set up random number generator
    random_device rd;
    mt19937 gen(rd());

    // Cluster 1: mean (1,1), std dev 0.5
    normal_distribution<double> dist1x(1, 0.5);
    normal_distribution<double> dist1y(1, 0.5);
    for (int i = 0; i < 10; i++) {
        DataPoint p = {dist1x(gen), dist1y(gen), 0};
        data.push_back(p);
    }

    // Cluster 2: mean (5,5), std dev 0.5
    normal_distribution<double> dist2x(5, 0.5);
    normal_distribution<double> dist2y(5, 0.5);
    for (int i = 0; i < 10; i++) {
        DataPoint p = {dist2x(gen), dist2y(gen), 1};
        data.push_back(p);
    }

    // Cluster 3: mean (1,5), std dev 0.5
    normal_distribution<double> dist3x(1, 0.5);
    normal_distribution<double> dist3y(5, 0.5);
    for (int i = 0; i < 10; i++) {
        DataPoint p = {dist3x(gen), dist3y(gen), 2};
        data.push_back(p);
    }
    
    // Cluster 4: mean (8,6), std dev 0.5
    normal_distribution<double> dist4x(8, 0.5);
    normal_distribution<double> dist4y(6, 0.5);
    for (int i = 0; i < 10; i++) {
        DataPoint p = {dist4x(gen), dist4y(gen), 3};
        data.push_back(p);
    }
    
    // Cluster 3: mean (9,2), std dev 0.5
    normal_distribution<double> dist5x(9, 0.5);
    normal_distribution<double> dist5y(2, 0.5);
    for (int i = 0; i < 10; i++) {
        DataPoint p = {dist5x(gen), dist5y(gen), 4};
        data.push_back(p);
    }
    return data;
    }

// Classify a new point based on its K nearest neighbors
int classify_knn(const vector<DataPoint>& training_data, DataPoint new_point, int k, int num_clusters) {
    // Compute distances between new point and all training data points
    vector<pair<double, int>> distances;
    for (int i = 0; i < training_data.size(); i++) {
        double dist = euclidean_distance(training_data[i], new_point);
        distances.push_back({dist, training_data[i].cluster});
    }

    // Sort distances in ascending order
    sort(distances.begin(), distances.end());

    // Count votes for each cluster among the k-nearest neighbors
    vector<int> votes(num_clusters, 0);
    for (int i = 0; i < k; i++) {
        votes[distances[i].second]++;
    }

    // Return cluster with the most votes
    int max_votes = votes[0];
    int max_cluster = 0;
    for (int i = 1; i < num_clusters; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            max_cluster = i;
        }
    }
    return max_cluster;
}

int main() {
       clock_t t;
        t=clock();
	srand(time(0));
    vector<DataPoint> training_data=generate_data();
    // Classify a new data point using KNN with k=3
    DataPoint new_point = {4, 4, -1}; // the cluster of the new point is unknown (-1)
    int num_clusters = 5;
    int cluster = classify_knn(training_data, new_point, 3, num_clusters);

   // cout << "Cluster of new point is: " << cluster << endl;
t=clock()-t;
	double time=((double)t)/CLOCKS_PER_SEC;
	cout<<time<<endl;
	return 0;
    return 0;
}

