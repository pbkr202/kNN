#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include<random>
#include<vector>
#include<cuda_runtime.h>
#include<sm_60_atomic_functions.h>
#include<thrust/device_vector.h>
#include<thrust/sort.h>

using namespace std;

#define BLOCK_SIZE 1024

struct DataPoint {
    double x;
    double y;
    int label;
};

// Compute the Euclidean distance between two points
__device__ double euclidean_distance(DataPoint a, DataPoint b) {
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
    for (int i = 0; i < 1000000; i++) {
        DataPoint p = {dist1x(gen), dist1y(gen), 0};
        data.push_back(p);
    }

    // Cluster 2: mean (5,5), std dev 0.5
    normal_distribution<double> dist2x(5, 0.5);
    normal_distribution<double> dist2y(5, 0.5);
    for (int i = 0; i < 1000000; i++) {
        DataPoint p = {dist2x(gen), dist2y(gen), 1};
        data.push_back(p);
    }

    // Cluster 3: mean (1,5), std dev 0.5
    normal_distribution<double> dist3x(1, 0.5);
    normal_distribution<double> dist3y(5, 0.5);
    for (int i = 0; i < 1000000; i++) {
        DataPoint p = {dist3x(gen), dist3y(gen), 2};
        data.push_back(p);
    }
    
    // Cluster 4: mean (8,6), std dev 0.5
    normal_distribution<double> dist4x(8, 0.5);
    normal_distribution<double> dist4y(6, 0.5);
    for (int i = 0; i < 1000000; i++) {
        DataPoint p = {dist4x(gen), dist4y(gen), 3};
        data.push_back(p);
    }
    
    // Cluster 3: mean (9,2), std dev 0.5
    normal_distribution<double> dist5x(9, 0.5);
    normal_distribution<double> dist5y(2, 0.5);
    for (int i = 0; i < 1000000; i++) {
        DataPoint p = {dist5x(gen), dist5y(gen), 4};
        data.push_back(p);
    }
    return data;
    }

__global__ void classify_knn(DataPoint* training_data, int num_training, DataPoint new_point, int k, int* votes, int num_clusters,int *cluster_votes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_training) {
        // Compute distance between new point and training data point
        double dist = euclidean_distance(training_data[tid], new_point);

        // Store the distance and label of the training data point in shared memory
        __shared__ double distances[BLOCK_SIZE];
        __shared__ int labels[BLOCK_SIZE];
        distances[threadIdx.x] = dist;
        labels[threadIdx.x] = training_data[tid].label;

        // Synchronize threads before sorting
        __syncthreads();

        // Sort the distances and labels in shared memory
        thrust::sort(thrust::seq, distances, distances + blockDim.x);
        thrust::sort(thrust::seq, labels, labels + blockDim.x);

        // Count the votes for the K nearest neighbors
        for (int i = 0; i < k; i++) {
            atomicAdd(&cluster_votes[labels[i]], 1);
        }

        // Add votes to global memory
        for (int i = 0; i < num_clusters; i++) {
            atomicAdd(&votes[i], cluster_votes[i]);
        }
    }

    __syncthreads();

    if (tid == 0) {
        // Find the cluster with the most votes
        int max_votes = votes[0];
        int max_cluster = 0;
        for (int i = 1; i < num_clusters; i++) {
            if (votes[i] > max_votes) {
                max_votes = votes[i];
                max_cluster = i;
            }
        }
        printf("Cluster of new point is: %d\n", max_cluster);
    }
}



int main() {
    // Define some training data 
    vector<DataPoint> training_data = generate_data():

    // Classify a new data point using KNN with k=2
    DataPoint new_point = {4, 4, -1}; // the label of the new point is unknown (-1)
    int num_clusters = 5;
    int k = 2;

    // Copy training data to device memory
    int num_training = training_data.size();
    DataPoint* d_training_data;
    cudaMalloc((void**)&d_training_data, num_training * sizeof(DataPoint));
    cudaMemcpy(d_training_data, &training_data[0], num_training * sizeof(DataPoint), cudaMemcpyHostToDevice);

    // Allocate and initialize votes vector in device memory
    int* d_votes;
    cudaMalloc((void**)&d_votes, num_clusters * sizeof(int));
    cudaMemset(d_votes, 0, num_clusters * sizeof(int));

    int* cluster_votes;
    cudaMalloc((void**)&cluster_votes, num_clusters * sizeof(int));
    cudaMemset(cluster_votes, 0, num_clusters * sizeof(int));
    // Launch kernel to classify new point
    classify_knn<<<(num_training + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_training_data, num_training, new_point, k, d_votes, num_clusters,cluster_votes);

    // Free device memory
    cudaFree(d_training_data);
    cudaFree(d_votes);
    cudaFree(cluster_votes);

    return 0;
}

