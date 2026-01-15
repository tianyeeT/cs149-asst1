#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <mutex>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;  // for centroids
  int start_m, end_m;  // for data points

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int *counts;  // global counts
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  double *minDist = new double[args->end_m - args->start_m];
  
  // Initialize arrays
  for (int mm = args->start_m; mm < args->end_m; mm++) {
    int m = mm - args->start_m;
    minDist[m] = 1e30;
    args->clusterAssignments[mm] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = 0; k < args->K; k++) {
    for (int mm = args->start_m; mm < args->end_m; mm++) {
      double d = dist(&args->data[mm * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[mm - args->start_m]) {
        minDist[mm - args->start_m] = d;
        args->clusterAssignments[mm] = k;
      }
    }
  }

  delete[] minDist;
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  // Use local arrays to avoid race
  double *localCentroids = new double[args->K * args->N];
  int *localCounts = new int[args->K];

  for (int k = 0; k < args->K; k++) {
    localCounts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      localCentroids[k * args->N + n] = 0.0;
    }
  }

  // Sum up contributions from assigned examples
  for (int mm = args->start_m; mm < args->end_m; mm++) {
    int k = args->clusterAssignments[mm];
    for (int n = 0; n < args->N; n++) {
      localCentroids[k * args->N + n] +=
          args->data[mm * args->N + n];
    }
    localCounts[k]++;
  }

  // Now add to global with lock
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  for (int k = 0; k < args->K; k++) {
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] += localCentroids[k * args->N + n];
    }
    args->counts[k] += localCounts[k];
  }

  delete[] localCentroids;
  delete[] localCounts;
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int mm = args->start_m; mm < args->end_m; mm++) {
    int k = args->clusterAssignments[mm];
    accum[k] += dist(&args->data[mm * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs with lock
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  for (int k = 0; k < args->K; k++) {
    args->currCost[k] += accum[k];
  }

  delete[] accum;
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  const int numThreads = 4;  // Hardcoded for now

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];
  int *counts = new int[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args[numThreads];
  std::thread workers[numThreads];

  for (int i = 0; i < numThreads; i++) {
    args[i].data = data;
    args[i].clusterCentroids = clusterCentroids;
    args[i].clusterAssignments = clusterAssignments;
    args[i].currCost = currCost;
    args[i].counts = counts;
    args[i].M = M;
    args[i].N = N;
    args[i].K = K;

    int chunk = M / numThreads;
    args[i].start_m = i * chunk;
    args[i].end_m = (i == numThreads - 1) ? M : (i + 1) * chunk;
  }

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
    counts[k] = 0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
      currCost[k] = 0.0;
      counts[k] = 0;
      for (int n = 0; n < N; n++) {
        clusterCentroids[k * N + n] = 0.0;
      }
    }

    // Parallel computeAssignments
    for (int i = 1; i < numThreads; i++) {
      workers[i] = std::thread(computeAssignments, &args[i]);
    }
    computeAssignments(&args[0]);
    for (int i = 1; i < numThreads; i++) {
      workers[i].join();
    }

    // Parallel computeCentroids
    for (int i = 1; i < numThreads; i++) {
      workers[i] = std::thread(computeCentroids, &args[i]);
    }
    computeCentroids(&args[0]);
    for (int i = 1; i < numThreads; i++) {
      workers[i].join();
    }

    // Compute means serially
    for (int k = 0; k < K; k++) {
      counts[k] = max(counts[k], 1);
      for (int n = 0; n < N; n++) {
        clusterCentroids[k * N + n] /= counts[k];
      }
    }

    // Parallel computeCost
    for (int i = 1; i < numThreads; i++) {
      workers[i] = std::thread(computeCost, &args[i]);
    }
    computeCost(&args[0]);
    for (int i = 1; i < numThreads; i++) {
      workers[i].join();
    }

    iter++;
  }

  delete[] currCost;
  delete[] prevCost;
  delete[] counts;
}
