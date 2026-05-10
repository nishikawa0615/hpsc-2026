#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }

  int *d_key;
  int *d_bucket;
  cudaMallocManaged(&d_key, n * sizeof(int));
  cudaMallocManaged(&d_bucket, range * sizeof(int));

  for (int i=0; i<n; i++) {
    d_key[i] = key[i];
  }
  for (int i=0; i<range; i++) {
    d_bucket[i] = bucket[i];
  }
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  count_bucket<<<blocks, threads>>>(d_key, d_bucket, n);
  cudaDeviceSynchronize();

  for (int i=0; i<range; i++) {
    bucket[i] = d_bucket[i];
  }
  cudaFree(d_key);
  cudaFree(d_bucket);

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
