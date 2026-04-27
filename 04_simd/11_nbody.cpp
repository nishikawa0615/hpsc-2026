#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 fxvec = _mm512_setzero_ps();
    __m512 fyvec = _mm512_setzero_ps();

    for(int j=0; j<N; j+=16) {
      __m512 xj = _mm512_loadu_ps(&x[j]);
      __m512 yj = _mm512_loadu_ps(&y[j]);
      __m512 mj = _mm512_loadu_ps(&m[j]);

      __m512 rx = _mm512_sub_ps(xi, xj);
      __m512 ry = _mm512_sub_ps(yi, yj);

      __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx),
                                _mm512_mul_ps(ry, ry));

      __m512 rinv = _mm512_rsqrt14_ps(r2);

      __m512 rinv3 = _mm512_mul_ps(_mm512_mul_ps(rinv, rinv), rinv);

      __m512 coef = _mm512_mul_ps(mj, rinv3);

      __m512 dfx = _mm512_mul_ps(rx, coef);
      __m512 dfy = _mm512_mul_ps(ry, coef);

      __m512i jidx = _mm512_setr_epi32(j+0, j+1, j+2, j+3,
                                       j+4, j+5, j+6, j+7,
                                       j+8, j+9, j+10, j+11,
                                       j+12, j+13, j+14, j+15);
      
      __m512i ivec = _mm512_set1_epi32(i);

      __mmask16 mask = _mm512_cmpneq_epi32_mask(ivec, jidx);

      fxvec = _mm512_mask_sub_ps(fxvec, mask, fxvec, dfx);
      fyvec = _mm512_mask_sub_ps(fyvec, mask, fyvec, dfy);
    }

    fx[i] = _mm512_reduce_add_ps(fxvec);
    fy[i] = _mm512_reduce_add_ps(fyvec);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
