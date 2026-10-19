/* SPDX-License-Identifier: MulanPSL-2.0
 * Standalone mathematical, alias, misalignment and exact-tail tests.
 * POSIX guard pages make even non-ASan vector overreads deterministic failures.
 */
#define _GNU_SOURCE
#include "numeric.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
static size_t checks;
static void close_to(double got, long double ref, long double magnitude, int n)
{
    long double tol=64*DBL_EPSILON*(n+1)*(magnitude+1);
    if (!isfinite(got) || fabsl((long double)got-ref)>tol) {
        fprintf(stderr,"mismatch n=%d got=%.17g expected=%.21Lg tolerance=%Lg\n",n,got,ref,tol);
        abort();
    }
    checks++;
}
int main(void)
{
    const int sizes[]={0,1,2,3,4,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,1024};
    int mode,si,offset,i;
    long pagesize=sysconf(_SC_PAGESIZE);
    assert(pagesize>0);
    for(mode=0;mode<3;mode++) {
        const Db4aiNumericOps *ops=db4ai_numeric_choose(mode);
        printf("mode=%d selected=%s\n",mode,ops->name);
        for(si=0;si<(int)(sizeof(sizes)/sizeof(sizes[0]));si++) for(offset=0;offset<4;offset++) {
            int n=sizes[si];
            double *aa=calloc((size_t)n+4,sizeof(double)),*bb=calloc((size_t)n+4,sizeof(double));
            double *mm=calloc((size_t)n+4,sizeof(double)),*yy=calloc((size_t)n+4,sizeof(double));
            double *a=aa+offset,*b=bb+offset,*mean=mm+offset,*y=yy+offset;
            long double dot=0,absdot=0,center=0,abscenter=0,dist[4]={0,0,0,0},norm=0;
            assert(aa&&bb&&mm&&yy);
            for(i=0;i<n;i++) {
                a[i]=sin(i*.71)*pow(10.0,(i%9)-4);b[i]=cos(i*.49)*pow(10.0,4-(i%9));mean[i]=i*.003;
                y[i]=b[i];dot+=(long double)a[i]*b[i];absdot+=fabsl((long double)a[i]*b[i]);
                center+=((long double)a[i]-mean[i])*b[i];abscenter+=fabsl(((long double)a[i]-mean[i])*b[i]);
                norm+=(long double)a[i]*a[i];
                {long double d=fabsl((long double)a[i]-b[i]);dist[0]+=d;dist[2]+=d*d;if(d>dist[3])dist[3]=d;}
            }
            dist[1]=sqrtl(dist[2]);
            close_to(ops->dot(a,b,n),dot,absdot,n);
            close_to(ops->dot(a,a,n),norm,norm,n); /* read-only self-alias legal */
            close_to(ops->centered_dot(a,mean,b,n),center,abscenter,n);
            for(i=0;i<4;i++) close_to(ops->distance(a,b,n,i),dist[i],dist[i],n);
            ops->axpy(y,a,.123,n);
            for(i=0;i<n;i++) {assert(y[i]==b[i]+.123*a[i]);checks++;}
            free(aa);free(bb);free(mm);free(yy);
            /* End exactly at a protected page boundary, including odd tails. */
            if(n>0 && (size_t)n*sizeof(double)<=(size_t)pagesize) {
                char *map=mmap(NULL,(size_t)pagesize*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
                double *tail;
                assert(map!=MAP_FAILED);assert(mprotect(map+pagesize,pagesize,PROT_NONE)==0);
                tail=(double *)(map+pagesize)-(size_t)n;
                for(i=0;i<n;i++)tail[i]=(i+1)*.125;
                close_to(ops->dot(tail,tail,n),db4ai_numeric_scalar.dot(tail,tail,n),1e12L,n);
                close_to(ops->distance(tail,tail,n,2),0,0,n);
                assert(munmap(map,(size_t)pagesize*2)==0);
            }
        }
        { int n=33,k=129,c;double x[33],offsets[129],expected[129],out[129];
          double *w=malloc((size_t)n*k*sizeof(double));assert(w);
          for(i=0;i<n;i++)x[i]=sin(i*.17);
          for(c=0;c<k;c++)offsets[c]=c*.07;
          for(i=0;i<n*k;i++)w[i]=cos(i*.21);
          db4ai_numeric_scalar.gaussian(expected,x,w,offsets,n,k);
          ops->gaussian(out,x,w,offsets,n,k);
          for(c=0;c<k;c++){assert(out[c]==expected[c]);checks++;}
          free(w);
        }
        {double a[32],b[32];for(i=0;i<32;i++){a[i]=1;b[i]=1;}
         a[15]=NAN;assert(isnan(ops->dot(a,b,32)));checks++;
         a[15]=INFINITY;assert(isinf(ops->dot(a,b,32)));checks++;
         a[15]=DBL_MAX;b[15]=2;assert(isinf(ops->dot(a,b,32)));checks++;
         b[15]=-DBL_MAX;assert(isinf(ops->distance(a,b,32,2)));checks++;
        }
    }
    printf("PASS %zu numeric checks (including exact Gaussian and AXPY, guard pages and NaN/Inf).\n",checks);
    return 0;
}
