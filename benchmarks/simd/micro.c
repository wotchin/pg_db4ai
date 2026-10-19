/* SPDX-License-Identifier: MulanPSL-2.0
 * Warm-resident microbenchmarks. Includes dispatch call; no PG overhead.
 * Exact same binary/compiler/host: off, portable and auto randomized per trial.
 */
#define _POSIX_C_SOURCE 200809L
#include "numeric.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static volatile double sink;
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double bench(const Db4aiNumericOps *ops,int op,int n,int count,double *a,double *b,double *mean,double *out,double *w)
{
    int i;double start=now(),s=0;
    for(i=0;i<count;i++) {
        switch(op) {
        case 0: s+=ops->dot(a,b,n);break;
        case 1: ops->axpy(out,a,1e-9,n);break;
        case 2: s+=ops->distance(a,b,n,2);break;
        case 3: s+=ops->centered_dot(a,mean,b,n);break;
        case 4: ops->gaussian(out,a,w,mean,32,n);break;
        }
    }
    sink=s+out[0];return now()-start;
}
int main(void)
{
    int widths[]={8,31,65,129,256,1024};
    const char *names[]={"dot","axpy","l2_squared","centered_dot","gaussian_linear32"};
    const char *modes[]={"off","portable","auto"};
    int wi,op,t,i,mode;
    unsigned rng=271828;
    printf("kernel,width,mode,selected_backend,trial,iterations,ns_per_call\n");
    for(wi=0;wi<6;wi++) for(op=0;op<5;op++) {
        int n=widths[wi],count=1000;
        double *aa=calloc(1056,sizeof(double)),*bb=calloc(1056,sizeof(double));
        double *mm=calloc(1056,sizeof(double)),*oo=calloc(1056,sizeof(double));
        double *w=calloc((size_t)32*n,sizeof(double));
        double *a=aa+1,*b=bb+1,*mean=mm+1,*out=oo+1; /* deliberately not 32B-aligned */
        if(!aa||!bb||!mm||!oo||!w)abort();
        for(i=0;i<1024;i++){a[i]=sin(i*.017);b[i]=cos(i*.071);mean[i]=i*.003;}
        for(i=0;i<32*n;i++)w[i]=sin(i*.0017);
        while(bench(db4ai_numeric_choose(0),op,n,count,a,b,mean,out,w)<.02 && count<16000000)count*=2;
        for(t=-1;t<9;t++) {
            int order[]={0,1,2};
            for(i=2;i>0;i--){int j,tmp;rng=rng*1664525U+1013904223U;j=rng%(i+1);tmp=order[i];order[i]=order[j];order[j]=tmp;}
            for(i=0;i<3;i++) {
                double elapsed;const Db4aiNumericOps *ops;
                mode=order[i];ops=db4ai_numeric_choose(mode);
                elapsed=bench(ops,op,n,count,a,b,mean,out,w);
                if(t>=0)printf("%s,%d,%s,%s,%d,%d,%.6f\n",names[op],n,modes[mode],ops->name,t,count,elapsed*1e9/count);
            }
        }
        free(aa);free(bb);free(mm);free(oo);free(w);
    }
    return 0;
}
