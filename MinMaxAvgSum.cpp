#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

int minvalsequential(int arr[], int n)
{
    int minval = arr[0];
    for(int i = 0; i < n;i++){
       if(arr[i] < minval){
        minval = arr[i];
       }
    
    }
    return minval; 

}

int minvalparallel(int arr[], int n)
{
    int minval = arr[0];
    #pragma omp parallel for reduction(min:minval)
    for(int i = 0; i < n; i++){
        if(arr[i] < minval)
        {
            minval = arr[i];
        }
    }
    return minval;
}

int maxvalsequential(int arr[], int n)
{
    int maxval = arr[0];
    for(int i = 0;i<n;i++){
        if(arr[i] > maxval){
            maxval = arr[i];
        }
    }
    return maxval;
    
}

int maxvalparallel(int arr[], int n){
    int maxval = arr[0];
    #pragma omp parallel for reduction(max:maxval)
    for(int i = 0; i < n; i++){
        if(arr[i] > maxval){
            maxval = arr[i];
        }
    }
    return maxval;
}

double sumsequential(int arr[], int n){
    int sum = 0;
    for(int i = 0; i<n; i++){
        sum+=arr[i];
    }
    return sum;
}

double sumparallel(int arr[], int n){
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0;i<n; i++){
        sum+=arr[i];
    }
    return sum;
}

double avgsequential(int arr[], int n){
    return (double)sumsequential(arr,n)/n;
}

double avgparallel(int arr[], int n){
    return (double)sumparallel(arr,n)/n;
}

int main(){
    int n;
    cout<<"Enter number of elements: ";
    cin>>n;
    cout<<endl;

    int* arr = new int[n];

    srand(time(0));

    for(int i=0;i<n;i++)
    {
        arr[i] = rand() % 200;
    }
    
    cout<<"Sum Sequential: "<<endl;
    auto start = high_resolution_clock::now();
    cout<<sumsequential(arr,n)<<endl;
    auto stop = high_resolution_clock::now();
    auto sumseqdur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for Sum Sequential: "<<sumseqdur.count()<<endl;

    cout<<"Sum parallel: "<<endl;
    start = high_resolution_clock::now();
    cout<<sumparallel(arr,n)<<endl;
    stop = high_resolution_clock::now();
    auto sumpardur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for Sum Parallel: "<<sumpardur.count()<<endl;

    cout<<"Speed Up Factor: "<<sumseqdur/sumpardur<<endl;

    cout<<"MinVal Sequential: "<<endl;
     start = high_resolution_clock::now();
    cout<<minvalsequential(arr,n)<<endl;
     stop = high_resolution_clock::now();
    auto minseqdur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for MinVal Sequential: "<<minseqdur.count()<<endl;

    cout<<"MinVal parallel: "<<endl;
    start = high_resolution_clock::now();
    cout<<minvalparallel(arr,n)<<endl;
    stop = high_resolution_clock::now();
    auto minpardur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for MinVal Parallel: "<<minpardur.count()<<endl;

    cout<<"Speed Up Factor: "<<minseqdur/minpardur<<endl;

    cout<<"MaxVal Sequential: "<<endl;
     start = high_resolution_clock::now();
    cout<<maxvalsequential(arr,n)<<endl;
     stop = high_resolution_clock::now();
    auto maxseqdur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for MaxVal Sequential: "<<maxseqdur.count()<<endl;

    cout<<"MaxVal parallel: "<<endl;
    start = high_resolution_clock::now();
    cout<<maxvalparallel(arr,n)<<endl;
    stop = high_resolution_clock::now();
    auto maxpardur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for MaxVal Parallel: "<<maxpardur.count()<<endl;

    cout<<"Speed Up Factor: "<<maxseqdur/maxpardur<<endl;

    cout<<"Avg Sequential: "<<endl;
     start = high_resolution_clock::now();
    cout<<avgsequential(arr,n)<<endl;
    stop = high_resolution_clock::now();
    auto avgseqdur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for Avg Sequential: "<<avgseqdur.count()<<endl;

    cout<<"Avg parallel: "<<endl;
    start = high_resolution_clock::now();
    cout<<avgparallel(arr,n)<<endl;
    stop = high_resolution_clock::now();
    auto avgpardur = duration_cast<microseconds>(stop-start);
    cout<<"Time taken for Avg Parallel: "<<avgpardur.count()<<endl;

    cout<<"Speed Up Factor: "<<avgseqdur/avgpardur<<endl;
}