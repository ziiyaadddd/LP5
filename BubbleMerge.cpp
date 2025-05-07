#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

void bubblesortsequential(int arr[], int n)
{
    for(int i = 0;i<n;i++)
    {
        for(int j = i+1;j<n;j++){
            if(arr[j] < arr[i]) swap (arr[i], arr[j]);
        }
    }
}

void bubblesortparallel(int arr[], int n){
    #pragma omp parallel
    {
        for(int i =0; i< n; i++){
            //even computation
            #pragma omp for
                for(int j = 2;j<n;j+=2){
                    if(arr[j-1]>arr[j])
                    {
                        swap(arr[j],arr[j-1]); 
                }
              
            }

            #pragma omp for
                for(int j = 1; j< n;j+=2)
                {
                    if(arr[j-1]>arr[j]){
                        swap(arr[j],arr[j-1]);
                    }
                }
        }
   
    }
}

void mergesortsequential(int arr[], int low, int mid, int high)
{
    //size of subarrays
    int n1 = (mid-low)+1;
    int n2 = (high - mid);

    //create arrays
    int left[n1], right[n2];

    for(int i = 0; i< n1; i++){
        left[i] = arr[low+i];
    }

    for(int j = 0; j<n2;j++){
        right[j] = arr[mid+1+j];
    }

    int i = 0, j = 0,k = low;

    while(i < n1 && j < n2){
        if(left[i] <= right[j]){
            arr[k++] = left[i++];

        }else{
            arr[k++] = right[j++];
        }
    }

    //copy remaining elements of left[]
    while (i < n1)
        arr[k++] = left[i++];

    // Copy remaining elements of right[]
    while (j < n2)
        arr[k++] = right[j++];

}

void mergesort(int arr[], int low, int high){
    if(low < high){
        int mid = (low + high) / 2;
        
        mergesort(arr, low, mid);
        mergesort(arr,mid+1, high);

        mergesortsequential(arr, low, mid, high);
    }
}

void mergesortparallel(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        // Parallelize the recursive calls using OpenMP
        #pragma omp parallel sections
        {
            #pragma omp section
            mergesort(arr, low, mid);  // Sort left half

            #pragma omp section
            mergesort(arr, mid + 1, high);  // Sort right half
        }

        // Merge the sorted halves sequentially
        mergesortsequential(arr, low, mid, high);
    }
}

int main()
{
    omp_set_num_threads(8);

    int n;
    cout<<"Enter number of elements: ";
    cin>>n;
    cout<<endl;

    int* arr = new int[n];

    srand(time(0));

    for(int i = 0; i< n; i++)
    {
        arr[i] = rand() % 20;
    }

    int* temp = new int[n];
    for(int i = 0; i< n; i++)
    {
        temp[i] = arr[i];
    }

    auto start = high_resolution_clock::now();
    bubblesortsequential(temp,n);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    for(int i = 0; i<n;i++){
        cout<<temp[i]<<" ";
    }
    
    delete[] temp;
    temp = new int[n];
    for (int i = 0; i < n; i++) {
        temp[i] = arr[i];
    }
    cout<<endl;

     start = high_resolution_clock::now();
    bubblesortparallel(temp,n);
     stop = high_resolution_clock::now();
    auto parduration = duration_cast<microseconds>(stop-start);
    for(int i = 0; i<n;i++){
        cout<<temp[i]<<" ";
    }
    
    delete[] temp;
    temp = new int[n];
    for (int i = 0; i < n; i++) {
        temp[i] = arr[i];
    }
    cout<<endl;

    cout<<"Merge Sort: "<<endl;
    start = high_resolution_clock::now();
    mergesort(temp,0,n-1);
    stop = high_resolution_clock::now();
    auto mergduration = duration_cast<microseconds>(stop-start);
    for(int i = 0; i<n;i++){
        cout<<temp[i]<<" ";
    }
    
    delete[] temp;
    temp = new int[n];
    for (int i = 0; i < n; i++) {
        temp[i] = arr[i];
    }
    cout<<endl;

    cout<<"Merge Sort Parallel: "<<endl;
    start = high_resolution_clock::now();
    mergesortparallel(temp,0,n-1);
    stop = high_resolution_clock::now();
    auto mergparduration = duration_cast<microseconds>(stop-start);
    for(int i = 0; i<n;i++){
        cout<<temp[i]<<" ";
    }
    
    cout<<endl<<"Time taken Bubble Sequential: "<<duration.count()<<endl;
    cout<<endl<<"Time taken Bubble parallel: "<<parduration.count()<<endl;
    cout<<endl<<"Time taken merge sequential: "<<mergduration.count()<<endl;
    cout<<endl<<"Time taken parallel: "<<mergparduration.count()<<endl;
}