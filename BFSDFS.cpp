#include<bits/stdc++.h>
#include<omp.h>
using namespace std;
using namespace std::chrono;

struct treenode {
    int val;
    treenode* left, *right;
    treenode(int v): val(v), left(NULL), right(NULL) {};
};

treenode* createsimpletree() {
    treenode* root = new treenode(1);
    root-> left = new treenode(2);
    root-> right = new treenode(3);
    root->left-> left = new treenode(4);
    root->right->right = new treenode(5);
    return root;
}

void parallelBFS(treenode* root){
    if(!root) return;
    queue<treenode*> q;
    q.push(root);
    
    while(!q.empty()) {
        int size = q.size();

        #pragma omp parallel for
        for(int i = 0; i<size;i++){
            treenode* node;
            #pragma omp critical
            {
                node = q.front(); q.pop();
                cout<< "BFS Node: " << node->val << " by thread: " << omp_get_thread_num() << endl;
            }

            #pragma omp critical
            {
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
            }
        }
    }
}

void parallelDFS(treenode* root) {
    if (!root) return;
    stack<treenode*> s;
    s.push(root);

    #pragma omp parallel
    {
        while (true) {
            treenode* node = nullptr;
            #pragma omp critical
            {
                if (!s.empty()) {
                    node = s.top(); s.pop();
                    cout << "DFS node " << node->val << " by thread " << omp_get_thread_num() << endl;
                }
            }

            if (!node) break;

            #pragma omp critical
            {
                if (node->right) s.push(node->right);
                if (node->left) s.push(node->left);
            }
        }
    }
}

int main() {
    treenode* root = createsimpletree();
    cout << "Parallel BFS:\n";
    auto start = high_resolution_clock::now();
    parallelBFS(root);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << " BFS time: " << duration.count();
    cout << "\nParallel DFS:\n";
    start = high_resolution_clock::now();
    parallelDFS(root);
    stop = high_resolution_clock::now();
    auto parduration = duration_cast<microseconds>(stop-start);
    cout << " DFS time: " << parduration.count();
    return 0;
}