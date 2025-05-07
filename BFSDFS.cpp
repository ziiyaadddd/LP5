#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

class Graph {
    unordered_map<int, vector<int>> adj;
    int V, E;

public:
    Graph() : V(0), E(0) {}

    void input() {
        cout << "Enter number of vertices: ";
        cin >> V;
        cout << "Enter number of edges: ";
        cin >> E;

        for (int i = 0; i < E; ++i) {
            int u, v;
            cout << "Enter edge (source destination): ";
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u); // For undirected graph
        }
    }

    void display() const {
        cout << "\nAdjacency List:\n";
        for (const auto &pair : adj) {
            cout << pair.first << ": ";
            for (int neighbor : pair.second)
                cout << neighbor << " ";
            cout << endl;
        }
    }

    void DFS(int start) {
        vector<bool> visited(V, false);
        cout << "\nSequential DFS starting from " << start << ": ";
        DFSUtil(start, visited);
        cout << endl;
    }

    void DFSUtil(int node, vector<bool> &visited) {
        visited[node] = true;
        cout << node << " ";
        for (int neighbor : adj[node]) {
            if (!visited[neighbor])
                DFSUtil(neighbor, visited);
        }
    }

    void ParallelDFS(int start) {
        vector<bool> visited(V, false);
        cout << "\nParallel DFS starting from " << start << ": ";
        visited[start] = true;
        cout << start << " ";

#pragma omp parallel for
        for (int i = 0; i < adj[start].size(); ++i) {
            int neighbor = adj[start][i];
#pragma omp critical
            {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    DFSUtil(neighbor, visited);
                }
            }
        }
        cout << endl;
    }

    void BFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;

        cout << "\nSequential BFS starting from " << start << ": ";
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            cout << node << " ";

            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
    }

    void ParallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;

        cout << "\nParallel BFS starting from " << start << ": ";
        while (!q.empty()) {
            int node;
#pragma omp critical
            {
                node = q.front();
                q.pop();
            }

            cout << node << " ";

#pragma omp parallel for
            for (int i = 0; i < adj[node].size(); ++i) {
                int neighbor = adj[node][i];
#pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
        cout << endl;
    }
};

int main() {
    Graph g;
    g.input();
    g.display();

    int start;
    cout << "\nEnter start node for traversals: ";
    cin >> start;

    using namespace std::chrono;

    auto t1 = high_resolution_clock::now();
    g.DFS(start);
    auto t2 = high_resolution_clock::now();
    duration<double> d1 = t2 - t1;
    cout << "Sequential DFS Time: " << d1.count() << "s\n";

    t1 = high_resolution_clock::now();
    g.ParallelDFS(start);
    t2 = high_resolution_clock::now();
    duration<double> d2 = t2 - t1;
    cout << "Parallel DFS Time: " << d2.count() << "s\n";

    t1 = high_resolution_clock::now();
    g.BFS(start);
    t2 = high_resolution_clock::now();
    duration<double> d3 = t2 - t1;
    cout << "Sequential BFS Time: " << d3.count() << "s\n";

    t1 = high_resolution_clock::now();
    g.ParallelBFS(start);
    t2 = high_resolution_clock::now();
    duration<double> d4 = t2 - t1;
    cout << "Parallel BFS Time: " << d4.count() << "s\n";

    return 0;
}
