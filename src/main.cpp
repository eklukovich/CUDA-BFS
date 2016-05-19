/******************************Include Files***********************************/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <list>
#include <iterator>
#include <map>
#include <algorithm>
#include <queue>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include "TopologyGenerator.h"
#include "Graph.h"
#include <sys/time.h>
#include <cuda_runtime.h>
#include "GraphData.h"


/******************************Global Consts***********************************/
#define MAX_NUM_OF_INGRESS_ 4

std::ofstream timingFile;
std::ofstream ASFile;

float totalOldBFSSequential = 0;
float totalNewBFSSequential = 0;
float totalSingleGPU = 0;
float totalMultipleGPU = 0;

/***************************Function Prototypes********************************/
std::mt19937 generator6(system_clock::to_time_t(system_clock::now()));

void error(const char *);
void thread_task(const int AS, const int num_of_nodes, int ingr_node_counter);
int random_ingress_node_selector(Graph & G);
int RT_BFS_1(Graph &G, int source, std::ofstream & outputfile2, int *);
inline void print_path(std::vector<int>path, std::ofstream & outputfile2);
bool isadjacency_node_not_present_in_current_path(int node, std::vector<int>path);
long long int getElaspedTime(struct timeval *tod1, struct timeval *tod2);
void convertAdjListToArray(Graph &G, int * vertexArray, int * edgeArray);

float processGraphSequential(Graph * graph, int source, int * sequentialCost);
float processOldGraphSequential(Graph * graph, int source, int * sequentialCost);
float processGraphSingleGPU(Graph * graph, int * source, int numIngress, int ** gpuCost);
float processGraphMultipleGPU(Graph * graph, int * source, int numIngress, int ** gpuCost, int numDevices);

bool verifyResults(int ** sequentialCost, int ** gpuCost, int ** multipleGPUCostint, int numVertices, int numIngress);

void BFS(Graph * G, int s, int * cost);

/******************************Main Program************************************/
int main(int argc, char *argv[]) 
{
    // declare variables
    std::ifstream fin;
    std::string temp;
    int numASes, numSources, refCounter;

     struct timeval startTime, finishTime;
    long long int time;

    // start timer to see how long the entire program takes
    gettimeofday(&startTime, NULL);

    std::cout << std::endl << std::endl << "Reading in AS Data..." << std::endl;

    fin.open("../data/ALL_ASes.txt.csv");

    // read in file header infomation
    fin >> temp >> temp >> temp >> temp;

    // read in the number of ASes
    fin >> temp >> numASes;

    // read in the average number of ingress nodes
    fin >> temp >> numSources;

    // create an array to hold all the data
    GraphData * ASList = new GraphData[numASes]; 



    // read all the AS data from the file
    for(int i = 0; i < numASes; i++)
    {
        fin >> refCounter >> ASList[i].ASNum >> ASList[i].numNodes >> ASList[i].numIngress;
    }

    // close the file
    fin.close();

    std::cout << "Finished Reading in AS Data" << std::endl << std::endl;

    // open the timing file
    timingFile.open("../bin/timings.txt");

    // process all the ASes using 100 ingress nodes
    for(int i = 0; i < numASes; i++)
    {
        thread_task(ASList[i].ASNum, ASList[i].numNodes, 100);
    }

    // process all the ASes using 50 ingress nodes
    for(int i = 0; i < numASes; i++)
    {
        thread_task(ASList[i].ASNum, ASList[i].numNodes, 50);
    }

    gettimeofday(&finishTime, NULL);
    int programTime = getElaspedTime(&finishTime, &startTime);


    timingFile << std::endl << std::endl; //<< "Total Old BFS Execution Time (sec), " << totalOldBFSSequential << std::endl;
    timingFile << "Total New BFS Execution Time (sec), " << totalNewBFSSequential << std::endl;
    timingFile << "Total Single GPU Execution Time (sec), " << totalSingleGPU << std::endl;
    timingFile << "Total Multiple GPU Execution Time (sec), " << totalMultipleGPU << std::endl;


    timingFile << std::endl << std::endl << "Total Program Execution Time (sec), " << programTime / 1000000.0f << std::endl;

    // clean up memory
    delete [] ASList;

    // close the timing file
    timingFile.close();

    // end the program
    return 0;
}

/*************************Function Implementation*******************************/
void error(const char *msg) 
{
    perror(msg);
    exit(0);
}


void thread_task(const int AS, const int num_of_nodes, int ingr_node_counter) 
{
    std::cout << std::endl << "------------ PROCESSING AS GRAPH --------------------" << std::endl << std::endl;

    // declare variables
    std::stringstream sstm;
    std::string ID_ = "AS";
    Graph * gr = NULL;

    std::uniform_real_distribution<double> RANDOM_GENERATOR(1, MAX_NUM_OF_INGRESS_);
    float multipleDeviceElapsedTime = 0, singleDeviceElapsedTime = 0, hostElapsedTime = 0, oldHostElapsedTime = 0;

    // create the AS id for the file name
    sstm << ID_ << AS;
    ID_ = sstm.str();

    std::cout << "Generating Topology..." << std::endl;

    // create brite and generate the topology
    Brite *b_topology = new Brite();
    
    if(num_of_nodes < 500000)
    {
	std::cout << "Generating" << std::endl;
        gr = b_topology->GenerateTopology(AS, num_of_nodes);
    }
    else
    {
	std::cout << "From File!" << std::endl;
        // convert number of nodes to a string
        std::string node = std::to_string(num_of_nodes);

        // read in the topology from BRITE file
        gr = b_topology->Populate_Topology_Result("../data/" + node +".brite");
    }

    std::cout << "Finished Generating" << std::endl;


    // get the number of vertices
    int numVertices = gr->numberOfVertex();


    // if no ingress nodes then generate a random amount (1 to 10)
    if (ingr_node_counter == 0) 
        {
            ingr_node_counter = RANDOM_GENERATOR(generator6);
        }

    // create memory for the results
    int ** gpuCost = new int * [ingr_node_counter];
    int ** sequentialCost = new int * [ingr_node_counter];
    int ** oldSequentialCost = new int * [ingr_node_counter];
    int ** multipleGPUResult = new int * [ingr_node_counter];
    
    for(int i = 0; i < ingr_node_counter; i++)
    {
        gpuCost[i] = new int[numVertices];
        sequentialCost[i] = new int[numVertices];
        oldSequentialCost[i] = new int[numVertices];
        multipleGPUResult[i] = new int[numVertices];
    }

    int * source = new int [ingr_node_counter];

    std::cout << std::endl << "Running Sequential Algorithm..." << std::endl;

    // loop for all ingress nodes
    for (int i = 0; i < ingr_node_counter; i++) 
    {
        source[i] = random_ingress_node_selector(*gr);
        hostElapsedTime += processGraphSequential(gr, source[i], sequentialCost[i]);
    }

    //totalOldBFSSequential += oldHostElapsedTime;
    totalNewBFSSequential += hostElapsedTime;

    std::cout << "Finished Sequential" << std::endl;

    // print the CPU results to a file and the screen
    std::cout << ID_ << ", CPU Execution time (sec),                   " << hostElapsedTime << ", Num Vertices, " << numVertices << ", Num Ingress, " << ingr_node_counter  << std::endl; 
    timingFile << ID_ << ", CPU Execution time (sec),                  " << hostElapsedTime << ", Num Vertices, " << numVertices << ", Num Ingress, " << ingr_node_counter  << std::endl; 






    std::cout << std::endl << "Running Single GPU Algorithm..." << std::endl;
    totalSingleGPU += singleDeviceElapsedTime = processGraphSingleGPU(gr, source, ingr_node_counter, gpuCost);     
    std::cout << "Finished Single GPU" << std::endl;

    // print the Single GPU results to a file and the screen
    std::cout << ID_ << ", CUDA Single Device Execution time (sec),    " << singleDeviceElapsedTime << ", Num Vertices, " << numVertices << ", Num Ingress, " << ingr_node_counter  << std::endl;
    timingFile << ID_ << ", CUDA Single Device Execution time (sec),   " << singleDeviceElapsedTime << ", Num Vertices, " << numVertices << ", Num Ingress, " << ingr_node_counter  << std::endl; 






    // CALL MULTI-GPU FUNCTION HERE!!!!
    std::cout << std::endl << "Running Multiple GPU Algorithm..." << std::endl;
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    for(int i = 2; i <= numDevices; i++)
    {
        multipleDeviceElapsedTime = processGraphMultipleGPU(gr, source, ingr_node_counter, multipleGPUResult, i);

        std::cout << ID_ << ", CUDA Multiple Device Execution time (sec),  " << multipleDeviceElapsedTime << ", Num GPUs, " << i << ", Num Vertices, " << numVertices << ", Num Ingress, " << ingr_node_counter  << std::endl;
        timingFile << ID_ << ", CUDA Multiple Device Execution time (sec), " << multipleDeviceElapsedTime << ", Num GPUs, " << i << ", Num Vertices, " << numVertices << ", Num Ingress, " << ingr_node_counter  << std::endl;
    }
    std::cout << "Finished Multiple GPU" << std::endl << std::endl;




    std::cout << std::endl << "Verifying Results..." << std::endl;

    // verify results
    if(verifyResults(sequentialCost, gpuCost, multipleGPUResult, numVertices, ingr_node_counter))
        std::cout << "Results Match!!" << std::endl;
    else
        std::cout << "NOT EQUAL!!" << std::endl;

    std::cout << "Finished Verifying" << std::endl;

    // clean up memory   
    for(int i = 0; i < ingr_node_counter; i++)
    {
        delete [] gpuCost[i];
        delete [] sequentialCost[i];
        delete [] oldSequentialCost[i];
        delete [] multipleGPUResult[i];
    }


    // deallocate memory
    delete [] gpuCost;
    delete [] sequentialCost;
    delete [] oldSequentialCost;
    delete [] multipleGPUResult;
    delete [] source;
    delete b_topology;
    delete gr;
        
    std::cout << std::endl << "------------ FINISHED PROCESSING --------------------" << std::endl << std::endl;

}


int random_ingress_node_selector(Graph & G) {
    std::uniform_real_distribution<double> which_node(0, G.numberOfVertex());

    return which_node(generator6);
}


void BFS(Graph * G, int s, int * cost)
{
    // 
    int numVertices = G->numberOfVertex();
    int numEdges = G->numberOfEdges();

    // Mark all the vertices as not visited
    bool *visited = new bool[numVertices];

    for(int i = 0; i < numVertices; i++)
    {
        visited[i] = false;
        cost[i] = -1;
    }
 
    // Create a queue for BFS
    std::list<int> queue;
 
    // Mark the current node as visited and enqueue it
    visited[s] = true;
    cost[s] = 0;
    queue.push_back(s);
 
    // 'i' will be used to get all adjacent vertices of a vertex
 
    while(!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        s = queue.front();
       // std::cout << s << " ";
        queue.pop_front();
 
        // Get all adjacent vertices of the dequeued vertex s
        // If a adjacent has not been visited, then mark it visited
        // and enqueue it
        for (auto i : G->graphVector[s].edges)
        {
            int dest = std::get<2>(i).get_Vertex_ID();

            if(!visited[dest])
            {
                visited[dest] = true;
                queue.push_back(dest);
                cost[dest] = cost[s] + 1;

            }
        }
    }

    // delete memory
    delete [] visited;
}


int RT_BFS_1(Graph &G, int source, std::ofstream & outputfile2, int * cost) {

    std::vector<int> path;
    
    path.push_back(source);
    std::queue<std::vector<int> >q;
    q.push(path);

    std::map <int, unsigned int> flag1, vertex_to_index;
    std::map <int, bool> flag2;

    int ii = 0;
    
    // for each vertex in the graph
    for (auto aa : G.graphVector) 
        {
            // set source to vertex distance to infinity
            flag1.insert(std::make_pair(aa.ID, 9999));
            
            // set vertex as not visited
            flag2.insert(std::make_pair(aa.ID, false));
            
            vertex_to_index.insert(std::make_pair(aa.ID, ii++));
        }
    
    int deepness = 1;
    
    // loop until queue is empty
    while (!q.empty()) 
        {
            // get a path from the queue
            path = q.front();
            q.pop();

            // get the last node in the path
            int last_nodeof_path = path[path.size() - 1];

            // print out the current path to the file
            //print_path(path, outputfile2);

            int grp_pos = last_nodeof_path;

            // loop through all the vertex's edges
            for (auto it : G.graphVector[grp_pos].edges) 
                {
                    // get the connected vertex id
                    int v_id = std::get<2>(it).get_Vertex_ID();
                    
                    // get the index of the vertex
                    int index_id = vertex_to_index.find(v_id)->second;
                    
                    // if the vertex distance is greater than the path size
                    if (flag1.find(v_id)->second >= path.size()) 
                        {
                            // if the vertex is not in the current path and has not been visited then add to queue
                            if (isadjacency_node_not_present_in_current_path(index_id, path) && flag2.find(v_id)->second == false) 
                                {
                                    // copy the current path vector into a new one
                                    std::vector<int> new_path(path.begin(), path.end());
                                    
                                    // add the new vertex
                                    new_path.push_back(index_id);
                                    
                                    // push the new path on the queue
                                    q.push(new_path);
                                    
                                    // set the distance to the vertex to the path size
                                    flag1[v_id] = path.size();
                                    cost[v_id] = path.size();
                                }
                        }
                }
            
            flag2.find(G.graphVector[grp_pos].ID)->second = true;

            deepness++;
        }

    //std::cout << "\nRT_BFS finished Process  \n ------------";

    return 1;
}


inline void print_path(std::vector<int>path, std::ofstream & outputfile2) {

    for (unsigned int i = 0; i < path.size(); ++i) {

        outputfile2 << path[i] << "    ";
    }

    outputfile2 << "\n";
}


bool isadjacency_node_not_present_in_current_path(int node, std::vector<int>path) {

    std::vector<int>::iterator it = std::find(path.begin(), path.end(), node);

    if (it == path.end())
        return true;
    else
        return false;
}


long long int getElaspedTime(struct timeval *tod1, struct timeval *tod2)
  {  
      long long t1, t2;
      t1 = tod1->tv_sec * 1000000 + tod1->tv_usec;
      t2 = tod2->tv_sec * 1000000 + tod2->tv_usec;
      return t1 - t2;
  }


bool verifyResults(int ** sequentialCost, int ** gpuCost, int ** multipleGPUCost, int numVertices, int numIngress)
{
    for(int i = 0; i < numIngress; i++)
    {
        for(int j = 0; j < numVertices; j++)
        {
            if((multipleGPUCost[i][j] != gpuCost[i][j]) && (multipleGPUCost[i][j] != sequentialCost[i][j]))
            {
                return false;
            }
        }        
    }

    return true;
} 


float processGraphSequential(Graph * graph, int source, int * sequentialCost)
{
    // declare variables
    struct timeval startTime, finishTime;
    long long int time;

    // get the number of vertice
    int numVertices = graph->numberOfVertex();

    std::ofstream outputfile2;
    //outputfile2.open("RESULTS/PATHS/" + ID_ + ".txt");

    for(int i = 0; i < numVertices; i++)
    {
        sequentialCost[i]= 0;
    }

    // start sequential timer
    gettimeofday(&startTime, NULL);

    // run BFS on the Graph
    //RT_BFS_1(*graph, source, outputfile2, sequentialCost);
    BFS(graph, source, sequentialCost);

    // end sequential timer
    gettimeofday(&finishTime, NULL);

    // calculate amount of execution timeprocessOldGraphSequential
    int sequentialTime = getElaspedTime(&finishTime, &startTime);

    return sequentialTime / 1000000.0f;
} 

float processOldGraphSequential(Graph * graph, int source, int * sequentialCost)
{
    // declare variables
    struct timeval startTime, finishTime;
    long long int time;

    // get the number of vertice
    int numVertices = graph->numberOfVertex();

    std::ofstream outputfile2;
    //outputfile2.open("RESULTS/PATHS/" + ID_ + ".txt");

    for(int i = 0; i < numVertices; i++)
    {
        sequentialCost[i]= 0;
    }

    // start sequential timer
    gettimeofday(&startTime, NULL);

    // run BFS on the Graph
    RT_BFS_1(*graph, source, outputfile2, sequentialCost);
    //BFS(graph, source, sequentialCost);

    // end sequential timer
    gettimeofday(&finishTime, NULL);

    // calculate amount of execution time
    int sequentialTime = getElaspedTime(&finishTime, &startTime);

    return sequentialTime / 1000000.0f;
} 
