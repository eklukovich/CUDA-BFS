#ifndef _BFS_KERNEL_H_
#define _BFS_KERNEL_H_

/******************************Include Files***********************************/
#include <iostream>
#include "Graph.h"
#include <stdio.h>
#include <fstream>
#include <thread>

/******************************Global Consts***********************************/
const int MAX_THREADS_PER_BLOCK = 512;



/****************************Function Prototypes*******************************/
void convertAdjListToArray(Graph * G, int * vertexArray, int * edgeArray);
void multipleGPUThreadFunction(int * vertexArray, int * edgeArray, int numVertices, int numEdges, int * source, int deviceNum, int ** gpuCost, int numIngress, int numDevices);



/******************************CUDA Kernels************************************/
__global__ void bfs_kernel(int * vertexArray, int * edgeArray, bool * frontierArray, bool * frontierUpdatingArray, bool * visitedArray, int * costArray, int numVertices )
{
	// get the thread id
    unsigned int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    // check if the thread should process the vertex
    if(tid < numVertices && frontierArray[tid])
    {                   
    	// mark as the vertex has been processed
        frontierArray[tid] = false;

        // calulate the start and end index for the edge array
        int startIndex = vertexArray[tid];
        int endIndex = vertexArray[tid + 1];

        // loop through all the edges
        for(int i = startIndex; i < endIndex; i++)
        {                       
        	// get the destination vertex
            int destVertex = edgeArray[i];

            // if the destination has not been visited
            if(!visitedArray[destVertex])
            {
            	// increment the cost
                costArray[destVertex] = costArray[tid] + 1;
                
                // mark that the vertex needs to be processed
                frontierUpdatingArray[destVertex] = true;
            }
        }       
    }

}


__global__ void bfsUpdateArrays_kernel(bool * frontierArray, bool * frontierUpdatingArray, bool * visitedArray, bool * searching, int numVertices)
{
	// get the thread id
    unsigned int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

    // check if the thread should update the array for the vertex
    if(tid < numVertices && frontierUpdatingArray[tid])
    {
        frontierArray[tid] = true;
        visitedArray[tid] = true;
        *searching = true;
        frontierUpdatingArray[tid] = false;
    }
}



/*************************Function Implementation*******************************/
float processGraphSingleGPU(Graph * graph, int * source, int numIngress, int ** gpuCost)
    {
        // declare variables
        bool searching = true;
  		cudaEvent_t start, end;

        // get the number of vertices and edges
        int numVertices = graph->numberOfVertex();
        int numEdges = graph->numberOfEdges() * 2;

		// create device event timers
		cudaEventCreate(&start);
		cudaEventCreate(&end);

        // allocate the memory for the arrays
        int * vertexArray = new int [numVertices + 1];
        int * edgeArray = new int [numEdges];

        // start GPU timer
        cudaEventRecord( start, 0 );

        convertAdjListToArray(graph, vertexArray, edgeArray);


   
		multipleGPUThreadFunction(vertexArray, edgeArray, numVertices, numEdges, source, 0, gpuCost, numIngress, 1);
		

        // stop the GPU timer
        cudaEventRecord( end, 0 );
        cudaEventSynchronize( end );


        // calculate the amount of execution time
        float deviceWithTransferTime;
        cudaEventElapsedTime( &deviceWithTransferTime, start, end );

        // clean up timer variables
        cudaEventDestroy(start);
    	cudaEventDestroy(end);   

        // clean up memory
        delete [] vertexArray;
        delete [] edgeArray;    		

        // return the execution time
   	    return (deviceWithTransferTime / 1000.0);
    }


float processGraphMultipleGPU(Graph * graph, int * source, int numIngress, int ** gpuCost, int numDevices)
    {
        // declare variables
  		cudaEvent_t start, end;
  		int deviceThreads = 0;

        // get the number of vertices and edges
        int numVertices = graph->numberOfVertex();
        int numEdges = graph->numberOfEdges() * 2;      

		// create device event timers
		cudaEventCreate(&start);
		cudaEventCreate(&end);

        // allocate the memory for the arrays
        int * vertexArray = new int [numVertices + 1];
        int * edgeArray = new int [numEdges];


        // start GPU timer
        cudaEventRecord( start, 0 );

        // convert the graph to GPU format (one dim array)
        convertAdjListToArray(graph, vertexArray, edgeArray);

        // create a thread for each GPU
        std::thread threadList[numDevices];

		// 
		int i = 0;
		while( i < numDevices )
		{

				// launch one thread
				threadList[i] = std::thread(multipleGPUThreadFunction, vertexArray, edgeArray, numVertices, numEdges, source, i, gpuCost, numIngress, numDevices);

				// increment number of devices
				i++;
		
        }

        // wait for any remaining threads
        for(int j = 0; j < numDevices; j++)
		{
			threadList[j].join();
		}

        // stop the GPU timer
        cudaEventRecord( end, 0 );
        cudaEventSynchronize( end );

        // calculate the amount of execution time
        float deviceWithTransferTime;
        cudaEventElapsedTime( &deviceWithTransferTime, start, end );

        // clean up timer variables
        cudaEventDestroy(start);
    	cudaEventDestroy(end);

    	// clean up all the CPU memory
    	delete [] vertexArray;
    	delete [] edgeArray;


        // return the execution time
   	    return (deviceWithTransferTime /1000.0);
   	    
    }


void multipleGPUThreadFunction(int * vertexArray, int * edgeArray, int numVertices, int numEdges, int * source, int deviceNum, int ** gpuCost, int numIngress, int numDevices)
{
        // declare variables
        int numBlocks, numThreads;
        bool searching = true;

        // set the device number
        cudaSetDevice(deviceNum); 	

        for(int n = deviceNum; n < numIngress; n += numDevices )
        {
            // allocate memory for the arrays on the Host
            int * costArray = new int [numVertices];
            bool * frontierArray = new bool [numVertices];
            bool * frontierUpdatingArray = new bool [numVertices];
            bool * visitedArray = new bool [numVertices];

            // initialize array values
            for(int i = 0; i < numVertices; i++)
            {
                costArray[i] = -1;
                frontierArray[i] = false;
                frontierUpdatingArray[i] = false;
                visitedArray[i] = false;
            }

            // set values for source node
            frontierArray[source[n]] = true;
            visitedArray[source[n]] = true;
            costArray[source[n]] = 0;


            // create the cuda memory and send to the device 

                // vertex array
                int* deviceVertexArray;
                cudaMalloc( (void**) &deviceVertexArray, sizeof(int)*(numVertices + 1));
                cudaMemcpy( deviceVertexArray, vertexArray, sizeof(int)*(numVertices + 1), cudaMemcpyHostToDevice);


                // edge array
                int* deviceEdgeArray;
                cudaMalloc( (void**) &deviceEdgeArray, sizeof(int)*numEdges);
                cudaMemcpy( deviceEdgeArray, edgeArray, sizeof(int)*numEdges, cudaMemcpyHostToDevice);

                // cost array
                int* deviceCostArray;
                cudaMalloc( (void**) &deviceCostArray, sizeof(int)*numVertices);
                cudaMemcpy( deviceCostArray, costArray, sizeof(int)*numVertices, cudaMemcpyHostToDevice);

                // frontier array
                bool* deviceFrontierArray;
                cudaMalloc( (void**) &deviceFrontierArray, sizeof(bool)*numVertices);
                cudaMemcpy( deviceFrontierArray, frontierArray, sizeof(bool)*numVertices, cudaMemcpyHostToDevice);

                // frontier updating array
                bool* deviceFrontierUpdatingArray;
                cudaMalloc( (void**) &deviceFrontierUpdatingArray, sizeof(bool)*numVertices);
                cudaMemcpy( deviceFrontierUpdatingArray, frontierUpdatingArray, sizeof(bool)*numVertices, cudaMemcpyHostToDevice);

                // visited array
                bool* deviceVisitedArray;
                cudaMalloc( (void**) &deviceVisitedArray, sizeof(bool)*numVertices);
                cudaMemcpy( deviceVisitedArray, visitedArray, sizeof(bool)*numVertices, cudaMemcpyHostToDevice);

                // bool to stop BFS
                bool * deviceSearching;
                cudaMalloc( (void**) &deviceSearching, sizeof(bool));


            // calculate the number of blocks and threads
            if(numVertices > MAX_THREADS_PER_BLOCK)
            {
                numBlocks = (int) ceil(numVertices/(double)MAX_THREADS_PER_BLOCK); 
                numThreads = MAX_THREADS_PER_BLOCK; 
            }
            else
            {
                numBlocks = 1;
                numThreads = numVertices;
            }

            // start BFS search
            while(searching)
            {
            	// set the searching value to false (stop the search)
                searching = false;

                // copy the search value to the GPU
                cudaMemcpy(deviceSearching, &searching, sizeof(bool), cudaMemcpyHostToDevice);

                // start BFS kernel
                bfs_kernel<<<numBlocks, numThreads>>>(deviceVertexArray, deviceEdgeArray, deviceFrontierArray, deviceFrontierUpdatingArray, deviceVisitedArray, deviceCostArray, numVertices);

              	// start bfs update array kernel
                bfsUpdateArrays_kernel<<<numBlocks, numThreads>>>(deviceFrontierArray, deviceFrontierUpdatingArray, deviceVisitedArray, deviceSearching, numVertices);

                // get the search value back from the GPU
                cudaMemcpy(&searching, deviceSearching, sizeof(bool), cudaMemcpyDeviceToHost);
            }

            // copy the cost array from the GPU
            cudaMemcpy( gpuCost[n], deviceCostArray, sizeof(int)*numVertices, cudaMemcpyDeviceToHost);

        	// clean up all the CPU memory
        	delete [] costArray;
        	delete [] frontierArray;
        	delete [] frontierUpdatingArray;
        	delete [] visitedArray;

        	// clean up all the CUDA memory
            cudaFree(deviceVertexArray);
            cudaFree(deviceEdgeArray);
            cudaFree(deviceCostArray);
            cudaFree(deviceFrontierArray);
            cudaFree(deviceFrontierUpdatingArray);
            cudaFree(deviceVisitedArray);	
        }
}        



void convertAdjListToArray(Graph * G, int * vertexArray, int * edgeArray)
    {
        // declare variables
        int edgeCount = 0;

        // loop through all the vertices
        for (int i = 0; i < G->graphVector.size(); i++)
        {
            vertexArray[i] = edgeCount;

            for (auto it : G->graphVector[i].edges)
            {
                // put the destination in the edge array
                edgeArray[edgeCount] = std::get<2>(it).get_Vertex_ID();
                
                // increment the edge count
                edgeCount ++;
            }   
        }

        // add the index of the last edge (makes it easier to loop)
        vertexArray[G->graphVector.size()] = edgeCount;
    }      
           


#endif
