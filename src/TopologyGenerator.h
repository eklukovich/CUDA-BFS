#ifndef TOPOLOGY_GENERATORS_H
#define TOPOLOGY_GENERATORS_H

#include <list> //std::list
#include <vector> //std::vector
#include <random> //std::mt19937, std::uniform_real_distribution
#include <limits> //std::numeric_limits
#include <utility> //std::pair
#include <map>
#include <chrono> //std::chrono, ::system_clock, ::to_time_t, ::now
#include <istream> //std::istream
#include <fstream>
#include "vertex.h"
#include "Graph.h"
#include <string>

typedef std::map<int,int> IngresPairs;


class Brite {
private:
	int AS_ID;
	int NumberOfNodes;
	int NumberofNeighboringNode;
        int NodePlacement;
	int BWDist;		//Constant = 1, Uniform =2, HeavyTailed = 3, Exponential =4
	int BWMin;
	int BWMax;
	Graph *  EndProduct;

public:
	void Prepare_Configuration();
	void Run_Generator_Script();

	Graph* Populate_Topology_Result();
	Graph* Populate_Topology_Result(std::string fileName);

    Graph* GenerateTopology(const int AS,const int num_of_nodes);
	Graph*  GenerateTopology(IngresPairs::iterator it,const int Dist,const int Min,const int Max); //const int AS,const int NOfNodes,const int NrofNeighboringNode
	Graph*  GenerateTopology(const int AS,const int Dist,const int Min,const int Max); //const int AS,const int NOfNodes,const int NrofNeighboringNode
	Brite();

};






#endif