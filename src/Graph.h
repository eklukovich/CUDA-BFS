#ifndef GRAPH_H
#define GRAPH_H

#include <list> //std::list
#include <vector> //std::vector
#include <random> //std::mt19937, std::uniform_real_distribution
#include <limits> //std::numeric_limits
#include <utility> //std::pair
#include <map>
#include <chrono> //std::chrono, ::system_clock, ::to_time_t, ::now
#include <istream> //std::istream
#include "vertex.h"

using std::chrono::system_clock;

typedef std::multimap<int, std::pair<int, int>> MultiMaptype;
typedef std::map<int,int> IngresPairs;
 typedef std::multimap<std::pair<int, int>,int > myMap;


class Graph {
private:
	
	
	int numVertex, numEdges;
	
	//Add an edge to a vertex
	void addEdge(int v, int b, double len,std::string layer);

	//Add an edge to a vertex
	void addEdge2(int v, int b, double len,std::string layer);
	
public:
	int ID;
	std::vector<Vertex> graphVector;
	

	
	
	//Constructors

	Graph();

	//This constructor creates a randomly generated graph given:
	//density, number_of_vertex, and maximum edge length
	Graph(const double density, const double distance_range, const int numVert) ;

	//This constructor is specifically for Brite Topology Generator 
	Graph(int num_of_vertex,int num_of_edges, std::vector <int> &nodes,myMap edges);

	// // Density,Distance Range,AS ID,Vertex Number
	Graph(const double density, const double distance_range, const int AS_ID, const int NumberofNodes);


	//This constructor creates a graph from
	//an input file
	//Read from input file constructor
	/* format is:
	    number of vertex

	    vertex_i vertex_j cost_i_to_j
	    vertex_i vertex_j cost_i_to_j
	    ...
	*/
	Graph(int ASID,std::istream &iss,IngresPairs ingr) ;
	
	//INF
	static double KMAX;
	
	int numberOfVertex() const ;
	int numberOfEdges() const;
	
	typedef std::tuple<std::string,double, Vertex&> PIRV;
	~Graph();
	
	//Add an edge between two different graph nodes
	void addEdgebetweenLayers(int v,Graph *g2, int b, double len,std::string layer);
	
	void PrintEdges();
};



#endif