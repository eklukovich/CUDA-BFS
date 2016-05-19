#include "Graph.h"
#include "vertex.h"

#include <iostream>
#include <string>
#include <map>
#include <numeric>
#include <set>
#include <vector>
#include <algorithm>    // std::sort


#define PERCENTAGE_OF_BEGINNING_NODES_ 0.15
#define PERCENTAGE_OF_MEDIAN_NODES_ 0.40
#define DISTANCE_RANGE_ 4
#define DENSITY_ 0.5

#define OFFSET_EXPONENT_ 0.20

using std::chrono::system_clock;

typedef std::multimap<int, std::pair<int, int>> MultiMaptype;	
typedef std::multimap<int, std::pair<int, int>>::const_iterator it_type;
typedef std::map<int,int> IngresPairs;
 typedef std::multimap<std::pair<int, int>,int > myMap;


//Seed random number generator
std::mt19937 generator(system_clock::to_time_t(system_clock::now()));

int distict_abs_counter(const std::vector<std::pair<int,int>>& v)
{
   std::set<int> distinct_container;

   for(auto curr_int = v.begin(), end = v.end(); // no need to call v.end() multiple times
       curr_int != end;
       ++curr_int)
   {
       // std::set only allows single entries
       // since that is what we want, we don't care that this fails 
       // if the second (or more) of the same value is attempted to 
       // be inserted.
	   distinct_container.insert(abs(curr_int->first));
   }

   return distinct_container.size();
}	
std::set<int> distict_abs_dataset(const std::vector<std::pair<int,int>>& v)
{
   std::set<int> distinct_container;

   for(auto curr_int = v.begin(), end = v.end(); // no need to call v.end() multiple times
       curr_int != end;
       ++curr_int)
   {
       // std::set only allows single entries
       // since that is what we want, we don't care that this fails 
       // if the second (or more) of the same value is attempted to 
       // be inserted.
	   distinct_container.insert(abs(curr_int->first));
   }

   return distinct_container;
}

	//Add an edge to a vertex
void Graph:: addEdge(int v, int b, double len,std::string layer) 
{ 
//  std::cout<<"adding edge.."<<std::endl; 
  graphVector[v].edges.emplace_back(layer,len, graphVector[b]); 
}

	//Add an edge to a vertex
void Graph:: addEdge2(int v, int b, double len,std::string layer) 
{ 
//  std::cout<<"adding edge.."<<std::endl; 
  graphVector[v].edges.emplace_back(layer,len, graphVector[b]); 
}


//Add an edge between two different graph nodes
void Graph::addEdgebetweenLayers(int v,Graph *g2, int b, double len,std::string layer){
	graphVector[v].edges.emplace_back(layer,len, g2->graphVector[b]); 
	numEdges++;
}

	//Constructors

Graph::Graph(){}

	/*	This constructor creates a randomly generated graph given:
	*	density, number_of_vertex, and maximum edge length
	*/
Graph::Graph(const double density, const double distance_range, const int numVert): numVertex(numVert), numEdges(0) 
{
		std::string layer = "self";
		std::uniform_real_distribution<double> edge_exist(0, 1);
		std::uniform_real_distribution<double> edge_len(1, distance_range);
		
		graphVector.resize(numVertex, Vertex(std::list<PIRV>(), 0,"",""));  // nullptr });


	

		for (int s = 0; s < numVertex; ++s) {
			for (int k = 0; k < s; ++k )
				if (edge_exist(generator) < density) {
					double edgeLength = edge_len(generator);
					++numEdges;
					addEdge(k, s, edgeLength,layer);
					addEdge(s, k, edgeLength,layer);
				}
		}		
}


	/*	This constructor creates a randomly generated graph given:
	*	density, number_of_vertex, and maximum edge length
	*/
Graph::Graph(int num_of_vertex,int num_of_edges, std::vector <int> &nodes,myMap edges):numVertex(num_of_vertex),numEdges(num_of_edges) 
{
		std::string layer = "self";

		graphVector.resize(numVertex, Vertex(std::list<PIRV>(), 0,"",""));  // nullptr });

		auto it2 = nodes.begin();

		//Assign a number to each vertex for unique identification
        for (int s = 0; s < numVertex; ++s)
        	{
			 graphVector[s].ID = *it2;
             it2++;
            }
		
	
		for(auto aa: edges)
		{
			addEdge(aa.first.first, aa.first.second, aa.second,layer);
			addEdge(aa.first.second, aa.first.first, aa.second,layer);
		}

	
}

	/*	This constructor creates a preferential attachment graph given:
	*	density, number_of_vertex,AS_ID and maximum edge length
	*/
Graph::Graph(const double density, const double distance_range, const int AS_ID,const int NumberofNodes):	
	numVertex(NumberofNodes),
	numEdges(0),ID(AS_ID)
{


		
		std::string layer = std::to_string(AS_ID);
		std::uniform_real_distribution<double> edge_exist(0, 1);
		std::uniform_real_distribution<double> pick_a_random_value(0, 1);
		std::uniform_real_distribution<double> edge_len(1, distance_range);
		
		// Add m0 nodes to G.
		int m0=numVertex * PERCENTAGE_OF_BEGINNING_NODES_; 
		if(m0==0 || m0==1) m0=2;
		graphVector.resize(numVertex, Vertex(std::list<PIRV>(),0,"",""));  // nullptr });

			for (int s = 0; s < numVertex; ++s)
			graphVector[s].ID = s;


		//Connect every node in G to every other node in G, i.e. create a complete graph.
		for (int s = 0; s < m0; ++s) {
			graphVector[s].ID = s;
			for (int k = 0; k < s; ++k )
				 {
					double edgeLength = edge_len(generator);
					++numEdges;
					addEdge(k, s, edgeLength,layer);
					addEdge(s, k, edgeLength,layer);
				}
		}

		int m1=(1 + rand() % numVertex )*PERCENTAGE_OF_MEDIAN_NODES_;
		//Create a new node i.
		int Vertex_i_pos_=m0;

		while (m1==0)
		{
			m1=(1 + rand() % numVertex )*PERCENTAGE_OF_MEDIAN_NODES_;
		}

		while(Vertex_i_pos_<numVertex)
		{
			for(int y=0;y<m1;++y){
				//Pick a node j uniformly at random from the graph G
				int randomIndex = rand() % Vertex_i_pos_;
		
				//Set P = (k(j)/k_tot)^a		
			
				double val=(double)(graphVector[randomIndex].edges.size())/(double)((2*numberOfEdges()));
				double Probability= std::pow(val,OFFSET_EXPONENT_);//
		
				double R=pick_a_random_value(generator);
		
				if(Probability>R)
				{
							//double edgeLength = edge_len(generator);
							++numEdges;
							addEdge(randomIndex, Vertex_i_pos_, 1, layer);//edgeLength,layer);
								addEdge(Vertex_i_pos_, randomIndex, 1, layer);//edgeLength,layer);
				}
			}
			m1=(1 + rand() % numVertex )*PERCENTAGE_OF_MEDIAN_NODES_;
			while (m1==0)
			{
				m1=(1 + rand() % numVertex )*PERCENTAGE_OF_MEDIAN_NODES_;
			}
			Vertex_i_pos_++;
		}

		//guarantee that there is no node without edges
		for(auto &it:graphVector)
			while(it.edges.size()==0)
			{
				//Pick a node j uniformly at random from the graph G
				int randomIndex = rand() % Vertex_i_pos_;
				

				if (edge_exist(generator) < density) 
					{
						//double edgeLength = edge_len(generator);
						++numEdges;
						addEdge(it.ID, randomIndex, 1, layer);//edgeLength,layer);
						addEdge(randomIndex, it.ID, 1, layer);//edgeLength,layer);
					}

			}
		
		
		
}

	/*	This constructor creates a graph from
		an input file
		Read from input file constructor
		vertex numbers start from 0, i.g. i,j=0,1,2,3
		format is:
	    number of vertex

	    vertex_i vertex_j cost_i_to_j
	    vertex_i vertex_j cost_i_to_j
	    ...
	*/


Graph::	Graph(int ASID,std::istream &iss,IngresPairs ingr) : numEdges(0),ID(ASID) {
		int i, j, cost;
		std::string layer = "AS_Layer";

		std::uniform_real_distribution<double> edge_exist(0, 1);
		std::uniform_real_distribution<double> edge_len(1, DISTANCE_RANGE_);
		

		std::vector<std::pair<int,int>> data;


		while(iss >> i >> j >> cost) 
		{
			data.push_back(std::make_pair(i,j));
		}


		std::sort( data.begin(), data.end() );
        data.erase( std::unique( data.begin(), data.end() ), data.end() );

		std::vector<int> unique;

		for(auto & d:data){
			auto search =find (unique.begin(), unique.end(), d.first) ;
				
			if(search == unique.end())
					unique.push_back(d.first);
			
			auto search2 = find (unique.begin(), unique.end(), d.second) ;
				
			if(search2 == unique.end())
					unique.push_back(d.second);
			/*
			if((std::find(unique.begin(), unique.end(), d.first)==unique.end()))
				unique.push_back(d.first);
			if((std::find(unique.begin(), unique.end(), d.second)==unique.end()))
				unique.push_back(d.second);
			*/
		}
		std::sort( unique.begin(), unique.end() );
		unique.erase( std::unique( unique.begin(), unique.end() ), unique.end() );
		numVertex= unique.size();
		

		graphVector.resize(numVertex, Vertex( std::list<PIRV>(),  0,"",""));
		int k=0;
		std::vector<int>::iterator it2 = unique.begin();
		//Assign a number to each vertex for unique identification
		for (std::vector<Vertex>::iterator it = graphVector.begin();it!=graphVector.end();++it,++k) 
		{
			
			it->ID = *it2;
			it2++;
		}


		for(auto &a:data)
		{
			auto pos_1=std::find(unique.begin(), unique.end(), a.first);
			if(pos_1!=unique.end()){
				int firstval=pos_1 - unique.begin();
				
				auto pos_2=std::find(unique.begin(), unique.end(), a.second);
				if(pos_2!=unique.end()){
					int secondval=pos_2 - unique.begin();
						addEdge(firstval, secondval, 1 ,layer);
						addEdge(secondval, firstval, 1 ,layer);
						numEdges ++;
				}
			}
		}


		//guarantee that there is no node without edges
		for(auto &it:graphVector)
			if(it.edges.size()==0)
			{
				//Pick a node j uniformly at random from the graph G
				int randomIndex = rand() % numVertex;
				
				auto i = graphVector.begin ();
				int pos;
			   while (i != graphVector.end ()){
				   if(i->ID==it.ID){
					   pos=distance (graphVector.begin (), i);
					break;
				   }
				   i++;
			   }


				if (edge_exist(generator) < 1) //if (edge_exist(generator) < DENSITY_) 
					{
						//double edgeLength = edge_len(generator);
						++numEdges;
						addEdge(pos, randomIndex, 1,layer);
						addEdge(randomIndex, pos, 1,layer);
					}

			}

	}

	
int Graph::numberOfVertex() const  { return numVertex; }

int Graph::numberOfEdges() const {return numEdges;}
	
double Graph::KMAX = std::numeric_limits<double>::infinity();

Graph::~Graph(){}

void Graph::PrintEdges(){
	std::list<std::pair<double, vert&> >::const_iterator  k;

	for (unsigned int i =0; i<graphVector.size();++i)
		graphVector[i].print_edges();
}

/*		GENERATE AS GRAPH BASED ON INGRESS ANNOUNCEMENTS


Graph::	Graph(int ASID,std::istream &iss,IngresPairs ingr) : numEdges(0),ID(ASID) {
		int i, j, cost, inc = 1, index = 0;
		std::string layer = "AS_Layer";

		std::uniform_real_distribution<double> edge_exist(0, 1);
		std::uniform_real_distribution<double> edge_len(1, DISTANCE_RANGE_);
		

		std::vector<std::pair<int,int>> data;


		while(iss >> i >> j >> cost) 
		{
			data.push_back(std::make_pair(i,j));
		}


		std::sort( data.begin(), data.end() );
        data.erase( std::unique( data.begin(), data.end() ), data.end() );

		std::vector<int> unique;

		for(auto & d:data){
			auto search = ingr.find(d.first);
				
			if(search != ingr.end())
					unique.push_back(d.first);
			
			auto search2 = ingr.find(d.second);
				
			if(search2 != ingr.end())
					unique.push_back(d.second);
			
			if((std::find(unique.begin(), unique.end(), d.first)==unique.end()))
				unique.push_back(d.first);
			if((std::find(unique.begin(), unique.end(), d.second)==unique.end()))
				unique.push_back(d.second);
			}
		std::sort( unique.begin(), unique.end() );
		unique.erase( std::unique( unique.begin(), unique.end() ), unique.end() );
		numVertex= unique.size();
		

		graphVector.resize(numVertex, Vertex( std::list<PIRV>(),  0,"",""));
		int k=0;
		std::vector<int>::iterator it2 = unique.begin();
		//Assign a number to each vertex for unique identification
		for (std::vector<Vertex>::iterator it = graphVector.begin();it!=graphVector.end();++it,++k) 
		{
			
			it->ID = *it2;
			it2++;
		}


		for(auto &a:data)
		{
			if(std::find(unique.begin(), unique.end(), a.first)!=unique.end()){
				int firstval=find(unique.begin(), unique.end(), a.first) - unique.begin();
				
				if(std::find(unique.begin(), unique.end(), a.second)!=unique.end()){
					int secondval=find(unique.begin(), unique.end(), a.second) - unique.begin();
						addEdge(firstval, secondval, 0.0 ,layer);
						addEdge(secondval, firstval, 0.0 ,layer);
						numEdges ++;
				}
			}
		}


		//guarantee that there is no node without edges
		for(auto &it:graphVector)
			if(it.edges.size()==0)
			{
				//Pick a node j uniformly at random from the graph G
				int randomIndex = rand() % numVertex;
				
				auto i = graphVector.begin ();
				int pos;
			   while (i != graphVector.end ()){
				   if(i->ID==it.ID){
					   pos=distance (graphVector.begin (), i);
					break;
				   }
				   i++;
			   }


				if (edge_exist(generator) < 1) //if (edge_exist(generator) < DENSITY_) 
					{
						double edgeLength = edge_len(generator);
						++numEdges;
						addEdge(pos, randomIndex, edgeLength,layer);
						addEdge(randomIndex, pos, edgeLength,layer);
					}

			}

	}


*/

/* HOW TO USE DIRECTLY



	Graph *g=new Graph(4,3,10);

	cout<<"g->numberOfEdges() :: "<< g->numberOfEdges()<<endl;
	cout<<"g->numberOfVertex() :: "<< g->numberOfVertex()<<endl;
	
	
	
	

	
	
	
	
	
*/
