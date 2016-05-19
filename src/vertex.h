#ifndef VERTEX_H
#define VERTEX_H


#include <list> //std::list
#include <utility> //std::pair
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>


class Graph;
typedef struct vert {

	//List of edges connected to current vertex
	std::list<std::tuple< std::string,double, vert&> > edges;


	
        //unique id to use for indexing this vertex
	int ID;
	std::string type;
	std::string AS_ID;

	


	vert(std::list<std::tuple<std::string,double, vert&> > edge, int ID,std::string type,std::string AS_ID): 
		
		edges(edge),
		ID(ID),
		type(type),
		AS_ID(AS_ID){

	}

	void print_edges(){

		PRINT_ELEMENTS(edges,"\nEdges: ");
		std::cout<<std::endl;
		//for (std::list<std::tuple<int,double, vert&>>::const_iterator iterator = edges.begin(), end = edges.end(); iterator != end; ++iterator) 

			// prints the edge distances for each vertex id
			//			std::cout << iterator->first << "-"<< iterator->second.get_Vertex_ID() ;
			//	std::cout << std::get <3>(iterator)->second.get_Vertex_ID()  << "-";
		//std::cout<<std::endl;
	}

	int get_Vertex_ID(){return ID;}

	void PRINT_ELEMENTS (const std::list<std::tuple<std::string,double, vert&>> coll, const char* optcstr="")
	{
		std::list<std::tuple<std::string,double, vert&> >::const_iterator pos;
		std::cout << "VERTEX: " << ID << " ";
		std::cout << optcstr;
		for (auto&& item : coll) {
			std::cout << std::get<2>(item).get_Vertex_ID()   << ' ';
		}
		std::cout << std::endl;
	}

Graph *Subgraph_;
} Vertex;

#endif