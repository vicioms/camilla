#include "halfedge.h"
#include <string>
#include <iostream>
using namespace std;


int main()
{
    float3 a(0.0f,0.0f,0.0f);
    float3 b(1.0,0.0,0.0f);
    float3 c(0.5, 1.0,0.0f);
    vector<float3> vertices = {a,b,c};
    halfedge* ab = new halfedge(0,1);
    halfedge* bc = new halfedge(1,2);
    halfedge* ca = new halfedge(2,0);
    ab->join_next(bc);
    bc->join_next(ca);
    ca->join_next(ab);
    polygon poly;
    poly.root = ab;

    cout << poly.size() << endl;

    auto area_vector = poly.area_vector(vertices);

    cout << area_vector.x << endl;
    cout << area_vector.y << endl;
    cout << area_vector.z << endl;

    float perimeter = poly.perimeter(vertices);

    cout << perimeter << endl;
};