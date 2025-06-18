#pragma once
#include "base.h"
#include <vector>
using namespace std;
using qt_node_id = int;
static constexpr qt_node_id qt_node_null = qt_node_id(-1);
static constexpr std::size_t MAX_QUADTREE_DEPTH = 64;
struct quadtree
{
private:
    struct quadtree_node
    {
        qt_node_id children[2][2]{
            {qt_node_null,qt_node_null},
            {qt_node_null,qt_node_null}};
    };
    box2 bbox;
    qt_node_id root;
    std::vector<qt_node_id> nodes;
    std::vector<float2> points;
    std::vector<int> node_points_begin;
    
    template <typename Iterator>
    qt_node_id build_tree_internal(quadtree & tree, box2 const & bbox,Iterator begin, Iterator end, std::size_t depth_limit)
    {
        if (begin == end) return -1;
        qt_node_id result = tree.nodes.size();
        tree.node_points_begin[result] = (begin - tree.points.begin());
        if (depth_limit == 0) return result;
        tree.nodes.emplace_back();
        if (begin + 1 == end) return result;
        float2 center = midpoint(bbox.min, bbox.max);
        auto bottom = [center](float2 const & p){ return p.y < center.y; };
        auto left   = [center](float2 const & p){ return p.x < center.x; };
        Iterator split_y = std::partition(begin, end, bottom);
        Iterator split_x_lower = std::partition(begin, split_y, left);
        Iterator split_x_upper = std::partition(split_y, end, left);
        tree.nodes[result].children[0][0] = build_tree_internal(tree, { bbox.min, center }, begin, split_x_lower, depth_limit-1);
        tree.nodes[result].children[0][1] = build_tree_internal(tree, { { center.x, bbox.min.y }, { bbox.max.x, center.y } }, split_x_lower, split_y, depth_limit-1);
        tree.nodes[result].children[1][0] = build_tree_internal(tree, { { bbox.min.x, center.y }, { center.x, bbox.max.y } }, split_y, split_x_upper, depth_limit-1);
        tree.nodes[result].children[1][1] = build_tree_internal(tree, { center, bbox.max }, split_x_upper, end, depth_limit-1);
        return result;
    };

public:
    template <typename Iterator>
    quadtree build(vector<float2> points)
    {
        quadtree result;
        result.points = std::move(points);
        result.root = build_impl(result,
            compute_bbox(result.points.begin, result.points.end),
            result.points.begin, result.points.end);
        result.node_points_begin.push_back(result.points.size());
        return result;
    }
};


