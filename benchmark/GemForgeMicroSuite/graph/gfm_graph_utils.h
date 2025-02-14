#ifndef GEM_FORGE_MICRO_SUITE_GRAPH_UTILS_H
#define GEM_FORGE_MICRO_SUITE_GRAPH_UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t GraphIndexT;
typedef struct {
  GraphIndexT numNodes, numEdges;
  GraphIndexT *edgePtr; // Pointer into edgeList.
  GraphIndexT *edges;
  GraphIndexT root; // Used for BFS.
} GraphCSR;

GraphCSR readInGraph(const char *fn);

typedef uint32_t WeightT;
typedef struct {
  GraphIndexT destVertex;
  WeightT weight;
} WeightEdgeT;

typedef struct {
  GraphIndexT numNodes, numEdges;
  GraphIndexT *edgePtr; // Pointer into edgeList.
  WeightEdgeT *edges;
  GraphIndexT root; // Used for BFS.
} WeightGraphCSR;

WeightGraphCSR readInWeightGraph(const char *fn);

GraphCSR readInGraph(const char *fn) {
  GraphCSR graph;
  FILE *f = fopen(fn, "rb");
  fread(&graph.numNodes, sizeof(graph.numNodes), 1, f);
  fread(&graph.numEdges, sizeof(graph.numEdges), 1, f);

  // We make the last pos numEdges so that we can calculate out-degree.
  graph.edgePtr =
      aligned_alloc(sizeof(graph.edgePtr[0]),
                    sizeof(graph.edgePtr[0]) * (graph.numNodes + 1));
  fread(graph.edgePtr, sizeof(graph.edgePtr[0]), graph.numNodes, f);
  graph.edgePtr[graph.numNodes] = graph.numEdges;

  graph.edges = aligned_alloc(sizeof(graph.edges[0]),
                              sizeof(graph.edges[0]) * graph.numEdges);
  fread(graph.edges, sizeof(graph.edges[0]), graph.numEdges, f);
  // Sanity check.
#ifndef GEM_FORGE
  for (uint64_t i = 0; i < graph.numEdges; ++i) {
    if (graph.edges[i] >= graph.numNodes) {
      printf("Illegal %u.\n", graph.edges[i]);
    }
  }
#endif
  // Read in the root.
  fread(&graph.root, sizeof(graph.root), 1, f);
  printf("NumNodes %u NumEdges %u Root %u Pos %luMB EdgePtr %luMB.\n",
         graph.numNodes, graph.numEdges, graph.root,
         sizeof(graph.edgePtr[0]) * graph.numNodes / 1024 / 1024,
         sizeof(graph.edges[0]) * graph.numEdges / 1024 / 1024);

  return graph;
}

WeightGraphCSR readInWeightGraph(const char *fn) {
  WeightGraphCSR graph;
  FILE *f = fopen(fn, "rb");
  fread(&graph.numNodes, sizeof(graph.numNodes), 1, f);
  fread(&graph.numEdges, sizeof(graph.numEdges), 1, f);

  // We make the last pos numEdges so that we can calculate out-degree.
  graph.edgePtr =
      aligned_alloc(sizeof(graph.edgePtr[0]),
                    sizeof(graph.edgePtr[0]) * (graph.numNodes + 1));
  fread(graph.edgePtr, sizeof(graph.edgePtr[0]), graph.numNodes, f);
  graph.edgePtr[graph.numNodes] = graph.numEdges;

  graph.edges = aligned_alloc(sizeof(graph.edges[0]),
                              sizeof(graph.edges[0]) * graph.numEdges);
  fread(graph.edges, sizeof(graph.edges[0]), graph.numEdges, f);
  // Sanity check.
#ifndef GEM_FORGE
  for (uint64_t i = 0; i < graph.numEdges; ++i) {
    if (graph.edges[i].destVertex >= graph.numNodes) {
      printf("Illegal %u.\n", graph.edges[i].destVertex);
    }
  }
#endif
  // Read in the root.
  fread(&graph.root, sizeof(graph.root), 1, f);
  printf("NumNodes %u NumEdges %u Root %u Pos %luMB EdgePtr %luMB.\n",
         graph.numNodes, graph.numEdges, graph.root,
         sizeof(graph.edgePtr[0]) * graph.numNodes / 1024 / 1024,
         sizeof(graph.edges[0]) * graph.numEdges / 1024 / 1024);

  return graph;
}

#endif
