package main

type Node struct {
	X, Y, Z int
}

type Graph struct {
	vertices map[Node][]Node
}

func (g *Graph) AddVertex(node Node) {
	g.vertices[node] = []Node{}
}

func (g *Graph) AddEdge(node1, node2 Node) {
	g.vertices[node1] = append(g.vertices[node1], node2)
	//g.vertices[node2] = append(g.vertices[node2], node1)
}

func (g *Graph) ConnectAdjacentNodes() {
	for key := range g.vertices {
		x, y, z := key.X, key.Y, key.Z
		adjacent := []Node{
			{x + 1, y, z}, {x - 1, y, z},
			{x, y + 1, z}, {x, y - 1, z},
			{x, y, z + 1}, {x, y, z - 1},
		}
		for _, node := range adjacent {
			if _, exists := g.vertices[node]; exists {
				g.AddEdge(key, node)
			}
		}
	}
}

func (g *Graph) GetNeighbours(node Node) []Node {
	return g.vertices[node]
}

func NewGraph() *Graph {
	return &Graph{
		vertices: make(map[Node][]Node),
	}
}
