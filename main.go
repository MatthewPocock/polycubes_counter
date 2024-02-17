package main

import (
	"fmt"
	"math"
	"os"
	"runtime/pprof"
	"strconv"
	"time"
)

var branchDepth = 4

func CreateLattice(n int) *Graph {
	latticeGraph := NewGraph()

	for x := 0; x < n; x++ {
		node := Node{X: x, Y: 0, Z: 0}
		latticeGraph.AddVertex(node)
	}

	for y := 1; y < n; y++ {
		for x := -n + y + 1; x < n-y; x++ {
			node := Node{X: x, Y: y, Z: 0}
			latticeGraph.AddVertex(node)
		}
	}

	for z := 1; z < n; z++ {
		for y := -n + z + 1; y < n-z; y++ {
			for x := -n + z + 1 + int(math.Abs(float64(y))); x < n-z-int(math.Abs(float64(y))); x++ {
				node := Node{X: x, Y: y, Z: z}
				latticeGraph.AddVertex(node)
			}
		}
	}

	latticeGraph.ConnectAdjacentNodes()
	return latticeGraph
}

func copyMap(originalMap map[Node]int) map[Node]int {
	newMap := make(map[Node]int)
	for key, value := range originalMap {
		newMap[key] = value
	}
	return newMap
}

func compareCodes(code1, code2 []uint32) int {
	for i := 0; i < len(code1); i++ {
		if code1[i] < code2[i] {
			return -1
		} else if code1[i] > code2[i] {
			return 1
		}
	}
	return 0
}

func rotateNodes(nodes []Node, axis int) {
	if axis == 0 {
		for i := range nodes {
			nodes[i].X, nodes[i].Y = nodes[i].Y, -nodes[i].X
		}
	}
	if axis == 1 {
		for i := range nodes {
			nodes[i].X, nodes[i].Z = nodes[i].Z, -nodes[i].X
		}
	}
	if axis == 2 {
		for i := range nodes {
			nodes[i].Z, nodes[i].Y = nodes[i].Y, -nodes[i].Z
		}
	}
}

func rotateBoundary(axis int, minW, maxW, minH, maxH, minD, maxD *int) {
	if axis == 0 {
		*maxH, *maxW, *minH, *minW = -*minW, *maxH, -*maxW, *minH
	}
	if axis == 1 {
		*maxD, *maxW, *minD, *minW = -*minW, *maxD, -*maxW, *minD
	}
	if axis == 2 {
		*maxH, *maxD, *minH, *minD = -*minD, *maxH, -*maxD, *minH
	}
}

// generate unique code for polycube using method from lunnon (1971)
func getCode(code []uint32, nodes []Node, minW, maxW, minH, maxH, minD, maxD int) {
	for i := range code {
		code[i] = 0
	}
	squareSize := max(maxW-minW+1, maxH-minH+1, maxD-minD+1)
	var positionCode uint32
	var bitmap uint32
	for _, node := range nodes {
		positionCode = uint32((node.X - minW) + (maxH-node.Y)*squareSize + (maxD-node.Z)*squareSize*squareSize)
		bitmap = 2147483648 >> (positionCode % 32)
		code[positionCode/32] += bitmap
	}
}

// determines if polycube is in the "canonical" form by rotating polycube through all possible orientations
// returns 1 if canonical form for 3d cubes and 2 if also canonical form for 4d cubes (with inversion)
func isCanonical(nodes []Node) int {
	result := 0
	// get dimension of polycube for code generation
	minW, maxW, minH, maxH, minD, maxD := 0, 0, 0, 0, 0, 0
	for _, node := range nodes {
		if node.X < minW {
			minW = node.X
		}
		if node.X > maxW {
			maxW = node.X
		}
		if node.Y > maxH {
			maxH = node.Y
		}
		if node.Y < minH {
			minH = node.Y
		}
		if node.Z < minD {
			minD = node.Z
		}
		if node.Z > maxD {
			maxD = node.Z
		}
	}
	squareSize := max(maxW-minW+1, maxH-minH+1, maxD-minD+1)
	totalBits := squareSize * squareSize * squareSize

	// get unique code of polycube
	code := make([]uint32, totalBits/32+1)
	comparisonCode := make([]uint32, totalBits/32+1)
	getCode(code, nodes, minW, maxW, minH, maxH, minD, maxD)

	// compare with codes of rotated polycubes
	rotatedNodes := make([]Node, len(nodes))
	copy(rotatedNodes, nodes)

	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	rotateNodes(rotatedNodes, 2)
	rotateBoundary(2, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}
	rotateNodes(rotatedNodes, 2)
	rotateBoundary(2, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	rotateNodes(rotatedNodes, 2)
	rotateBoundary(2, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}
	result++

	// compare with codes of inverted rotated polycubes
	copy(rotatedNodes, nodes)

	for i := range rotatedNodes {
		rotatedNodes[i].X = -rotatedNodes[i].X
	}

	// regenerate dimensions
	minW, maxW, minH, maxH, minD, maxD = 0, 0, 0, 0, 0, 0
	for _, node := range rotatedNodes {
		if node.X < minW {
			minW = node.X
		}
		if node.X > maxW {
			maxW = node.X
		}
		if node.Y > maxH {
			maxH = node.Y
		}
		if node.Y < minH {
			minH = node.Y
		}
		if node.Z < minD {
			minD = node.Z
		}
		if node.Z > maxD {
			maxD = node.Z
		}
	}

	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}

	rotateNodes(rotatedNodes, 1)
	rotateBoundary(1, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	rotateNodes(rotatedNodes, 2)
	rotateBoundary(2, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}
	rotateNodes(rotatedNodes, 2)
	rotateBoundary(2, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	rotateNodes(rotatedNodes, 2)
	rotateBoundary(2, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	for i := 0; i < 4; i++ {
		getCode(comparisonCode, rotatedNodes, minW, maxW, minH, maxH, minD, maxD)
		if compareCodes(code, comparisonCode) == 1 {
			return result
		}
		rotateNodes(rotatedNodes, 0)
		rotateBoundary(0, &minW, &maxW, &minH, &maxH, &minD, &maxD)
	}
	result++
	return result
}

// CountPolycubes implements Redelmeier's algorithm to enumerate fixed polycubes w/ an additional isCanonical check
// to determine if polycube is unique when considering rotations.
func CountPolycubes(
	graph *Graph,
	depth int,
	maxSize int,
	untriedSet []Node,
	cellsAdded []Node,
	oldNeighbours map[Node]int,
	ch chan map[string][]int,
	wg *ProgressWaitGroup,
) map[string][]int {
	newUntriedSet := make([]Node, len(untriedSet))
	copy(newUntriedSet, untriedSet)

	elementCount := map[string][]int{
		"fixed":  make([]int, maxSize),
		"free3d": make([]int, maxSize),
		"free4d": make([]int, maxSize),
	}

	for len(newUntriedSet) != 0 {
		randomElement := newUntriedSet[len(newUntriedSet)-1] // step 1
		newUntriedSet = newUntriedSet[:len(newUntriedSet)-1] // step 2

		// step 3
		canonicalForms := isCanonical(append(cellsAdded, randomElement))
		if canonicalForms == 1 {
			elementCount["free3d"][depth]++

		} else if canonicalForms == 2 {
			elementCount["free3d"][depth]++
			elementCount["free4d"][depth]++
		}
		elementCount["fixed"][depth]++

		if depth+1 < maxSize { // Step 4
			var newNeighbours []Node
			for _, neighbour := range graph.GetNeighbours(randomElement) {
				value, exists := oldNeighbours[neighbour]
				if !exists || value == 0 {
					newNeighbours = append(newNeighbours, neighbour)
				}
			}
			for _, neighbour := range graph.GetNeighbours(randomElement) {
				oldNeighbours[neighbour]++
			}
			if depth == branchDepth { // branchDepth is the depth at which to start parallelization
				wg.Add(1)
				newOldNeighbours := copyMap(oldNeighbours)
				untriedSetCopy := make([]Node, len(newUntriedSet), len(newUntriedSet)+len(newNeighbours))
				copy(untriedSetCopy, newUntriedSet)
				cellsAddedCopy := make([]Node, len(cellsAdded), len(cellsAdded)+1)
				copy(cellsAddedCopy, cellsAdded)
				go CountPolycubes(graph, depth+1, maxSize, append(untriedSetCopy, newNeighbours...), append(cellsAddedCopy, randomElement), newOldNeighbours, ch, wg)
			} else {
				newCounts := CountPolycubes(graph, depth+1, maxSize, append(newUntriedSet, newNeighbours...), append(cellsAdded, randomElement), oldNeighbours, ch, wg)
				for i := range elementCount["fixed"] {
					elementCount["fixed"][i] += newCounts["fixed"][i]
					elementCount["free3d"][i] += newCounts["free3d"][i]
					elementCount["free4d"][i] += newCounts["free4d"][i]
				}
			}
			for _, neighbour := range graph.GetNeighbours(randomElement) {
				oldNeighbours[neighbour]--
			}

		} else {
			//fmt.Printf("%v\n", append(cellsAdded, randomElement))
		}
	}
	if depth == branchDepth+1 {
		ch <- elementCount
		wg.Done()
	}
	if depth == 0 {
		go func() {
			wg.Wait()
			close(ch)
		}()
		for result := range ch {
			for i := range elementCount["fixed"] {
				elementCount["fixed"][i] += result["fixed"][i]
				elementCount["free3d"][i] += result["free3d"][i]
				elementCount["free4d"][i] += result["free4d"][i]
			}
			fmt.Printf("Total counts (%.2f%%): \n\tFixed: %v\n\tFree3D: %v\n\tFree4D: %v", wg.Progress()*100, elementCount["fixed"], elementCount["free3d"], elementCount["free4d"])
			fmt.Printf("\r\033[%dA", 3)
		}
	}
	return elementCount
}

func main() {
	cpuProfile, err := os.Create("cpu.pprof")
	if err != nil {
		fmt.Println("Could not create CPU profile: ", err)
		return
	}
	defer func(cpuProfile *os.File) {
		err := cpuProfile.Close()
		if err != nil {
		}
	}(cpuProfile)

	if err := pprof.StartCPUProfile(cpuProfile); err != nil {
		fmt.Println("Could not start CPU profile: ", err)
		return
	}
	defer pprof.StopCPUProfile()

	// Check if an argument is provided
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go [n]")
		return
	}

	// Convert the argument from string to int
	n, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Printf("Invalid input: %v\n", err)
		return
	}
	startTime := time.Now()

	latticeGraph := CreateLattice(n)
	//fmt.Printf("Lattice: %v\n", latticeGraph)

	untriedSet := []Node{{X: 0, Y: 0, Z: 0}}
	oldNeighbours := make(map[Node]int)
	oldNeighbours[Node{X: 0, Y: 0, Z: 0}]++
	cellsAdded := make([]Node, 0, n)

	ch := make(chan map[string][]int)
	var wg ProgressWaitGroup

	fmt.Printf("Total counts: \n\tFixed: []\n\tFree3D: []\n\tFree4D: []")
	fmt.Printf("\r\033[%dA", 3)
	count := CountPolycubes(latticeGraph, 0, n, untriedSet, cellsAdded, oldNeighbours, ch, &wg)
	fmt.Printf("Total counts (100.00%%): \n\tFixed: %v\n\tFree3D: %v\n\tFree4D: %v\n", count["fixed"], count["free3d"], count["free4d"])
	fmt.Println("Completed in:", time.Since(startTime))
}
