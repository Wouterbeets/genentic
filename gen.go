/*
	Package gen allows you to manipulate a neural network using genetec generation of its weights.

	You create a pool of ai's using CreatePool. You must then assign a fitness function to the the pool
	This function will be used to assign scrores each generated neural network. The networks with the highest scores
	have the best chance of having their genes used in the next generartion.
*/
package gen

import (
	"encoding/gob"
	"fmt"
	"github.com/Wouterbeets/nn"
	"math/rand"
	"os"
	"sort"
	"time"
)

//Ai is a struct that holds a neural network, some varaibles to keep track of its perfomance, and a slice of genes.
type Ai struct {
	*nn.Net
	Score       float64
	GamesPlayed float64
	gene        []float64
}

//ByScore is a wrapper for a slice of ais and implements the sort interface
type ByScore []*Ai

func (ais ByScore) Len() int {
	return len(ais)
}

func (ais ByScore) Swap(i, j int) {
	ais[i], ais[j] = ais[j], ais[i]
}

func (ais ByScore) Less(i, j int) bool {
	return ais[i].Score > ais[j].Score
}

//Pool is a struct that holds a slice of Ais.
type Pool struct {
	Ai        []*Ai
	size      int // number of ais
	roullete  [100]int
	mutatePer float64          //percentage of genes to be mutated
	mStrength float64          //how much the origninal value should be mutated by
	FightFunc func([]*Ai, int) //if function is set ,it will be used instead of standard Fight func. It must fill the ais Score field.
}

//Evolve runs the loop for that aplies the genetic channges per generation
func (p *Pool) Evolve(generations int, inp [][]float64, want []float64) {
	file, err := os.OpenFile("foo.gob", os.O_RDWR|os.O_APPEND, 0660)
	if err != nil {
		file, err = os.Create("foo.gob")
	}
	saver := gob.NewEncoder(file)
	for i := 0; i < generations; i++ {
		//fmt.Println("generation", i)
		if i == 299000 {
			str := ""
			fmt.Scanln(&str)
		}
		if p.FightFunc != nil {
			p.FightFunc(p.Ai, i)
		} else if inp != nil && want != nil {
			p.Fight(inp, want)
		}
		p.Breed()
		saver.Encode(p.Ai[0].GetWeights())
	}
	sort.Sort(ByScore(p.Ai))
}

func (p *Pool) String() (str string) {
	for _, Ai := range p.Ai {
		str += fmt.Sprintln(Ai.Score)
	}
	return
}

func (p *Pool) getSumScores() (sum float64) {
	for _, Ai := range p.Ai {
		sum += Ai.Score
	}
	return
}

// makeRoullete creates an array filled with indexes of neural networks.
// the fittest neuralnetworks are more often represented in the array
// this array is later used to choose which neuralnetworks genes  from which we later randomly pick new parents
func (p *Pool) makeRoullete() {
	sum := p.getSumScores()
	i := 0
	for k, Ai := range p.Ai {
		perc := (Ai.Score / sum) * float64(100)
		for j := 0; j < int(perc); i, j = i+1, j+1 {
			p.roullete[i] = k
		}
	}
}

func (p *Pool) mutate(genes []float64) {
	gToMutate := float64(len(genes)) * p.mutatePer
	if gToMutate < 1 {
		gToMutate = 1
	}
	for i := 0; i < int(gToMutate); i++ {
		r := rand.Int()
		if r%2 == 0 {
			genes[rand.Intn(len(genes))] += rand.NormFloat64() * p.mStrength
		} else {
			genes[rand.Intn(len(genes))] -= rand.NormFloat64() * p.mStrength
		}
	}
}

func (p *Pool) makeBaby(m, f int) (baby []float64) {
	mGene := p.Ai[m].gene
	fGene := p.Ai[f].gene
	baby = make([]float64, 0, len(mGene))
	baby = append(baby, mGene[:len(mGene)/2]...)
	baby = append(baby, fGene[len(fGene)/2:]...)
	p.mutate(baby)
	return
}

//Breed make new ai from the best ai this genereation
func (p *Pool) Breed() {
	sort.Sort(ByScore(p.Ai))
	p.makeRoullete()
	for i := 1; i < p.size; i++ {
		m, f := p.roullete[rand.Intn(len(p.roullete))], p.roullete[rand.Intn(len(p.roullete))]
		if m == f {
			f = rand.Intn(p.size)
		}
		p.Ai[i].SetWeights(p.makeBaby(m, f))
		p.Ai[i].gene = p.Ai[i].GetWeights()
	}
	for i, ai := range p.Ai {
		if i < 5 {
			fmt.Println(ai.Score)
		}
		ai.Score = 0
		ai.GamesPlayed = 0
	}
	fmt.Println("\n")
}

func abs(n float64) float64 {
	if n < 0 {
		return -n
	}
	return n
}

func fitnessFunc(resp, want float64) float64 {
	if want == 0 {
		return 1 - resp
	}
	return resp
}

//Fight is the standard fitness func for whne you want to create dumb things
func (p *Pool) Fight(input [][]float64, want []float64) {
	for i, Ai := range p.Ai {
		Score := float64(0)
		for j, v := range input {
			go Ai.In(v)
			dec := Ai.Out()
			Score += fitnessFunc(dec[0], want[j])
		}
		p.Ai[i].Score = Score
	}
}

//CreatePool is the constructor of the pool of ais that are to be evolved
func CreatePool(size int, mutatePer, mStrength float64, input, hidden, layers, output int) *Pool {
	pool := &Pool{
		Ai:        make([]*Ai, size),
		size:      size,
		mutatePer: mutatePer,
		mStrength: mStrength,
		FightFunc: nil,
	}
	for i := range pool.Ai {
		pool.Ai[i] = &Ai{
			nn.NewNet(input, hidden, layers, output),
			0,
			0,
			nil,
		}
		pool.Ai[i].gene = pool.Ai[i].GetWeights()
	}
	return pool
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
